// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Creation and growth of the renderer's GPU resources.

use crate::draw::EXTERNAL_TEXTURE_SLOT_COUNT;
use crate::render::wgpu::RenderTargetConfig;
use crate::render::wgpu::passes::Programs;
use crate::render::wgpu::passes::filter::{
    FilterPairBindGroups, create_filter_original_texture_bind_group,
};
use crate::render::wgpu::passes::strip::StripTargetKind;
use crate::{
    LayersConfig, RenderSize,
    filter::FilterContext,
    gradient_cache::GradientRampCache,
    paint::{ExternalSampler, SamplerAddress, SamplerFilter},
    render::{
        Config,
        common::{GpuEncodedPaint, ScratchTexture},
    },
    schedule::Schedule,
    target::{BlendPassBindings, FilterPassBindings, LayerTextureId},
};
use alloc::vec::Vec;
use core::{fmt::Debug, num::NonZeroU64};
use hashbrown::HashMap;
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::AtlasConfig;
use vello_common::{geometry::SizeU16, tile::Tile};
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferDescriptor, BufferViewMut, COPY_BUFFER_ALIGNMENT,
    CommandEncoder, Device, Extent3d, Sampler, Texture, TextureView, TextureViewDescriptor,
    util::{DeviceExt, StagingBelt},
};

#[derive(Debug)]
pub(crate) struct StripBufferArena {
    pub(crate) buffer: Buffer,
    capacity: u64,
    cursor: u64,
}

#[derive(Debug)]
pub(crate) struct UploadStagingBelt {
    pub(crate) belt: StagingBelt,
    capacity: u64,
    frame_allocated: u64,
}

impl UploadStagingBelt {
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            belt: StagingBelt::new(device.clone(), COPY_BUFFER_ALIGNMENT),
            capacity: COPY_BUFFER_ALIGNMENT,
            frame_allocated: 0,
        }
    }

    fn prepare(&mut self, device: &Device) {
        let max_buffer_size = device.limits().max_buffer_size;
        if self.frame_allocated > self.capacity && self.capacity < max_buffer_size {
            self.capacity = (self.frame_allocated * 2).min(max_buffer_size);
            self.belt = StagingBelt::new(device.clone(), self.capacity);
        }
        self.frame_allocated = 0;
    }

    pub(crate) fn write_buffer(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &Buffer,
        offset: u64,
        size: NonZeroU64,
    ) -> BufferViewMut {
        self.frame_allocated += size.get();
        self.belt.write_buffer(encoder, target, offset, size)
    }

    pub(crate) fn upload_texture(
        &mut self,
        encoder: &mut CommandEncoder,
        texture: &Texture,
        data: &[u8],
        bytes_per_texel: u32,
    ) {
        self.upload_texture_chunks(encoder, texture, data.len(), bytes_per_texel, [data]);
    }

    pub(crate) fn upload_texture_chunks<'a>(
        &mut self,
        encoder: &mut CommandEncoder,
        texture: &Texture,
        data_len: usize,
        bytes_per_texel: u32,
        chunks: impl IntoIterator<Item = &'a [u8]>,
    ) {
        if data_len == 0 {
            return;
        }

        let bytes_per_row = texture.width() * bytes_per_texel;
        let rows = data_len.div_ceil(bytes_per_row as usize) as u32;
        let size = NonZeroU64::new(u64::from(rows) * u64::from(bytes_per_row)).unwrap();
        self.frame_allocated += size.get();
        let slice = self
            .belt
            .allocate(size, NonZeroU64::new(u64::from(bytes_per_texel)).unwrap());
        {
            let mut view = slice
                .get_mapped_range_mut()
                .expect("staging buffer must be mapped");

            let mut offset = 0;
            for chunk in chunks {
                view.slice(offset..offset + chunk.len())
                    .copy_from_slice(chunk);
                offset += chunk.len();
            }
            debug_assert_eq!(
                offset, data_len,
                "mismatched between uploaded chunks size vs provided len"
            );
        }

        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: slice.buffer(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: slice.offset(),
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            texture.as_image_copy(),
            Extent3d {
                width: texture.width(),
                height: rows,
                depth_or_array_layers: 1,
            },
        );
    }
}

pub(crate) fn create_arena_buffer(device: &Device, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some("Strips Arena"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl StripBufferArena {
    pub(crate) fn new(device: &Device) -> Self {
        Self {
            buffer: create_arena_buffer(device, 0),
            capacity: 0,
            cursor: 0,
        }
    }

    pub(crate) fn begin_frame(&mut self, device: &Device, frame_size: u64) {
        self.cursor = 0;
        if frame_size > self.capacity {
            let new_capacity = frame_size
                .max(self.capacity * 2)
                .min(device.limits().max_buffer_size);
            self.buffer = create_arena_buffer(device, new_capacity);
            self.capacity = new_capacity;
        }
    }

    pub(crate) fn alloc(&mut self, len: u64) -> u64 {
        let offset = self.cursor.next_multiple_of(COPY_BUFFER_ALIGNMENT);
        self.cursor = offset + len;
        offset
    }
}

/// Contains all GPU resources needed for rendering
#[derive(Debug)]
pub(crate) struct GpuResources {
    /// Alpha texture.
    pub(crate) alphas_texture: Texture,
    /// Maximum width or height of packed resource textures.
    pub(crate) resource_texture_dimension_2d: u32,
    /// One 2D texture per image atlas.
    pub(crate) atlas_textures: Vec<Texture>,
    /// Default view corresponding to each image atlas texture.
    pub(crate) atlas_texture_views: Vec<TextureView>,
    /// Configured atlas dimensions.
    pub(crate) atlas_size: (u16, u16),
    /// Transparent 1x1 placeholder used for unoccupied external texture slots.
    pub(crate) placeholder_external_texture_view: TextureView,
    /// Bind group used when a draw does not sample an external texture.
    pub(crate) empty_external_texture_bind_group: BindGroup,
    /// Texture for encoded paints
    pub(crate) encoded_paints_texture: Texture,
    /// Bind group for encoded paints
    pub(crate) encoded_paints_bind_group: BindGroup,
    /// Texture for gradient lookup table
    pub(crate) gradient_texture: Texture,
    /// Bind group for gradient texture
    pub(crate) gradient_bind_group: BindGroup,
    /// Texture holding serialized `GpuFilterData` for all filter layers.
    pub(crate) filter_data_texture: Texture,
    /// Bind group for the filter data texture.
    pub(crate) filter_base_bind_group: BindGroup,
    /// Intermediate texture size.
    pub(crate) texture_size: SizeU16,
    /// Root config buffer.
    pub(crate) view_config_buffer: Buffer,
    /// Layer config buffer.
    pub(crate) layer_config_buffer: Buffer,

    /// Layer textures by parity.
    pub(crate) layer_textures: [Vec<TextureView>; 2],
    /// Scratch texture.
    pub(crate) scratch_texture: Option<ScratchTexture<TextureView>>,
    /// Filter bind group.
    pub(crate) filter_original_bind_group: BindGroup,
    /// Copy bind group.
    pub(crate) scratch_copy_bind_group: BindGroup,
    /// Native image samplers shared by external texture slots.
    pub(crate) external_samplers: HashMap<ExternalSampler, Sampler>,
    /// Strip bind group cache.
    pub(crate) strip_layer_bind_groups:
        HashMap<(StripTargetKind, Option<LayerTextureId>), BindGroup>,
    /// Blend bind group cache.
    pub(crate) blend_layer_bind_groups: HashMap<BlendPassBindings, BindGroup>,
    /// Filter bind group cache.
    pub(crate) filter_pair_bind_groups: HashMap<FilterPassBindings, FilterPairBindGroups>,
    /// Arena holding all [`GpuStrip`](crate::GpuStrip) data.
    pub(crate) strips_arena: StripBufferArena,
    /// Shared staging memory for buffer and texture uploads.
    pub(crate) staging_belt: UploadStagingBelt,
    /// Dimensions of the rendering target
    pub(crate) render_size: RenderSize,
    /// Offset of the scene within the rendering target.
    pub(crate) scene_offset: (i32, i32),
    /// Whether the depth buffer has been cleared this frame.
    pub(crate) depth_cleared_this_frame: bool,
}

impl GpuResources {
    pub(crate) fn new(
        device: &Device,
        programs: &Programs,
        image_cache: &ImageCache,
        render_target_config: &RenderTargetConfig,
        layer_config: LayersConfig,
        resource_texture_dimension_2d: u32,
    ) -> Self {
        let texture_size = layer_config.min_texture_size;

        // The empty bind group needs a sampler even before any external image is drawn.
        let mut external_samplers = HashMap::new();
        external_samplers.insert(
            ExternalSampler::DEFAULT,
            Self::create_external_sampler(device, ExternalSampler::DEFAULT),
        );

        let layer_textures: [Vec<TextureView>; 2] = core::array::from_fn(|_| Vec::new());
        let scratch_texture = None;
        let placeholder_external_texture_view = Self::create_placeholder_external_texture(device);
        let filter_original_bind_group = create_filter_original_texture_bind_group(
            device,
            &programs.filter.input_bind_group_layouts[1],
            &placeholder_external_texture_view,
        );
        let layer_config_buffer = Self::create_config_buffer(
            device,
            u32::from(texture_size.width()),
            u32::from(texture_size.height()),
            resource_texture_dimension_2d,
        );

        const INITIAL_ALPHA_TEXTURE_HEIGHT: u32 = 1;
        let alphas_texture = Self::create_alphas_texture(
            device,
            resource_texture_dimension_2d,
            INITIAL_ALPHA_TEXTURE_HEIGHT,
        );
        let view_config_buffer = Self::create_config_buffer(
            device,
            u32::from(render_target_config.width),
            u32::from(render_target_config.height),
            resource_texture_dimension_2d,
        );

        let AtlasConfig {
            atlas_size: (atlas_width, atlas_height),
            initial_atlas_count,
            ..
        } = image_cache.atlas_manager().config();
        let atlas_size = (*atlas_width, *atlas_height);
        let atlas_textures: Vec<_> = (0..*initial_atlas_count)
            .map(|_| Self::create_atlas_texture(device, *atlas_width, *atlas_height))
            .collect();
        let atlas_texture_views = atlas_textures
            .iter()
            .map(|texture| texture.create_view(&TextureViewDescriptor::default()))
            .collect();
        let empty_external_texture_bind_group =
            Programs::create_external_texture_bind_group(
                device,
                &programs.strip.external_texture_bind_group_layout,
                [&placeholder_external_texture_view; EXTERNAL_TEXTURE_SLOT_COUNT],
                [&external_samplers[&ExternalSampler::DEFAULT]; EXTERNAL_TEXTURE_SLOT_COUNT],
                programs.strip.wgpu_external_textures.then_some(
                    wgpu::BindingResource::TextureView(&placeholder_external_texture_view),
                ),
            );

        const INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT: u32 = 1;
        let encoded_paints_texture = Self::create_encoded_paints_texture(
            device,
            resource_texture_dimension_2d,
            INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT,
        );
        let encoded_paints_bind_group = Self::create_encoded_paints_bind_group(
            device,
            &programs.encoded_paints_bind_group_layout,
            &encoded_paints_texture.create_view(&TextureViewDescriptor::default()),
        );

        const INITIAL_GRADIENT_TEXTURE_HEIGHT: u32 = 1;
        let gradient_texture = Self::create_gradient_texture(
            device,
            resource_texture_dimension_2d,
            INITIAL_GRADIENT_TEXTURE_HEIGHT,
        );
        let gradient_bind_group = Self::create_gradient_bind_group(
            device,
            &programs.gradient_bind_group_layout,
            &gradient_texture.create_view(&TextureViewDescriptor::default()),
        );

        // TODO: We really should deduplicate handling of this this with encoded paints texture.
        const INITIAL_FILTER_TEXTURE_HEIGHT: u32 = 1;
        let filter_data_texture = Self::create_filter_data_texture(
            device,
            resource_texture_dimension_2d,
            INITIAL_FILTER_TEXTURE_HEIGHT,
        );
        let filter_base_bind_group = Self::create_filter_base_bind_group(
            device,
            &programs.filter.bind_group_layout,
            &filter_data_texture.create_view(&TextureViewDescriptor::default()),
        );
        let scratch_copy_bind_group = Programs::create_copy_bind_group(
            device,
            &programs.copy.bind_group_layout,
            &placeholder_external_texture_view,
        );

        Self {
            layer_textures,
            scratch_texture,
            filter_original_bind_group,
            scratch_copy_bind_group,
            layer_config_buffer,
            alphas_texture,
            resource_texture_dimension_2d,
            atlas_textures,
            atlas_texture_views,
            atlas_size,
            placeholder_external_texture_view,
            empty_external_texture_bind_group,
            encoded_paints_texture,
            encoded_paints_bind_group,
            gradient_texture,
            gradient_bind_group,
            filter_data_texture,
            filter_base_bind_group,
            texture_size,
            view_config_buffer,
            external_samplers,
            strip_layer_bind_groups: HashMap::new(),
            blend_layer_bind_groups: HashMap::new(),
            filter_pair_bind_groups: HashMap::new(),
            strips_arena: StripBufferArena::new(device),
            staging_belt: UploadStagingBelt::new(device),
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            scene_offset: (0, 0),
            depth_cleared_this_frame: false,
        }
    }

    pub(crate) fn layer_views(&self, layer_ids: [LayerTextureId; 2]) -> [&TextureView; 2] {
        layer_ids.map(|id| self.layer_view(id))
    }

    pub(crate) fn layer_view(&self, id: LayerTextureId) -> &TextureView {
        self.layer_textures[id.texture_parity.get_parity()]
            .get(usize::from(id.page_index))
            .unwrap()
    }

    pub(crate) fn scratch_view(&self) -> &TextureView {
        self.scratch_texture.as_ref().unwrap().get()
    }
}

pub(crate) const SIZE_OF_CONFIG: NonZeroU64 = NonZeroU64::new(size_of::<Config>() as u64).unwrap();

impl GpuResources {
    fn create_intermediate_texture(
        device: &Device,
        size: SizeU16,
        label: &'static str,
    ) -> TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: u32::from(size.width()),
                height: u32::from(size.height()),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        texture.create_view(&TextureViewDescriptor::default())
    }

    pub(crate) fn prepare_intermediate_textures(
        &mut self,
        device: &Device,
        programs: &Programs,
        schedule: &Schedule,
    ) {
        let requirements = schedule.intermediate_texture_requirements();
        let texture_size = requirements.size;
        let current_size = self.texture_size;
        let allocations = requirements.allocations;
        let layer_pages = allocations.layer_pages;
        let scratch_required = allocations.scratch;
        let size_changed = current_size != texture_size;

        if size_changed {
            self.layer_config_buffer = Self::create_config_buffer(
                device,
                u32::from(texture_size.width()),
                u32::from(texture_size.height()),
                self.resource_texture_dimension_2d,
            );
        }

        for (index, textures) in self.layer_textures.iter_mut().enumerate() {
            let required_page_count = layer_pages[index];
            // Note: Currently, `texture_size` only grows across frames, so if this condition
            // is true it means that the new required size is larger than what we currently have.
            if size_changed {
                for texture in textures.iter_mut() {
                    *texture = Self::create_intermediate_texture(
                        device,
                        texture_size,
                        "Layer Intermediate Texture",
                    );
                }
            }

            textures.extend((textures.len()..required_page_count).map(|_| {
                Self::create_intermediate_texture(
                    device,
                    texture_size,
                    "Layer Intermediate Texture",
                )
            }));
        }
        let scratch_changed = (self.scratch_texture.is_some() && size_changed)
            || (self.scratch_texture.is_none() && scratch_required);
        if scratch_changed {
            self.scratch_texture = Some(ScratchTexture::new(Self::create_intermediate_texture(
                device,
                texture_size,
                "Scratch Texture",
            )));
        }

        self.texture_size = texture_size;

        if size_changed {
            self.clear_layer_bind_group_caches();
        }

        if scratch_changed {
            self.update_scratch_bind_groups(device, programs);
        }
    }

    fn clear_layer_bind_group_caches(&mut self) {
        self.strip_layer_bind_groups.clear();
        self.blend_layer_bind_groups.clear();
        self.filter_pair_bind_groups.clear();
    }

    fn update_scratch_bind_groups(&mut self, device: &Device, programs: &Programs) {
        let scratch_view = self.scratch_view();
        let filter_original_bind_group = create_filter_original_texture_bind_group(
            device,
            &programs.filter.input_bind_group_layouts[1],
            scratch_view,
        );
        let scratch_copy_bind_group = Programs::create_copy_bind_group(
            device,
            &programs.copy.bind_group_layout,
            scratch_view,
        );

        self.filter_original_bind_group = filter_original_bind_group;
        self.scratch_copy_bind_group = scratch_copy_bind_group;
    }

    pub(crate) fn create_config_buffer(
        device: &Device,
        width: u32,
        height: u32,
        alpha_texture_width: u32,
    ) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&Config {
                width,
                height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: alpha_texture_width.trailing_zeros(),
                encoded_paints_tex_width_bits: alpha_texture_width.trailing_zeros(),
                strip_offset_x: 0,
                strip_offset_y: 0,
                negate_ndc: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub(crate) fn create_alphas_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Alpha Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    pub(crate) fn create_atlas_texture(device: &Device, width: u16, height: u16) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atlas Texture"),
            size: Extent3d {
                width: u32::from(width),
                height: u32::from(height),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    /// 1×1 placeholder bound to the external-texture slot when no real external texture is in
    /// use. `Rgba8Unorm` is accepted by both the `Texture` and `ExternalTexture` binding types.
    pub(crate) fn create_placeholder_external_texture(device: &Device) -> TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Placeholder External Texture"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        texture.create_view(&TextureViewDescriptor::default())
    }

    pub(crate) fn create_external_sampler(device: &Device, key: ExternalSampler) -> Sampler {
        let address_mode = |address| match address {
            SamplerAddress::Pad => wgpu::AddressMode::ClampToEdge,
            SamplerAddress::Repeat => wgpu::AddressMode::Repeat,
            SamplerAddress::Reflect => wgpu::AddressMode::MirrorRepeat,
        };
        let filter = match key.filter {
            SamplerFilter::Nearest => wgpu::FilterMode::Nearest,
            SamplerFilter::Linear => wgpu::FilterMode::Linear,
        };
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("External Texture Sampler"),
            address_mode_u: address_mode(key.address_x),
            address_mode_v: address_mode(key.address_y),
            mag_filter: filter,
            min_filter: filter,
            ..Default::default()
        })
    }

    pub(crate) fn create_encoded_paints_texture(
        device: &Device,
        width: u32,
        height: u32,
    ) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Encoded Paints Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    pub(crate) fn create_encoded_paints_bind_group(
        device: &Device,
        encoded_paints_bind_group_layout: &BindGroupLayout,
        encoded_paints_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Encoded Paints Bind Group"),
            layout: encoded_paints_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(encoded_paints_texture_view),
            }],
        })
    }

    pub(crate) fn create_gradient_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gradient Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    pub(crate) fn create_gradient_bind_group(
        device: &Device,
        gradient_bind_group_layout: &BindGroupLayout,
        gradient_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gradient Bind Group"),
            layout: gradient_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(gradient_texture_view),
            }],
        })
    }

    /// Prepare GPU buffers for rendering, given alphas.
    ///
    /// Specifically, updates the alpha texture with `alphas` and the config buffer when
    /// the rendering size changes.
    pub(crate) fn prepare(
        &mut self,
        device: &Device,
        programs: &Programs,
        encoder: &mut CommandEncoder,
        gradient_cache: &mut GradientRampCache,
        encoded_paints: &[GpuEncodedPaint],
        alphas: &[u8],
        new_render_size: &RenderSize,
        scene_offset: (i32, i32),
        paint_idxs: &[u32],
        filter_context: &FilterContext,
    ) {
        let resource_texture_dimension_2d = self.resource_texture_dimension_2d;
        self.staging_belt.prepare(device);
        self.maybe_resize_alphas_tex(device, resource_texture_dimension_2d, alphas.len());
        self.maybe_resize_encoded_paints_tex(
            device,
            programs,
            resource_texture_dimension_2d,
            paint_idxs,
        );
        self.maybe_resize_filter_tex(
            device,
            programs,
            resource_texture_dimension_2d,
            filter_context,
        );
        self.maybe_update_config_buffer(
            encoder,
            resource_texture_dimension_2d,
            new_render_size,
            scene_offset,
        );

        self.upload_alpha_texture(encoder, alphas);
        self.upload_encoded_paints_texture(encoder, encoded_paints, paint_idxs);
        self.upload_filter_texture(encoder, filter_context);

        if gradient_cache.has_changed() {
            self.maybe_resize_gradient_tex(
                device,
                programs,
                resource_texture_dimension_2d,
                gradient_cache,
            );
            self.upload_gradient_texture(encoder, gradient_cache);
            gradient_cache.mark_synced();
        }
    }

    /// Update the alpha texture size if needed.
    fn maybe_resize_alphas_tex(
        &mut self,
        device: &Device,
        resource_texture_dimension_2d: u32,
        alphas_len: usize,
    ) {
        let required_alpha_height = u32::try_from(alphas_len)
            .unwrap()
            // There are 16 1-byte alpha values per texel.
            .div_ceil(resource_texture_dimension_2d << 4);
        debug_assert!(
            self.alphas_texture.width() == resource_texture_dimension_2d,
            "Alpha texture width must match resource texture dimensions"
        );
        let current_alpha_height = self.alphas_texture.height();
        if required_alpha_height > current_alpha_height {
            // We need to resize the alpha texture to fit the new alpha data.
            assert!(
                required_alpha_height <= resource_texture_dimension_2d,
                "Alpha texture height exceeds resource texture dimensions"
            );

            // The alpha texture encodes 16 1-byte alpha values per texel, with 4 alpha values packed in each channel
            let alphas_texture = Self::create_alphas_texture(
                device,
                resource_texture_dimension_2d,
                required_alpha_height,
            );
            self.alphas_texture = alphas_texture;
            self.strip_layer_bind_groups.clear();
            self.blend_layer_bind_groups.clear();
        }
    }

    /// Update the encoded paints texture size if needed.
    fn maybe_resize_encoded_paints_tex(
        &mut self,
        device: &Device,
        programs: &Programs,
        resource_texture_dimension_2d: u32,
        paint_idxs: &[u32],
    ) {
        let required_texels = paint_idxs.last().unwrap();
        let required_encoded_paints_height =
            required_texels.div_ceil(resource_texture_dimension_2d);
        debug_assert!(
            self.encoded_paints_texture.width() == resource_texture_dimension_2d,
            "Encoded paints texture width must match resource texture dimensions"
        );
        let current_encoded_paints_height = self.encoded_paints_texture.height();
        if required_encoded_paints_height > current_encoded_paints_height {
            assert!(
                required_encoded_paints_height <= resource_texture_dimension_2d,
                "Encoded paints texture height exceeds resource texture dimensions"
            );
            let encoded_paints_texture = Self::create_encoded_paints_texture(
                device,
                resource_texture_dimension_2d,
                required_encoded_paints_height,
            );
            self.encoded_paints_texture = encoded_paints_texture;

            // Since the encoded paints texture has changed, we need to update the strip bind groups.
            self.encoded_paints_bind_group = Self::create_encoded_paints_bind_group(
                device,
                &programs.encoded_paints_bind_group_layout,
                &self
                    .encoded_paints_texture
                    .create_view(&TextureViewDescriptor::default()),
            );
        }
    }

    /// Update the gradient texture size if needed.
    fn maybe_resize_gradient_tex(
        &mut self,
        device: &Device,
        programs: &Programs,
        resource_texture_dimension_2d: u32,
        gradient_cache: &GradientRampCache,
    ) {
        let gradient_pixels = (gradient_cache.luts_size() / 4) as u32; // 4 bytes per RGBA8 pixel
        let required_gradient_height = gradient_pixels.div_ceil(resource_texture_dimension_2d);
        debug_assert!(
            self.gradient_texture.width() == resource_texture_dimension_2d,
            "Gradient texture width must match resource texture dimensions"
        );
        let current_gradient_height = self.gradient_texture.height();
        if required_gradient_height > current_gradient_height {
            assert!(
                required_gradient_height <= resource_texture_dimension_2d,
                "Gradient texture height exceeds resource texture dimensions"
            );
            let gradient_texture = Self::create_gradient_texture(
                device,
                resource_texture_dimension_2d,
                required_gradient_height,
            );
            self.gradient_texture = gradient_texture;

            // Since the gradient texture has changed, we need to update the gradient bind group.
            self.gradient_bind_group = Self::create_gradient_bind_group(
                device,
                &programs.gradient_bind_group_layout,
                &self
                    .gradient_texture
                    .create_view(&TextureViewDescriptor::default()),
            );
        }
    }

    /// Update config buffer if dimensions or the scene offset changed.
    fn maybe_update_config_buffer(
        &mut self,
        encoder: &mut CommandEncoder,
        resource_texture_dimension_2d: u32,
        new_render_size: &RenderSize,
        scene_offset: (i32, i32),
    ) {
        if self.render_size != *new_render_size || self.scene_offset != scene_offset {
            let config = Config {
                width: u32::from(new_render_size.width),
                height: u32::from(new_render_size.height),
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: resource_texture_dimension_2d.trailing_zeros(),
                encoded_paints_tex_width_bits: resource_texture_dimension_2d.trailing_zeros(),
                strip_offset_x: scene_offset.0,
                strip_offset_y: scene_offset.1,
                negate_ndc: 0,
            };
            let mut buffer = self.staging_belt.write_buffer(
                encoder,
                &self.view_config_buffer,
                0,
                SIZE_OF_CONFIG,
            );
            buffer.copy_from_slice(bytemuck::bytes_of(&config));

            self.render_size = *new_render_size;
            self.scene_offset = scene_offset;
        }
    }

    /// Create any newly allocated atlas textures.
    pub(crate) fn maybe_create_atlas_textures(
        &mut self,
        device: &Device,
        required_atlas_count: u32,
    ) {
        let (width, height) = self.atlas_size;
        while self.atlas_textures.len() < required_atlas_count as usize {
            let texture = Self::create_atlas_texture(device, width, height);
            let texture_view = texture.create_view(&TextureViewDescriptor::default());
            self.atlas_textures.push(texture);
            self.atlas_texture_views.push(texture_view);
        }
    }

    /// Upload alpha data to the texture.
    fn upload_alpha_texture(&mut self, encoder: &mut CommandEncoder, alphas: &[u8]) {
        if alphas.is_empty() {
            return;
        }

        let texture = &self.alphas_texture;
        self.staging_belt
            .upload_texture(encoder, texture, alphas, 16);
    }

    /// Upload encoded paints to the texture.
    fn upload_encoded_paints_texture(
        &mut self,
        encoder: &mut CommandEncoder,
        encoded_paints: &[GpuEncodedPaint],
        paint_idxs: &[u32],
    ) {
        if encoded_paints.is_empty() {
            return;
        }

        let texture = &self.encoded_paints_texture;
        let data_len = *paint_idxs.last().unwrap() as usize * 16;
        self.staging_belt.upload_texture_chunks(
            encoder,
            texture,
            data_len,
            16,
            encoded_paints.iter().map(GpuEncodedPaint::as_bytes),
        );
    }

    /// Upload gradient data to the texture.
    fn upload_gradient_texture(
        &mut self,
        encoder: &mut CommandEncoder,
        gradient_cache: &mut GradientRampCache,
    ) {
        if gradient_cache.is_empty() {
            return;
        }

        self.staging_belt
            .upload_texture(encoder, &self.gradient_texture, gradient_cache.luts(), 4);
    }
}
