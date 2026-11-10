// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Creation and growth of the renderer's GPU resources.

use crate::draw::EXTERNAL_TEXTURE_SLOT_COUNT;
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
        common::{DataTexture, DataTextureFormat, GpuEncodedPaint, ScratchTexture},
    },
    schedule::Schedule,
    target::{BlendPassBindings, FilterPassBindings, LayerTextureId},
};
use crate::{RenderError, ResourceTexture};
use alloc::vec::Vec;
use core::{borrow::Borrow, fmt::Debug, num::NonZeroU64};
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

impl From<DataTextureFormat> for wgpu::TextureFormat {
    fn from(format: DataTextureFormat) -> Self {
        match format {
            DataTextureFormat::Rgba8 => Self::Rgba8Unorm,
            DataTextureFormat::Rgba32Uint => Self::Rgba32Uint,
        }
    }
}

/// A texture and a bind group that binds it as the only entry of its layout.
#[derive(Debug)]
pub(crate) struct BoundTexture {
    pub(crate) texture: Texture,
    pub(crate) bind_group: BindGroup,
}

impl Borrow<Texture> for BoundTexture {
    fn borrow(&self) -> &Texture {
        &self.texture
    }
}

impl<T: Borrow<Texture>> DataTexture<T> {
    /// Upload `data` to the texture, starting at its first row.
    pub(crate) fn upload(
        &self,
        staging_belt: &mut UploadStagingBelt,
        encoder: &mut CommandEncoder,
        data: &[u8],
    ) {
        staging_belt.upload_texture(
            encoder,
            self.texture().borrow(),
            data,
            self.format().bytes_per_texel(),
        );
    }

    /// `texels` must match the total size of `chunks`.
    pub(crate) fn upload_chunks<'a>(
        &self,
        staging_belt: &mut UploadStagingBelt,
        encoder: &mut CommandEncoder,
        texels: u32,
        chunks: impl IntoIterator<Item = &'a [u8]>,
    ) {
        let bytes_per_texel = self.format().bytes_per_texel();
        staging_belt.upload_texture_chunks(
            encoder,
            self.texture().borrow(),
            texels as usize * bytes_per_texel as usize,
            bytes_per_texel,
            chunks,
        );
    }
}

/// Contains all GPU resources needed for rendering
#[derive(Debug)]
pub(crate) struct GpuResources {
    /// Alpha texture.
    pub(crate) alphas_texture: DataTexture<Texture>,
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
    pub(crate) encoded_paints_texture: DataTexture<BoundTexture>,
    /// Texture for gradient lookup table
    pub(crate) gradient_texture: DataTexture<BoundTexture>,
    /// Texture holding serialized `GpuFilterData` for all filter layers.
    pub(crate) filter_data_texture: DataTexture<BoundTexture>,
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
        render_size: RenderSize,
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

        let alphas_texture = Self::new_data_texture(
            device,
            "Alpha Texture",
            ResourceTexture::Alphas,
            DataTextureFormat::Rgba32Uint,
            resource_texture_dimension_2d,
        );
        let view_config_buffer = Self::create_config_buffer(
            device,
            u32::from(render_size.width),
            u32::from(render_size.height),
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

        let encoded_paints_texture = Self::new_bound_data_texture(
            device,
            "Encoded Paints",
            &programs.encoded_paints_bind_group_layout,
            ResourceTexture::EncodedPaints,
            DataTextureFormat::Rgba32Uint,
            resource_texture_dimension_2d,
        );
        let gradient_texture = Self::new_bound_data_texture(
            device,
            "Gradient",
            &programs.gradient_bind_group_layout,
            ResourceTexture::Gradients,
            DataTextureFormat::Rgba8,
            resource_texture_dimension_2d,
        );
        let filter_data_texture = Self::new_bound_data_texture(
            device,
            "Filter Data",
            &programs.filter.bind_group_layout,
            ResourceTexture::FilterData,
            DataTextureFormat::Rgba32Uint,
            resource_texture_dimension_2d,
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
            gradient_texture,
            filter_data_texture,
            texture_size,
            view_config_buffer,
            external_samplers,
            strip_layer_bind_groups: HashMap::new(),
            blend_layer_bind_groups: HashMap::new(),
            filter_pair_bind_groups: HashMap::new(),
            strips_arena: StripBufferArena::new(device),
            staging_belt: UploadStagingBelt::new(device),
            render_size,
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

    fn new_data_texture(
        device: &Device,
        label: &'static str,
        kind: ResourceTexture,
        format: DataTextureFormat,
        width: u32,
    ) -> DataTexture<Texture> {
        DataTexture::new(
            Self::create_data_texture(device, label, format, width, 1),
            kind,
            format,
            width,
            1,
        )
    }

    fn new_bound_data_texture(
        device: &Device,
        label: &'static str,
        layout: &BindGroupLayout,
        kind: ResourceTexture,
        format: DataTextureFormat,
        width: u32,
    ) -> DataTexture<BoundTexture> {
        DataTexture::new(
            Self::create_bound_data_texture(device, label, layout, format, width, 1),
            kind,
            format,
            width,
            1,
        )
    }

    pub(crate) fn create_bound_data_texture(
        device: &Device,
        label: &'static str,
        layout: &BindGroupLayout,
        format: DataTextureFormat,
        width: u32,
        height: u32,
    ) -> BoundTexture {
        let texture = Self::create_data_texture(device, label, format, width, height);
        let view = texture.create_view(&TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        BoundTexture {
            texture,
            bind_group,
        }
    }

    fn create_data_texture(
        device: &Device,
        label: &'static str,
        format: DataTextureFormat,
        width: u32,
        height: u32,
    ) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.into(),
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
    ) -> Result<(), RenderError> {
        let resource_texture_dimension_2d = self.resource_texture_dimension_2d;
        self.staging_belt.prepare(device);
        self.maybe_resize_alphas_tex(device, alphas.len())?;
        self.maybe_resize_encoded_paints_tex(device, programs, paint_idxs)?;
        self.maybe_resize_filter_tex(device, programs, filter_context)?;
        if gradient_cache.has_changed() {
            self.maybe_resize_gradient_tex(device, programs, gradient_cache)?;
        }
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
            self.upload_gradient_texture(encoder, gradient_cache);
            gradient_cache.mark_synced();
        }

        Ok(())
    }

    /// Update the alpha texture size if needed.
    fn maybe_resize_alphas_tex(
        &mut self,
        device: &Device,
        alphas_len: usize,
    ) -> Result<(), RenderError> {
        let resized = self
            .alphas_texture
            .ensure_bytes(alphas_len, |format, width, height| {
                // The alpha texture encodes 16 1-byte alpha values per texel, with 4 alpha values packed in each channel
                Ok::<_, RenderError>(Self::create_data_texture(
                    device,
                    "Alpha Texture",
                    format,
                    width,
                    height,
                ))
            })?;
        if resized {
            self.strip_layer_bind_groups.clear();
            self.blend_layer_bind_groups.clear();
        }

        Ok(())
    }

    /// Update the encoded paints texture size if needed.
    fn maybe_resize_encoded_paints_tex(
        &mut self,
        device: &Device,
        programs: &Programs,
        paint_idxs: &[u32],
    ) -> Result<(), RenderError> {
        let layout = &programs.encoded_paints_bind_group_layout;
        self.encoded_paints_texture.ensure_texels(
            *paint_idxs.last().unwrap(),
            |format, width, height| {
                Ok::<_, RenderError>(Self::create_bound_data_texture(
                    device,
                    "Encoded Paints",
                    layout,
                    format,
                    width,
                    height,
                ))
            },
        )?;

        Ok(())
    }

    /// Update the gradient texture size if needed.
    fn maybe_resize_gradient_tex(
        &mut self,
        device: &Device,
        programs: &Programs,
        gradient_cache: &GradientRampCache,
    ) -> Result<(), RenderError> {
        let layout = &programs.gradient_bind_group_layout;
        self.gradient_texture.ensure_bytes(
            gradient_cache.luts_size(),
            |format, width, height| {
                Ok::<_, RenderError>(Self::create_bound_data_texture(
                    device, "Gradient", layout, format, width, height,
                ))
            },
        )?;

        Ok(())
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

        self.alphas_texture
            .upload(&mut self.staging_belt, encoder, alphas);
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

        self.encoded_paints_texture.upload_chunks(
            &mut self.staging_belt,
            encoder,
            *paint_idxs.last().unwrap(),
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

        self.gradient_texture
            .upload(&mut self.staging_belt, encoder, gradient_cache.luts());
    }
}
