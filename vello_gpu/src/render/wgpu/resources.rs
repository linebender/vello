// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Creation and growth of the renderer's GPU resources.

use crate::render::wgpu::passes::Programs;
use crate::render::wgpu::passes::filter::create_filter_original_texture_bind_group;
use crate::{
    RenderSize,
    filter::FilterContext,
    gradient_cache::GradientRampCache,
    paint::{ExternalSampler, SamplerAddress, SamplerFilter},
    render::{
        Config,
        common::{GpuEncodedPaint, ScratchTexture},
    },
    schedule::Schedule,
    target::LayerTextureId,
};
use alloc::vec::Vec;
use core::{fmt::Debug, num::NonZeroU64};
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
        if data.is_empty() {
            return;
        }

        let bytes_per_row = texture.width() * bytes_per_texel;
        let rows = data.len().div_ceil(bytes_per_row as usize) as u32;
        let size = NonZeroU64::new(u64::from(rows) * u64::from(bytes_per_row)).unwrap();
        self.frame_allocated += size.get();
        let slice = self
            .belt
            .allocate(size, NonZeroU64::new(u64::from(bytes_per_texel)).unwrap());
        {
            let mut view = slice
                .get_mapped_range_mut()
                .expect("staging buffer must be mapped");

            // Any data beyond `data.len` will never be sampled, so we don't need to
            // pad it out and can just leave whatever was there already.
            view.slice(..data.len()).copy_from_slice(data);
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
}

impl GpuResources {
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

impl Programs {
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

    pub(crate) fn prepare_intermediate_textures(&mut self, device: &Device, schedule: &Schedule) {
        let requirements = schedule.intermediate_texture_requirements();
        let texture_size = requirements.size;
        let current_size = self.resources.texture_size;
        let allocations = requirements.allocations;
        let layer_pages = allocations.layer_pages;
        let scratch_required = allocations.scratch;
        let size_changed = current_size != texture_size;

        if size_changed {
            self.resources.layer_config_buffer = Self::create_config_buffer(
                device,
                u32::from(texture_size.width()),
                u32::from(texture_size.height()),
                self.resources.resource_texture_dimension_2d,
            );
        }

        for (index, textures) in self.resources.layer_textures.iter_mut().enumerate() {
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
        let scratch_changed = (self.resources.scratch_texture.is_some() && size_changed)
            || (self.resources.scratch_texture.is_none() && scratch_required);
        if scratch_changed {
            self.resources.scratch_texture = Some(ScratchTexture::new(
                Self::create_intermediate_texture(device, texture_size, "Scratch Texture"),
            ));
        }

        self.resources.texture_size = texture_size;

        if size_changed {
            self.clear_layer_bind_group_caches();
        }

        if scratch_changed {
            self.update_scratch_bind_groups(device);
        }
    }

    fn clear_layer_bind_group_caches(&mut self) {
        self.strip_layer_bind_groups.clear();
        self.blend_layer_bind_groups.clear();
        self.filter_pair_bind_groups.clear();
    }

    fn update_scratch_bind_groups(&mut self, device: &Device) {
        let scratch_view = self.resources.scratch_view();
        let filter_original_bind_group = create_filter_original_texture_bind_group(
            device,
            &self.filter_input_bind_group_layouts[1],
            scratch_view,
        );
        let scratch_copy_bind_group =
            Self::create_copy_bind_group(device, &self.copy_bind_group_layout, scratch_view);

        self.resources.filter_original_bind_group = filter_original_bind_group;
        self.resources.scratch_copy_bind_group = scratch_copy_bind_group;
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
        encoder: &mut CommandEncoder,
        gradient_cache: &mut GradientRampCache,
        encoded_paints: &[GpuEncodedPaint],
        alphas: &[u8],
        new_render_size: &RenderSize,
        scene_offset: (i32, i32),
        paint_idxs: &[u32],
        filter_context: &FilterContext,
    ) {
        let resource_texture_dimension_2d = self.resources.resource_texture_dimension_2d;
        self.staging_belt.prepare(device);
        self.maybe_resize_alphas_tex(device, resource_texture_dimension_2d, alphas.len());
        self.maybe_resize_encoded_paints_tex(device, resource_texture_dimension_2d, paint_idxs);
        self.maybe_resize_filter_tex(device, resource_texture_dimension_2d, filter_context);
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
            self.maybe_resize_gradient_tex(device, resource_texture_dimension_2d, gradient_cache);
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
            self.resources.alphas_texture.width() == resource_texture_dimension_2d,
            "Alpha texture width must match resource texture dimensions"
        );
        let current_alpha_height = self.resources.alphas_texture.height();
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
            self.resources.alphas_texture = alphas_texture;
            self.strip_layer_bind_groups.clear();
            self.blend_layer_bind_groups.clear();
        }
    }

    /// Update the encoded paints texture size if needed.
    fn maybe_resize_encoded_paints_tex(
        &mut self,
        device: &Device,
        resource_texture_dimension_2d: u32,
        paint_idxs: &[u32],
    ) {
        let required_texels = paint_idxs.last().unwrap();
        let required_encoded_paints_height =
            required_texels.div_ceil(resource_texture_dimension_2d);
        debug_assert!(
            self.resources.encoded_paints_texture.width() == resource_texture_dimension_2d,
            "Encoded paints texture width must match resource texture dimensions"
        );
        let current_encoded_paints_height = self.resources.encoded_paints_texture.height();
        if required_encoded_paints_height > current_encoded_paints_height {
            assert!(
                required_encoded_paints_height <= resource_texture_dimension_2d,
                "Encoded paints texture height exceeds resource texture dimensions"
            );
            let required_encoded_paints_size =
                (resource_texture_dimension_2d * required_encoded_paints_height) << 4;
            self.encoded_paints_data
                .resize(required_encoded_paints_size as usize, 0);
            let encoded_paints_texture = Self::create_encoded_paints_texture(
                device,
                resource_texture_dimension_2d,
                required_encoded_paints_height,
            );
            self.resources.encoded_paints_texture = encoded_paints_texture;

            // Since the encoded paints texture has changed, we need to update the strip bind groups.
            self.resources.encoded_paints_bind_group = Self::create_encoded_paints_bind_group(
                device,
                &self.encoded_paints_bind_group_layout,
                &self
                    .resources
                    .encoded_paints_texture
                    .create_view(&TextureViewDescriptor::default()),
            );
        }
    }

    /// Update the gradient texture size if needed.
    fn maybe_resize_gradient_tex(
        &mut self,
        device: &Device,
        resource_texture_dimension_2d: u32,
        gradient_cache: &GradientRampCache,
    ) {
        let gradient_pixels = (gradient_cache.luts_size() / 4) as u32; // 4 bytes per RGBA8 pixel
        let required_gradient_height = gradient_pixels.div_ceil(resource_texture_dimension_2d);
        debug_assert!(
            self.resources.gradient_texture.width() == resource_texture_dimension_2d,
            "Gradient texture width must match resource texture dimensions"
        );
        let current_gradient_height = self.resources.gradient_texture.height();
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
            self.resources.gradient_texture = gradient_texture;

            // Since the gradient texture has changed, we need to update the gradient bind group.
            self.resources.gradient_bind_group = Self::create_gradient_bind_group(
                device,
                &self.gradient_bind_group_layout,
                &self
                    .resources
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
                &self.resources.view_config_buffer,
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
        device: &Device,
        resources: &mut GpuResources,
        required_atlas_count: u32,
    ) {
        let (width, height) = resources.atlas_size;
        while resources.atlas_textures.len() < required_atlas_count as usize {
            let texture = Self::create_atlas_texture(device, width, height);
            let texture_view = texture.create_view(&TextureViewDescriptor::default());
            resources.atlas_textures.push(texture);
            resources.atlas_texture_views.push(texture_view);
        }
    }

    /// Upload alpha data to the texture.
    fn upload_alpha_texture(&mut self, encoder: &mut CommandEncoder, alphas: &[u8]) {
        if alphas.is_empty() {
            return;
        }

        let texture = &self.resources.alphas_texture;
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

        let texture = &self.resources.encoded_paints_texture;
        let rows = paint_idxs.last().unwrap().div_ceil(texture.width());
        let data = &mut self.encoded_paints_data[..(rows * texture.width() * 16) as usize];

        GpuEncodedPaint::serialize_to_buffer(encoded_paints, data);

        self.staging_belt.upload_texture(encoder, texture, data, 16);
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

        self.staging_belt.upload_texture(
            encoder,
            &self.resources.gradient_texture,
            gradient_cache.luts(),
            4,
        );
    }
}
