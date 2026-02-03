// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU rendering module for the sparse strips CPU/GPU rendering engine.
//!
//! This module provides the GPU-side implementation of the hybrid rendering system.
//! It handles:
//! - GPU resource management (buffers, textures, pipelines)
//! - Surface/window management and presentation
//! - Shader execution and rendering
//!
//! The hybrid approach combines CPU-side path processing with efficient GPU rendering
//! to balance flexibility and performance.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

use alloc::vec::Vec;
use alloc::{sync::Arc, vec};
use core::{fmt::Debug, num::NonZeroU64};

use crate::AtlasConfig;
use crate::multi_atlas::AtlasId;
use crate::{
    GpuStrip, RenderError, RenderSettings, RenderSize,
    gradient_cache::GradientRampCache,
    image_cache::{ImageCache, ImageResource},
    render::{
        Config,
        common::{
            GPU_ENCODED_IMAGE_SIZE_TEXELS, GPU_LINEAR_GRADIENT_SIZE_TEXELS,
            GPU_RADIAL_GRADIENT_SIZE_TEXELS, GPU_SWEEP_GRADIENT_SIZE_TEXELS, GpuEncodedImage,
            GpuEncodedPaint, GpuLinearGradient, GpuRadialGradient, GpuSweepGradient,
            pack_image_offset, pack_image_params, pack_image_size, pack_radial_kind_and_swapped,
            pack_texture_width_and_extend_mode,
        },
    },
    scene::Scene,
    schedule::{LoadOp, RendererBackend, Scheduler, SchedulerState},
};
use bytemuck::{Pod, Zeroable};
use vello_common::{
    coarse::WideTile,
    encode::{EncodedGradient, EncodedKind, EncodedPaint, MAX_GRADIENT_LUT_SIZE, RadialKind},
    kurbo::Affine,
    paint::ImageSource,
    peniko,
    pixmap::Pixmap,
    tile::Tile,
};
use wgpu::{
    BindGroup, BindGroupLayout, BlendState, Buffer, ColorTargetState, ColorWrites, CommandEncoder,
    Device, Extent3d, PipelineCompilationOptions, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, Texture, TextureView, TextureViewDescriptor,
    util::DeviceExt,
};

/// Placeholder value for uninitialized GPU encoded paints.
const GPU_PAINT_PLACEHOLDER: GpuEncodedPaint = GpuEncodedPaint::LinearGradient(GpuLinearGradient {
    texture_width_and_extend_mode: 0,
    gradient_start: 0,
    transform: [0.0; 6],
});

/// Options for the renderer
#[derive(Debug)]
pub struct RenderTargetConfig {
    /// Format of the rendering target
    pub format: wgpu::TextureFormat,
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
}

/// Vello Hybrid's Renderer.
#[derive(Debug)]
pub struct Renderer {
    /// Programs for rendering.
    programs: Programs,
    /// Scheduler for scheduling draws.
    scheduler: Scheduler,
    /// The state used by the scheduler.
    scheduler_state: SchedulerState,
    /// Image cache for storing images atlas allocations.
    image_cache: ImageCache,
    /// Encoded paints for storing encoded paints.
    encoded_paints: Vec<GpuEncodedPaint>,
    /// Stores the index (offset) of the encoded paints in the encoded paints texture.
    paint_idxs: Vec<u32>,
    /// Gradient cache for storing gradient ramps.
    gradient_cache: GradientRampCache,
}

impl Renderer {
    /// Creates a new renderer.
    pub fn new(device: &Device, render_target_config: &RenderTargetConfig) -> Self {
        Self::new_with(device, render_target_config, RenderSettings::default())
    }

    /// Creates a new renderer with specific settings.
    pub fn new_with(
        device: &Device,
        render_target_config: &RenderTargetConfig,
        settings: RenderSettings,
    ) -> Self {
        super::common::maybe_warn_about_webgl_feature_conflict();

        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        let total_slots = (max_texture_dimension_2d / u32::from(Tile::HEIGHT)) as usize;
        let image_cache = ImageCache::new_with_config(settings.atlas_config);
        // Estimate the maximum number of gradient cache entries based on the max texture dimension
        // and the maximum gradient LUT size - worst case scenario.
        let max_gradient_cache_size =
            max_texture_dimension_2d * max_texture_dimension_2d / MAX_GRADIENT_LUT_SIZE as u32;
        let gradient_cache = GradientRampCache::new(max_gradient_cache_size, settings.level);

        Self {
            programs: Programs::new(device, &image_cache, render_target_config, total_slots),
            scheduler: Scheduler::new(total_slots),
            scheduler_state: SchedulerState::default(),
            image_cache,
            gradient_cache,
            encoded_paints: Vec::new(),
            paint_idxs: Vec::new(),
        }
    }

    /// Render `scene` into the provided command encoder.
    ///
    /// This method creates GPU resources as needed and schedules potentially multiple
    /// render passes.
    pub fn render(
        &mut self,
        scene: &Scene,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        view: &TextureView,
    ) -> Result<(), RenderError> {
        self.prepare_gpu_encoded_paints(&scene.encoded_paints);
        // TODO: For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        self.programs.prepare(
            device,
            queue,
            &mut self.gradient_cache,
            &self.encoded_paints,
            &mut scene.strip_storage.borrow_mut().alphas,
            render_size,
            &self.paint_idxs,
        );
        let mut junk = RendererContext {
            programs: &mut self.programs,
            device,
            queue,
            encoder,
            view,
        };

        let result = self.scheduler.do_scene(
            &mut self.scheduler_state,
            &mut junk,
            scene,
            &self.paint_idxs,
        );
        self.gradient_cache.maintain();

        result
    }

    /// Upload image to cache and atlas in one step. Returns the `ImageId`.
    ///
    /// It's used when an image is not already in the cache.
    ///
    /// This is a convenience method that:
    /// 1. Reserves space in the image cache
    /// 2. Writes the image data directly to the atlas
    /// 3. Returns the `ImageId` for use in rendering
    pub fn upload_image<T: AtlasWriter>(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        writer: &T,
    ) -> vello_common::paint::ImageId {
        let width = writer.width();
        let height = writer.height();
        let image_id = self.image_cache.allocate(width, height).unwrap();
        let image_resource = self
            .image_cache
            .get(image_id)
            .expect("Image resource not found");

        Programs::maybe_resize_atlas_texture_array(
            device,
            encoder,
            &mut self.programs.resources,
            &self.programs.atlas_bind_group_layout,
            self.image_cache.atlas_count() as u32,
        );
        let offset = [
            image_resource.offset[0] as u32,
            image_resource.offset[1] as u32,
        ];
        writer.write_to_atlas_layer(
            device,
            queue,
            encoder,
            &self.programs.resources.atlas_texture_array,
            image_resource.atlas_id.as_u32(),
            offset,
            width,
            height,
        );

        image_id
    }

    /// Destroy an image from the cache and clear the allocated slot in the atlas.
    pub fn destroy_image(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        image_id: vello_common::paint::ImageId,
    ) {
        if let Some(image_resource) = self.image_cache.deallocate(image_id) {
            self.clear_atlas_region(
                device,
                queue,
                encoder,
                image_resource.atlas_id,
                [
                    image_resource.offset[0] as u32,
                    image_resource.offset[1] as u32,
                ],
                image_resource.width as u32,
                image_resource.height as u32,
            );
        }
    }

    /// Clear a specific region of the atlas texture.
    fn clear_atlas_region(
        &mut self,
        _device: &Device,
        _queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_id: AtlasId,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        // Create a texture view for the specific atlas layer
        let layer_view =
            self.programs
                .resources
                .atlas_texture_array
                .create_view(&TextureViewDescriptor {
                    label: Some("Atlas Layer Clear View"),
                    format: Some(wgpu::TextureFormat::Rgba8Unorm),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: atlas_id.as_u32(),
                    array_layer_count: Some(1),
                    // Inherit usage from the texture
                    usage: None,
                });

        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Clear Atlas Region"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &layer_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Don't clear entire texture, just the scissor region
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        // Set scissor rectangle to limit clearing to specific region
        render_pass.set_scissor_rect(offset[0], offset[1], width, height);
        // Use atlas clear pipeline to render transparent pixels
        render_pass.set_pipeline(&self.programs.atlas_clear_pipeline);
        // Draw fullscreen quad
        render_pass.draw(0..4, 0..1);
    }

    fn prepare_gpu_encoded_paints(&mut self, encoded_paints: &[EncodedPaint]) {
        self.encoded_paints
            .resize_with(encoded_paints.len(), || GPU_PAINT_PLACEHOLDER);
        self.paint_idxs.resize(encoded_paints.len() + 1, 0);

        let mut current_idx = 0;
        for (encoded_paint_idx, paint) in encoded_paints.iter().enumerate() {
            self.paint_idxs[encoded_paint_idx] = current_idx;
            match paint {
                EncodedPaint::Image(img) => {
                    if let ImageSource::OpaqueId(image_id) = img.source {
                        let image_resource: Option<&ImageResource> = self.image_cache.get(image_id);
                        if let Some(image_resource) = image_resource {
                            let image_paint = self.encode_image_paint(img, image_resource);
                            self.encoded_paints[encoded_paint_idx] = image_paint;
                            current_idx += GPU_ENCODED_IMAGE_SIZE_TEXELS;
                        }
                    }
                }
                EncodedPaint::Gradient(gradient) => {
                    let (gradient_start, gradient_width) =
                        self.gradient_cache.get_or_create_ramp(gradient);
                    let gradient_paint: GpuEncodedPaint =
                        self.encode_gradient_paint(gradient, gradient_width, gradient_start);
                    let gradient_size_texels = match &gradient_paint {
                        GpuEncodedPaint::LinearGradient(_) => GPU_LINEAR_GRADIENT_SIZE_TEXELS,
                        GpuEncodedPaint::RadialGradient(_) => GPU_RADIAL_GRADIENT_SIZE_TEXELS,
                        GpuEncodedPaint::SweepGradient(_) => GPU_SWEEP_GRADIENT_SIZE_TEXELS,
                        _ => unreachable!("encode_gradient_for_gpu only returns gradient types"),
                    };
                    self.encoded_paints[encoded_paint_idx] = gradient_paint;
                    current_idx += gradient_size_texels;
                }
                EncodedPaint::BlurredRoundedRect(_blurred_rect) => {
                    // TODO: Blurred rounded rectangles are not yet supported
                    log::warn!(
                        "Blurred rounded rectangles are not yet supported in sparse strips hybrid renderer"
                    );
                }
            }
        }
        self.paint_idxs[encoded_paints.len()] = current_idx;
    }

    fn encode_image_paint(
        &self,
        image: &vello_common::encode::EncodedImage,
        image_resource: &ImageResource,
    ) -> GpuEncodedPaint {
        let image_transform = image.transform * Affine::translate((-0.5, -0.5));
        let transform = image_transform.as_coeffs().map(|x| x as f32);
        let image_size = pack_image_size(image_resource.width, image_resource.height);
        let image_offset = pack_image_offset(image_resource.offset[0], image_resource.offset[1]);
        let image_params = pack_image_params(
            image.sampler.quality as u32,
            image.sampler.x_extend as u32,
            image.sampler.y_extend as u32,
            image_resource.atlas_id.as_u32(),
        );

        GpuEncodedPaint::Image(GpuEncodedImage {
            image_params,
            image_size,
            image_offset,
            transform,
            _padding: [0, 0, 0],
        })
    }

    fn encode_gradient_paint(
        &self,
        gradient: &EncodedGradient,
        gradient_width: u32,
        gradient_start: u32,
    ) -> GpuEncodedPaint {
        let gradient_transform = gradient.transform * Affine::translate((-0.5, -0.5));
        let transform = gradient_transform.as_coeffs().map(|x| x as f32);
        let extend_mode = match gradient.extend {
            peniko::Extend::Pad => 0,
            peniko::Extend::Repeat => 1,
            peniko::Extend::Reflect => 2,
        };
        let texture_width_and_extend_mode =
            pack_texture_width_and_extend_mode(gradient_width, extend_mode);

        match &gradient.kind {
            EncodedKind::Linear(_) => GpuEncodedPaint::LinearGradient(GpuLinearGradient {
                texture_width_and_extend_mode,
                gradient_start,
                transform,
            }),
            EncodedKind::Radial(radial) => {
                let (kind, bias, scale, fp0, fp1, fr1, f_focal_x, f_is_swapped, scaled_r0_squared) =
                    match radial {
                        RadialKind::Radial { bias, scale } => {
                            (0, *bias, *scale, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
                        }
                        RadialKind::Strip { scaled_r0_squared } => {
                            (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, *scaled_r0_squared)
                        }
                        RadialKind::Focal {
                            focal_data,
                            fp0,
                            fp1,
                        } => (
                            2,
                            *fp0,
                            *fp1,
                            *fp0,
                            *fp1,
                            focal_data.fr1,
                            focal_data.f_focal_x,
                            focal_data.f_is_swapped as u32,
                            0.0,
                        ),
                    };
                GpuEncodedPaint::RadialGradient(GpuRadialGradient {
                    texture_width_and_extend_mode,
                    gradient_start,
                    transform,
                    kind_and_f_is_swapped: pack_radial_kind_and_swapped(kind, f_is_swapped),
                    bias,
                    scale,
                    fp0,
                    fp1,
                    fr1,
                    f_focal_x,
                    scaled_r0_squared,
                })
            }
            EncodedKind::Sweep(sweep) => GpuEncodedPaint::SweepGradient(GpuSweepGradient {
                texture_width_and_extend_mode,
                gradient_start,
                transform,
                start_angle: sweep.start_angle,
                inv_angle_delta: sweep.inv_angle_delta,
                _padding: [0, 0],
            }),
        }
    }
}

/// Defines the GPU resources and pipelines for rendering.
#[derive(Debug)]
struct Programs {
    /// Pipeline for rendering wide tile commands.
    strip_pipeline: RenderPipeline,
    /// Bind group layout for strip draws
    strip_bind_group_layout: BindGroupLayout,
    /// Bind group layout for encoded paints
    encoded_paints_bind_group_layout: BindGroupLayout,
    /// Bind group layout for gradient texture
    gradient_bind_group_layout: BindGroupLayout,
    /// Bind group layout for atlas textures
    atlas_bind_group_layout: BindGroupLayout,
    /// Pipeline for clearing slots in slot textures.
    clear_pipeline: RenderPipeline,
    /// Pipeline for clearing atlas regions.
    atlas_clear_pipeline: RenderPipeline,
    /// GPU resources for rendering (created during prepare)
    resources: GpuResources,
    /// Dimensions of the rendering target
    render_size: RenderSize,
    /// Scratch buffer for staging encoded paints texture data.
    encoded_paints_data: Vec<u8>,
}

/// Contains all GPU resources needed for rendering
#[derive(Debug)]
struct GpuResources {
    /// Buffer for [`GpuStrip`] data
    strips_buffer: Buffer,
    /// Texture for alpha values (used by both view and slot rendering)
    alphas_texture: Texture,
    /// Textures for atlas data (multiple atlases supported)
    atlas_texture_array: Texture,
    /// View for atlas texture array
    atlas_texture_array_view: TextureView,
    /// Bind group for atlas textures (as texture array)
    atlas_bind_group: BindGroup,
    /// Texture for encoded paints
    encoded_paints_texture: Texture,
    /// Bind group for encoded paints
    encoded_paints_bind_group: BindGroup,
    /// Texture for gradient lookup table
    gradient_texture: Texture,
    /// Bind group for gradient texture
    gradient_bind_group: BindGroup,

    /// Config buffer for rendering wide tile commands into the view texture.
    view_config_buffer: Buffer,
    /// Config buffer for rendering wide tile commands into a slot texture.
    slot_config_buffer: Buffer,

    /// Buffer for slot indices used in `clear_slots`
    clear_slot_indices_buffer: Buffer,
    // Bind groups for rendering with clip buffers
    slot_bind_groups: [BindGroup; 3],
    /// Slot texture views
    slot_texture_views: [TextureView; 2],

    /// Bind group for clear slots operation
    clear_bind_group: BindGroup,
}

const SIZE_OF_CONFIG: NonZeroU64 = NonZeroU64::new(size_of::<Config>() as u64).unwrap();

/// Config for the clear slots pipeline
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct ClearSlotsConfig {
    /// Width of a slot
    pub slot_width: u32,
    /// Height of a slot
    pub slot_height: u32,
    /// Total height of the texture
    pub texture_height: u32,
    /// Padding for 16-byte alignment
    pub _padding: u32,
}

impl GpuStrip {
    /// Vertex attributes for the strip
    pub fn vertex_attributes() -> [wgpu::VertexAttribute; 5] {
        wgpu::vertex_attr_array![
            0 => Uint32,
            1 => Uint32,
            2 => Uint32,
            3 => Uint32,
            4 => Uint32,
        ]
    }
}

impl Programs {
    fn new(
        device: &Device,
        image_cache: &ImageCache,
        render_target_config: &RenderTargetConfig,
        slot_count: usize,
    ) -> Self {
        let strip_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Strip Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let atlas_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Atlas Texture Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let encoded_paints_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Encoded Paints Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let gradient_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Gradient Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        // Create bind group layout for clearing slots
        let clear_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Clear Slots Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let strip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Strip Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::RENDER_STRIPS.into()),
        });

        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clear Slots Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::CLEAR_SLOTS.into()),
        });

        let strip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Strip Pipeline Layout"),
                bind_group_layouts: &[
                    &strip_bind_group_layout,
                    &atlas_bind_group_layout,
                    &encoded_paints_bind_group_layout,
                    &gradient_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Clear Slots Pipeline Layout"),
                bind_group_layouts: &[&clear_bind_group_layout],
                push_constant_ranges: &[],
            });

        let strip_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Strip Pipeline"),
            layout: Some(&strip_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &strip_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<GpuStrip>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &GpuStrip::vertex_attributes(),
                }],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &strip_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: render_target_config.format,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let clear_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Clear Slots Pipeline"),
            layout: Some(&clear_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &clear_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<u32>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint32,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &clear_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: render_target_config.format,
                    // No blending needed for clearing
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create atlas clear pipeline
        let atlas_clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Atlas Clear Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });
        let atlas_clear_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Atlas Clear Pipeline"),
            layout: Some(&atlas_clear_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &clear_shader,
                // Use a different vertex shader entry point
                entry_point: Some("vs_main_fullscreen"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &clear_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let slot_texture_views: [TextureView; 2] = core::array::from_fn(|_| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("Slot Texture"),
                    size: Extent3d {
                        width: u32::from(WideTile::WIDTH),
                        height: u32::from(Tile::HEIGHT) * slot_count as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: render_target_config.format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        });

        let clear_config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Clear Slots Config"),
            contents: bytemuck::bytes_of(&ClearSlotsConfig {
                slot_width: u32::from(WideTile::WIDTH),
                slot_height: u32::from(Tile::HEIGHT),
                texture_height: u32::from(Tile::HEIGHT) * slot_count as u32,
                _padding: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let clear_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clear Slots Bind Group"),
            layout: &clear_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: clear_config_buffer.as_entire_binding(),
            }],
        });
        let clear_slot_indices_buffer = Self::create_clear_slot_indices_buffer(
            device,
            slot_count as u64 * size_of::<u32>() as u64,
        );

        let slot_config_buffer = Self::create_config_buffer(
            device,
            &RenderSize {
                width: u32::from(WideTile::WIDTH),
                height: u32::from(Tile::HEIGHT) * slot_count as u32,
            },
            device.limits().max_texture_dimension_2d,
        );

        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        const INITIAL_ALPHA_TEXTURE_HEIGHT: u32 = 1;
        let alphas_texture = Self::create_alphas_texture(
            device,
            max_texture_dimension_2d,
            INITIAL_ALPHA_TEXTURE_HEIGHT,
        );
        let view_config_buffer = Self::create_config_buffer(
            device,
            &RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            max_texture_dimension_2d,
        );

        let AtlasConfig {
            atlas_size: (atlas_width, atlas_height),
            initial_atlas_count,
            ..
        } = image_cache.atlas_manager().config();
        let (atlas_texture_array, atlas_texture_array_view) = Self::create_atlas_texture_array(
            device,
            *atlas_width,
            *atlas_height,
            *initial_atlas_count as u32,
        );
        let atlas_bind_group = Self::create_atlas_bind_group(
            device,
            &atlas_bind_group_layout,
            &atlas_texture_array_view,
        );

        const INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT: u32 = 1;
        let encoded_paints_data = vec![
            0;
            ((max_texture_dimension_2d * INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT) << 4)
                as usize
        ];
        let encoded_paints_texture = Self::create_encoded_paints_texture(
            device,
            max_texture_dimension_2d,
            INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT,
        );
        let encoded_paints_bind_group = Self::create_encoded_paints_bind_group(
            device,
            &encoded_paints_bind_group_layout,
            &encoded_paints_texture.create_view(&TextureViewDescriptor::default()),
        );

        const INITIAL_GRADIENT_TEXTURE_HEIGHT: u32 = 1;
        let gradient_texture = Self::create_gradient_texture(
            device,
            max_texture_dimension_2d,
            INITIAL_GRADIENT_TEXTURE_HEIGHT,
        );
        let gradient_bind_group = Self::create_gradient_bind_group(
            device,
            &gradient_bind_group_layout,
            &gradient_texture.create_view(&TextureViewDescriptor::default()),
        );

        let slot_bind_groups = Self::create_strip_bind_groups(
            device,
            &strip_bind_group_layout,
            &alphas_texture.create_view(&TextureViewDescriptor::default()),
            &slot_config_buffer,
            &view_config_buffer,
            &slot_texture_views,
        );

        let resources = GpuResources {
            strips_buffer: Self::create_strips_buffer(device, 0),
            clear_slot_indices_buffer,
            slot_texture_views,
            slot_config_buffer,
            slot_bind_groups,
            clear_bind_group,
            alphas_texture,
            atlas_texture_array,
            atlas_texture_array_view,
            atlas_bind_group,
            encoded_paints_texture,
            encoded_paints_bind_group,
            gradient_texture,
            gradient_bind_group,
            view_config_buffer,
        };

        Self {
            strip_pipeline,
            strip_bind_group_layout,
            encoded_paints_bind_group_layout,
            gradient_bind_group_layout,
            atlas_bind_group_layout,
            resources,
            encoded_paints_data,
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            clear_pipeline,
            atlas_clear_pipeline,
        }
    }

    fn create_strips_buffer(device: &Device, required_strips_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Strips Buffer"),
            size: required_strips_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_clear_slot_indices_buffer(device: &Device, required_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Slot Indices Buffer"),
            size: required_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_config_buffer(
        device: &Device,
        render_size: &RenderSize,
        alpha_texture_width: u32,
    ) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&Config {
                width: render_size.width,
                height: render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: alpha_texture_width.trailing_zeros(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn create_alphas_texture(device: &Device, width: u32, height: u32) -> Texture {
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

    fn create_atlas_texture_array(
        device: &Device,
        width: u32,
        height: u32,
        atlas_count: u32,
    ) -> (Texture, TextureView) {
        // Create a single texture array with multiple layers
        let atlas_texture_array = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atlas Texture Array"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: atlas_count,
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
        });

        let atlas_texture_array_view = atlas_texture_array.create_view(&TextureViewDescriptor {
            label: Some("Atlas Texture Array View"),
            format: None,
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(atlas_count),
            usage: None,
        });

        (atlas_texture_array, atlas_texture_array_view)
    }

    fn create_atlas_bind_group(
        device: &Device,
        atlas_bind_group_layout: &BindGroupLayout,
        atlas_texture_array_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Atlas Bind Group"),
            layout: atlas_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(atlas_texture_array_view),
            }],
        })
    }

    fn create_encoded_paints_texture(device: &Device, width: u32, height: u32) -> Texture {
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

    fn create_encoded_paints_bind_group(
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

    fn create_gradient_texture(device: &Device, width: u32, height: u32) -> Texture {
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

    fn create_gradient_bind_group(
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

    fn create_strip_bind_groups(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture_view: &TextureView,
        strip_config_buffer: &Buffer,
        config_buffer: &Buffer,
        strip_texture_views: &[TextureView],
    ) -> [BindGroup; 3] {
        [
            Self::create_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                strip_config_buffer,
                &strip_texture_views[1],
            ),
            Self::create_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                strip_config_buffer,
                &strip_texture_views[0],
            ),
            Self::create_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                config_buffer,
                &strip_texture_views[1],
            ),
        ]
    }

    fn create_strip_bind_group(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture_view: &TextureView,
        config_buffer: &Buffer,
        strip_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Strip Bind Group"),
            layout: strip_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(alphas_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(strip_texture_view),
                },
            ],
        })
    }

    /// Prepare GPU buffers for rendering, given alphas.
    ///
    /// Specifically, updates the alpha texture with `alphas` and the config buffer when
    /// the rendering size changes.
    fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        gradient_cache: &mut GradientRampCache,
        encoded_paints: &[GpuEncodedPaint],
        alphas: &mut Vec<u8>,
        new_render_size: &RenderSize,
        paint_idxs: &[u32],
    ) {
        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        self.maybe_resize_alphas_tex(device, max_texture_dimension_2d, alphas.len());
        self.maybe_resize_encoded_paints_tex(device, max_texture_dimension_2d, paint_idxs);
        self.maybe_update_config_buffer(queue, max_texture_dimension_2d, new_render_size);

        self.upload_alpha_texture(queue, alphas);
        self.upload_encoded_paints_texture(queue, encoded_paints);

        if gradient_cache.has_changed() {
            self.maybe_resize_gradient_tex(device, max_texture_dimension_2d, gradient_cache);
            self.upload_gradient_texture(queue, gradient_cache);
            gradient_cache.mark_synced();
        }
    }

    /// Update the alpha texture size if needed.
    fn maybe_resize_alphas_tex(
        &mut self,
        device: &Device,
        max_texture_dimension_2d: u32,
        alphas_len: usize,
    ) {
        let required_alpha_height = u32::try_from(alphas_len)
            .unwrap()
            // There are 16 1-byte alpha values per texel.
            .div_ceil(max_texture_dimension_2d << 4);
        debug_assert!(
            self.resources.alphas_texture.width() == max_texture_dimension_2d,
            "Alpha texture width must match max texture dimensions"
        );
        let current_alpha_height = self.resources.alphas_texture.height();
        if required_alpha_height > current_alpha_height {
            // We need to resize the alpha texture to fit the new alpha data.
            assert!(
                required_alpha_height <= max_texture_dimension_2d,
                "Alpha texture height exceeds max texture dimensions"
            );

            // The alpha texture encodes 16 1-byte alpha values per texel, with 4 alpha values packed in each channel
            let alphas_texture = Self::create_alphas_texture(
                device,
                max_texture_dimension_2d,
                required_alpha_height,
            );
            self.resources.alphas_texture = alphas_texture;

            // Since the alpha texture has changed, we need to update the clip bind groups.
            self.resources.slot_bind_groups = Self::create_strip_bind_groups(
                device,
                &self.strip_bind_group_layout,
                &self
                    .resources
                    .alphas_texture
                    .create_view(&TextureViewDescriptor::default()),
                &self.resources.slot_config_buffer,
                &self.resources.view_config_buffer,
                &self.resources.slot_texture_views,
            );
        }
    }

    /// Update the encoded paints texture size if needed.
    fn maybe_resize_encoded_paints_tex(
        &mut self,
        device: &Device,
        max_texture_dimension_2d: u32,
        paint_idxs: &[u32],
    ) {
        let required_texels = paint_idxs.last().unwrap();
        let required_encoded_paints_height = required_texels.div_ceil(max_texture_dimension_2d);
        debug_assert!(
            self.resources.encoded_paints_texture.width() == max_texture_dimension_2d,
            "Encoded paints texture width must match max texture dimensions"
        );
        let current_encoded_paints_height = self.resources.encoded_paints_texture.height();
        if required_encoded_paints_height > current_encoded_paints_height {
            assert!(
                required_encoded_paints_height <= max_texture_dimension_2d,
                "Encoded paints texture height exceeds max texture dimensions"
            );
            let required_encoded_paints_size =
                (max_texture_dimension_2d * required_encoded_paints_height) << 4;
            self.encoded_paints_data
                .resize(required_encoded_paints_size as usize, 0);
            let encoded_paints_texture = Self::create_encoded_paints_texture(
                device,
                max_texture_dimension_2d,
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
        max_texture_dimension_2d: u32,
        gradient_cache: &GradientRampCache,
    ) {
        let gradient_pixels = (gradient_cache.luts_size() / 4) as u32; // 4 bytes per RGBA8 pixel
        let required_gradient_height = gradient_pixels.div_ceil(max_texture_dimension_2d);
        debug_assert!(
            self.resources.gradient_texture.width() == max_texture_dimension_2d,
            "Gradient texture width must match max texture dimensions"
        );
        let current_gradient_height = self.resources.gradient_texture.height();
        if required_gradient_height > current_gradient_height {
            assert!(
                required_gradient_height <= max_texture_dimension_2d,
                "Gradient texture height exceeds max texture dimensions"
            );
            let gradient_texture = Self::create_gradient_texture(
                device,
                max_texture_dimension_2d,
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

    /// Update config buffer if dimensions changed.
    fn maybe_update_config_buffer(
        &mut self,
        queue: &Queue,
        max_texture_dimension_2d: u32,
        new_render_size: &RenderSize,
    ) {
        if self.render_size != *new_render_size {
            let config = Config {
                width: new_render_size.width,
                height: new_render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
            };
            let mut buffer = queue
                .write_buffer_with(&self.resources.view_config_buffer, 0, SIZE_OF_CONFIG)
                .expect("Buffer only ever holds `Config`");
            buffer.copy_from_slice(bytemuck::bytes_of(&config));

            self.render_size = new_render_size.clone();
        }
    }

    /// Resize the texture array to accommodate more atlases.
    fn maybe_resize_atlas_texture_array(
        device: &Device,
        encoder: &mut CommandEncoder,
        resources: &mut GpuResources,
        atlas_bind_group_layout: &BindGroupLayout,
        required_atlas_count: u32,
    ) {
        let Extent3d {
            width,
            height,
            depth_or_array_layers: current_atlas_count,
        } = resources.atlas_texture_array.size();
        if required_atlas_count > current_atlas_count {
            // Create new texture array with more layers
            let (new_atlas_texture_array, new_atlas_texture_array_view) =
                Self::create_atlas_texture_array(device, width, height, required_atlas_count);

            // Copy existing atlas data from old texture array to new one
            Self::copy_atlas_texture_data(
                encoder,
                &resources.atlas_texture_array,
                &new_atlas_texture_array,
                current_atlas_count,
                width,
                height,
            );

            // Update the bind group with the new texture array view
            let new_atlas_bind_group = Self::create_atlas_bind_group(
                device,
                atlas_bind_group_layout,
                &new_atlas_texture_array_view,
            );

            // Replace the old resources
            resources.atlas_texture_array = new_atlas_texture_array;
            resources.atlas_texture_array_view = new_atlas_texture_array_view;
            resources.atlas_bind_group = new_atlas_bind_group;
        }
    }

    /// Copy texture data from the old atlas texture array to a new one.
    /// This is necessary when resizing the texture array to preserve existing atlas data.
    fn copy_atlas_texture_data(
        encoder: &mut CommandEncoder,
        old_atlas_texture_array: &Texture,
        new_atlas_texture_array: &Texture,
        layer_count_to_copy: u32,
        width: u32,
        height: u32,
    ) {
        // Copy all layers from old texture array to new texture array
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: old_atlas_texture_array,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: new_atlas_texture_array,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: layer_count_to_copy,
            },
        );
    }

    /// Upload alpha data to the texture.
    fn upload_alpha_texture(&mut self, queue: &Queue, alphas: &mut Vec<u8>) {
        if alphas.is_empty() {
            return;
        }

        let texture_width = self.resources.alphas_texture.width();
        let texture_height = self.resources.alphas_texture.height();
        let total_size = texture_width as usize * texture_height as usize * 16;

        let original_len = alphas.len();

        // Temporarily pad the length of the alphas to the texture size before uploading.
        alphas.resize(total_size, 0);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.resources.alphas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            alphas,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 16 bytes per RGBA32Uint texel (4 u32s × 4 bytes each), which is equivalent to
                // a bit shift of 4.
                bytes_per_row: Some(texture_width << 4),
                rows_per_image: Some(texture_height),
            },
            Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
        );

        // Truncate back to the original size.
        alphas.truncate(original_len);
    }

    /// Upload encoded paints to the texture.
    fn upload_encoded_paints_texture(&mut self, queue: &Queue, encoded_paints: &[GpuEncodedPaint]) {
        let encoded_paints_texture = &self.resources.encoded_paints_texture;
        let encoded_paints_texture_width = encoded_paints_texture.width();
        let encoded_paints_texture_height = encoded_paints_texture.height();

        GpuEncodedPaint::serialize_to_buffer(encoded_paints, &mut self.encoded_paints_data);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: encoded_paints_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.encoded_paints_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 16 bytes per RGBA32Uint texel (4 u32s × 4 bytes each), equivalent to bit shift of 4
                bytes_per_row: Some(encoded_paints_texture_width << 4),
                rows_per_image: Some(encoded_paints_texture_height),
            },
            Extent3d {
                width: encoded_paints_texture_width,
                height: encoded_paints_texture_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload gradient data to the texture.
    fn upload_gradient_texture(&mut self, queue: &Queue, gradient_cache: &mut GradientRampCache) {
        let gradient_texture = &self.resources.gradient_texture;
        let gradient_texture_width = gradient_texture.width();
        let gradient_texture_height = gradient_texture.height();

        // Upload the gradient LUT data
        if !gradient_cache.is_empty() {
            let total_capacity = (gradient_texture_width * gradient_texture_height * 4) as usize;

            // Take ownership of the luts to avoid copying, then resize for texture padding
            let mut luts = gradient_cache.take_luts();
            let old_luts_len = luts.len();
            luts.resize(total_capacity, 0);

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: gradient_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &luts,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // 4 bytes per RGBA8 pixel
                    bytes_per_row: Some(gradient_texture_width << 2),
                    rows_per_image: Some(gradient_texture_height),
                },
                Extent3d {
                    width: gradient_texture_width,
                    height: gradient_texture_height,
                    depth_or_array_layers: 1,
                },
            );

            // Restore the luts back to the cache
            luts.truncate(old_luts_len);
            gradient_cache.restore_luts(luts);
        }
    }

    /// Upload the strip data by creating and assigning a new `self.resources.strips_buffer`.
    fn upload_strips(&mut self, device: &Device, queue: &Queue, strips: &[GpuStrip]) {
        let required_strips_size = size_of_val(strips) as u64;
        self.resources.strips_buffer = Self::create_strips_buffer(device, required_strips_size);
        // TODO: Consider using a staging belt to avoid an extra staging buffer allocation.
        let mut buffer = queue
            .write_buffer_with(
                &self.resources.strips_buffer,
                0,
                required_strips_size.try_into().unwrap(),
            )
            .expect("Capacity handled in creation");
        buffer.copy_from_slice(bytemuck::cast_slice(strips));
    }
}

/// A struct containing references to the many objects needed to get work
/// scheduled onto the GPU.
struct RendererContext<'a> {
    programs: &'a mut Programs,
    device: &'a Device,
    queue: &'a Queue,
    encoder: &'a mut CommandEncoder,
    view: &'a TextureView,
}

impl RendererContext<'_> {
    /// Render the strips to either the view or a slot texture (depending on `ix`).
    fn do_strip_render_pass(
        &mut self,
        strips: &[GpuStrip],
        ix: usize,
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        debug_assert!(ix < 3, "Invalid texture index");
        if strips.is_empty() {
            return;
        }
        // TODO: We currently allocate a new strips buffer for each render pass. A more efficient
        // approach would be to re-use buffers or slices of a larger buffer.
        self.programs.upload_strips(self.device, self.queue, strips);

        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render to Texture Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: if ix == 2 {
                    self.view
                } else {
                    &self.programs.resources.slot_texture_views[ix]
                },
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(&self.programs.strip_pipeline);
        render_pass.set_bind_group(0, &self.programs.resources.slot_bind_groups[ix], &[]);
        render_pass.set_bind_group(1, &self.programs.resources.atlas_bind_group, &[]);
        render_pass.set_bind_group(2, &self.programs.resources.encoded_paints_bind_group, &[]);
        render_pass.set_bind_group(3, &self.programs.resources.gradient_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.programs.resources.strips_buffer.slice(..));
        render_pass.draw(0..4, 0..u32::try_from(strips.len()).unwrap());
    }

    /// Clear specific slots from a slot texture.
    fn do_clear_slots_render_pass(&mut self, ix: usize, slot_indices: &[u32]) {
        if slot_indices.is_empty() {
            return;
        }

        let resources = &mut self.programs.resources;
        let size = size_of_val(slot_indices) as u64;
        // TODO: We currently allocate a new strips buffer for each render pass. A more efficient
        // approach would be to re-use buffers or slices of a larger buffer.
        resources.clear_slot_indices_buffer =
            Programs::create_clear_slot_indices_buffer(self.device, size);
        // TODO: Consider using a staging belt to avoid an extra staging buffer allocation.
        let mut buffer = self
            .queue
            .write_buffer_with(
                &resources.clear_slot_indices_buffer,
                0,
                size.try_into().unwrap(),
            )
            .expect("Capacity handled in creation");
        buffer.copy_from_slice(bytemuck::cast_slice(slot_indices));

        {
            let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Clear Slots Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &resources.slot_texture_views[ix],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Don't clear the entire texture, just specific slots
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.programs.clear_pipeline);
            render_pass.set_bind_group(0, &resources.clear_bind_group, &[]);
            render_pass.set_vertex_buffer(0, resources.clear_slot_indices_buffer.slice(..));
            render_pass.draw(0..4, 0..u32::try_from(slot_indices.len()).unwrap());
        }
    }
}

impl RendererBackend for RendererContext<'_> {
    /// Execute the render pass for clearing slots.
    fn clear_slots(&mut self, texture_index: usize, slots: &[u32]) {
        self.do_clear_slots_render_pass(texture_index, slots);
    }

    /// Execute the render pass for rendering strips.
    fn render_strips(&mut self, strips: &[GpuStrip], target_index: usize, load_op: LoadOp) {
        let wgpu_load_op = match load_op {
            LoadOp::Load => wgpu::LoadOp::Load,
            LoadOp::Clear => wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        };

        self.do_strip_render_pass(strips, target_index, wgpu_load_op);
    }
}

/// Trait for types that can write image data directly to the atlas texture.
///
/// This allows efficient uploading from different sources:
/// - `Pixmap`: Direct upload without intermediate texture
/// - `Texture`: Texture-to-texture copy
/// - Custom implementations for other image sources
pub trait AtlasWriter {
    /// Get the width of the image.
    fn width(&self) -> u32;
    /// Get the height of the image.
    fn height(&self) -> u32;

    /// Write image data to a specific layer of an atlas texture array at the specified offset.
    fn write_to_atlas_layer(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    );
}

/// Implementation for `wgpu::Texture` - uses texture-to-texture copy
impl AtlasWriter for Texture {
    fn width(&self) -> u32 {
        self.width()
    }

    fn height(&self) -> u32 {
        self.height()
    }

    fn write_to_atlas_layer(
        &self,
        _device: &Device,
        _queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: offset[0],
                    y: offset[1],
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Implementation for `Pixmap` - direct upload to atlas
impl AtlasWriter for Pixmap {
    fn width(&self) -> u32 {
        self.width() as u32
    }

    fn height(&self) -> u32 {
        self.height() as u32
    }

    fn write_to_atlas_layer(
        &self,
        _device: &Device,
        queue: &Queue,
        _encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: offset[0],
                    y: offset[1],
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            self.data_as_u8_slice(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Implementation for `Arc<Pixmap>`
impl AtlasWriter for Arc<Pixmap> {
    fn width(&self) -> u32 {
        self.as_ref().width() as u32
    }

    fn height(&self) -> u32 {
        self.as_ref().height() as u32
    }

    fn write_to_atlas_layer(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        self.as_ref().write_to_atlas_layer(
            device,
            queue,
            encoder,
            atlas_texture,
            layer,
            offset,
            width,
            height,
        );
    }
}
