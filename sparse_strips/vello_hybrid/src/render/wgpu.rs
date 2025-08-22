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

use alloc::vec;
use alloc::vec::Vec;
use core::{fmt::Debug, mem, num::NonZeroU64};

use bytemuck::{Pod, Zeroable};
use vello_common::{
    coarse::WideTile, encode::EncodedPaint, kurbo::Affine, paint::ImageSource, pixmap::Pixmap,
    tile::Tile,
};
use wgpu::{
    BindGroup, BindGroupLayout, BlendState, Buffer, ColorTargetState, ColorWrites, CommandEncoder,
    Device, PipelineCompilationOptions, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, Texture, TextureView, util::DeviceExt,
};

use crate::{
    GpuStrip, RenderError, RenderSize,
    image_cache::{ImageCache, ImageResource},
    render::{Config, common::GpuEncodedImage},
    scene::Scene,
    schedule::{LoadOp, RendererBackend, Scheduler},
};

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
    programs: Programs,
    scheduler: Scheduler,
    image_cache: ImageCache,
}

impl Renderer {
    /// Creates a new renderer.
    pub fn new(device: &Device, render_target_config: &RenderTargetConfig) -> Self {
        super::common::maybe_warn_about_webgl_feature_conflict();

        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        let total_slots = (max_texture_dimension_2d / u32::from(Tile::HEIGHT)) as usize;
        let image_cache = ImageCache::new(max_texture_dimension_2d, max_texture_dimension_2d);

        Self {
            programs: Programs::new(device, render_target_config, total_slots),
            scheduler: Scheduler::new(total_slots),
            image_cache,
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
        let encoded_paints = self.prepare_gpu_encoded_paints(&scene.encoded_paints);
        // TODO: For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        self.programs.prepare(
            device,
            queue,
            scene.strip_generator.alpha_buf(),
            encoded_paints,
            render_size,
        );
        let mut junk = RendererContext {
            programs: &mut self.programs,
            device,
            queue,
            encoder,
            view,
        };

        self.scheduler.do_scene(&mut junk, scene)
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
        let image_id = self.image_cache.allocate(width, height);
        let image_resource = self
            .image_cache
            .get(image_id)
            .expect("Image resource not found");
        let offset = [
            image_resource.offset[0] as u32,
            image_resource.offset[1] as u32,
        ];

        writer.write_to_atlas(
            device,
            queue,
            encoder,
            &self.programs.resources.atlas_texture,
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
                [
                    image_resource.offset[0] as u32,
                    image_resource.offset[1] as u32,
                ],
                image_resource.width as u32,
                image_resource.height as u32,
            );
        }
    }

    /// Clear a specific region of the atlas texture with uninitialized data.
    fn clear_atlas_region(
        &mut self,
        _device: &Device,
        queue: &Queue,
        _encoder: &mut CommandEncoder,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        // Rgba8Unorm is 4 bytes per pixel
        let uninitialized_data = vec![0_u8; (width * height * 4) as usize];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.programs.resources.atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: offset[0],
                    y: offset[1],
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &uninitialized_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn prepare_gpu_encoded_paints(&self, encoded_paints: &[EncodedPaint]) -> Vec<GpuEncodedImage> {
        let mut bytes: Vec<GpuEncodedImage> = Vec::new();
        for paint in encoded_paints {
            match paint {
                EncodedPaint::Image(img) => {
                    if let ImageSource::OpaqueId(image_id) = img.source {
                        let image_resource: Option<&ImageResource> = self.image_cache.get(image_id);
                        if let Some(image_resource) = image_resource {
                            let transform = img.transform * Affine::translate((-0.5, -0.5));
                            // pack two u16 as u32
                            let image_size = ((image_resource.width as u32) << 16)
                                | image_resource.height as u32;
                            let image_offset = ((image_resource.offset[0] as u32) << 16)
                                | image_resource.offset[1] as u32;
                            let quality_and_extend_modes = ((img.extends.1 as u32) << 4)
                                | ((img.extends.0 as u32) << 2)
                                | img.quality as u32;

                            bytes.push(GpuEncodedImage {
                                quality_and_extend_modes,
                                image_size,
                                image_offset,
                                _padding1: 0,
                                transform: transform.as_coeffs().map(|x| x as f32),
                                _padding2: [0, 0],
                            });
                        }
                    }
                }
                _ => {
                    unimplemented!("Gradient and rounded rectangle unsupported")
                }
            }
        }
        bytes
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

    /// Pipeline for clearing slots in slot textures.
    clear_pipeline: RenderPipeline,

    /// GPU resources for rendering (created during prepare)
    resources: GpuResources,
    /// Dimensions of the rendering target
    render_size: RenderSize,
    /// Scratch buffer for staging alpha texture data.
    alpha_data: Vec<u8>,
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
    /// Texture for atlas data
    // TODO: Support more than one atlas texture
    atlas_texture: Texture,
    /// Bind group for atlas texture
    atlas_bind_group: BindGroup,
    /// Texture for encoded paints
    encoded_paints_texture: Texture,
    /// Bind group for encoded paints
    encoded_paints_bind_group: BindGroup,

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
    fn new(device: &Device, render_target_config: &RenderTargetConfig, slot_count: usize) -> Self {
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
                        view_dimension: wgpu::TextureViewDimension::D2,
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

        let slot_texture_views: [TextureView; 2] = core::array::from_fn(|_| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("Slot Texture"),
                    size: wgpu::Extent3d {
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
                .create_view(&wgpu::TextureViewDescriptor::default())
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
        let clear_slot_indices_buffer = Self::make_clear_slot_indices_buffer(
            device,
            slot_count as u64 * size_of::<u32>() as u64,
        );

        let slot_config_buffer = Self::make_config_buffer(
            device,
            &RenderSize {
                width: u32::from(WideTile::WIDTH),
                height: u32::from(Tile::HEIGHT) * slot_count as u32,
            },
            device.limits().max_texture_dimension_2d,
        );

        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        const INITIAL_ALPHA_TEXTURE_HEIGHT: u32 = 1;
        let alphas_texture = Self::make_alphas_texture(
            device,
            max_texture_dimension_2d,
            INITIAL_ALPHA_TEXTURE_HEIGHT,
        );
        let alpha_data =
            vec![0; ((max_texture_dimension_2d * INITIAL_ALPHA_TEXTURE_HEIGHT) << 4) as usize];
        let view_config_buffer = Self::make_config_buffer(
            device,
            &RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            max_texture_dimension_2d,
        );
        let atlas_texture = Self::make_atlas_texture(
            device,
            max_texture_dimension_2d,
            max_texture_dimension_2d,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let atlas_bind_group = Self::make_atlas_bind_group(
            device,
            &atlas_bind_group_layout,
            &atlas_texture.create_view(&Default::default()),
        );
        const INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT: u32 = 1;
        let encoded_paints_data = vec![
            0;
            ((max_texture_dimension_2d * INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT) << 4)
                as usize
        ];
        let encoded_paints_texture = Self::make_encoded_paints_texture(
            device,
            max_texture_dimension_2d,
            INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT,
        );
        let encoded_paints_bind_group = Self::make_encoded_paints_bind_group(
            device,
            &encoded_paints_bind_group_layout,
            &encoded_paints_texture.create_view(&Default::default()),
        );
        let slot_bind_groups = Self::make_strip_bind_groups(
            device,
            &strip_bind_group_layout,
            &alphas_texture.create_view(&Default::default()),
            &slot_config_buffer,
            &view_config_buffer,
            &slot_texture_views,
        );

        let resources = GpuResources {
            strips_buffer: Self::make_strips_buffer(device, 0),
            clear_slot_indices_buffer,
            slot_texture_views,
            slot_config_buffer,
            slot_bind_groups,
            clear_bind_group,
            alphas_texture,
            atlas_texture,
            atlas_bind_group,
            encoded_paints_texture,
            encoded_paints_bind_group,
            view_config_buffer,
        };

        Self {
            strip_pipeline,
            strip_bind_group_layout,
            encoded_paints_bind_group_layout,
            resources,
            alpha_data,
            encoded_paints_data,
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },

            clear_pipeline,
        }
    }

    fn make_strips_buffer(device: &Device, required_strips_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Strips Buffer"),
            size: required_strips_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_clear_slot_indices_buffer(device: &Device, required_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Slot Indices Buffer"),
            size: required_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_config_buffer(
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

    fn make_alphas_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Alpha Texture"),
            size: wgpu::Extent3d {
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

    fn make_atlas_texture(
        device: &Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atlas Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn make_atlas_bind_group(
        device: &Device,
        atlas_bind_group_layout: &BindGroupLayout,
        atlas_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Atlas Bind Group"),
            layout: atlas_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(atlas_texture_view),
            }],
        })
    }

    fn make_encoded_paints_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Encoded Paints Texture"),
            size: wgpu::Extent3d {
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

    fn make_encoded_paints_bind_group(
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

    fn make_strip_bind_groups(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture_view: &TextureView,
        strip_config_buffer: &Buffer,
        config_buffer: &Buffer,
        strip_texture_views: &[TextureView],
    ) -> [BindGroup; 3] {
        [
            Self::make_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                strip_config_buffer,
                &strip_texture_views[1],
            ),
            Self::make_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                strip_config_buffer,
                &strip_texture_views[0],
            ),
            Self::make_strip_bind_group(
                device,
                strip_bind_group_layout,
                alphas_texture_view,
                config_buffer,
                &strip_texture_views[1],
            ),
        ]
    }

    fn make_strip_bind_group(
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
        alphas: &[u8],
        encoded_paints: Vec<GpuEncodedImage>,
        new_render_size: &RenderSize,
    ) {
        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        // Update the alpha texture size if needed
        self.maybe_resize_alphas_tex(device, alphas, max_texture_dimension_2d);
        // Update the encoded paints texture size if needed
        self.maybe_resize_encoded_paints_tex(device, &encoded_paints, max_texture_dimension_2d);

        // Update config buffer if dimensions changed.
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

        // Prepare alpha data for the texture with 16 1-byte alpha values per texel (4 per channel)
        let texture_width = self.resources.alphas_texture.width();
        let texture_height = self.resources.alphas_texture.height();
        debug_assert!(
            alphas.len() <= (texture_width * texture_height * 16) as usize,
            "Alpha texture dimensions are too small to fit the alpha data"
        );

        // After this copy to `self.alpha_data`, there may be stale trailing alpha values. These
        // are not sampled, so can be left as-is.
        self.alpha_data[0..alphas.len()].copy_from_slice(alphas);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.resources.alphas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.alpha_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 16 bytes per RGBA32Uint texel (4 u32s × 4 bytes each), which is equivalent to
                // a bit shift of 4.
                bytes_per_row: Some(texture_width << 4),
                rows_per_image: Some(texture_height),
            },
            wgpu::Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
        );

        // Upload encoded paints to the texture
        let encoded_paints_texture = &self.resources.encoded_paints_texture;
        let encoded_paints_texture_width = encoded_paints_texture.width();
        let encoded_paints_texture_height = encoded_paints_texture.height();

        let encoded_paints_bytes = bytemuck::cast_slice(&encoded_paints);
        self.encoded_paints_data[0..encoded_paints_bytes.len()]
            .copy_from_slice(encoded_paints_bytes);
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
            wgpu::Extent3d {
                width: encoded_paints_texture_width,
                height: encoded_paints_texture_height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn maybe_resize_alphas_tex(
        &mut self,
        device: &Device,
        alphas: &[u8],
        max_texture_dimension_2d: u32,
    ) {
        let required_alpha_height = u32::try_from(alphas.len())
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

            // Resize the alpha texture staging buffer.
            let required_alpha_size = (max_texture_dimension_2d * required_alpha_height) << 4;
            self.alpha_data.resize(required_alpha_size as usize, 0);
            // The alpha texture encodes 16 1-byte alpha values per texel, with 4 alpha values packed in each channel
            let alphas_texture =
                Self::make_alphas_texture(device, max_texture_dimension_2d, required_alpha_height);
            self.resources.alphas_texture = alphas_texture;

            // Since the alpha texture has changed, we need to update the clip bind groups.
            self.resources.slot_bind_groups = Self::make_strip_bind_groups(
                device,
                &self.strip_bind_group_layout,
                &self
                    .resources
                    .alphas_texture
                    .create_view(&Default::default()),
                &self.resources.slot_config_buffer,
                &self.resources.view_config_buffer,
                &self.resources.slot_texture_views,
            );
        }
    }

    fn maybe_resize_encoded_paints_tex(
        &mut self,
        device: &Device,
        encoded_paints: &[GpuEncodedImage],
        max_texture_dimension_2d: u32,
    ) {
        let required_encoded_paints_height = u32::try_from(size_of_val(encoded_paints))
            .unwrap()
            .div_ceil(max_texture_dimension_2d << 4);
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
            let encoded_paints_texture = Self::make_encoded_paints_texture(
                device,
                max_texture_dimension_2d,
                required_encoded_paints_height,
            );
            self.resources.encoded_paints_texture = encoded_paints_texture;

            // Since the encoded paints texture has changed, we need to update the strip bind groups.
            self.resources.encoded_paints_bind_group = Self::make_encoded_paints_bind_group(
                device,
                &self.encoded_paints_bind_group_layout,
                &self
                    .resources
                    .encoded_paints_texture
                    .create_view(&Default::default()),
            );
        }
    }

    /// Upload the strip data by creating and assigning a new `self.resources.strips_buffer`.
    fn upload_strips(&mut self, device: &Device, queue: &Queue, strips: &[GpuStrip]) {
        let required_strips_size = size_of_val(strips) as u64;
        self.resources.strips_buffer = Self::make_strips_buffer(device, required_strips_size);
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
        render_pass.set_vertex_buffer(0, self.programs.resources.strips_buffer.slice(..));
        render_pass.draw(0..4, 0..u32::try_from(strips.len()).unwrap());
    }

    /// Clear specific slots from a slot texture.
    fn do_clear_slots_render_pass(&mut self, ix: usize, slot_indices: &[u32]) {
        if slot_indices.is_empty() {
            return;
        }

        let resources = &mut self.programs.resources;
        let size = mem::size_of_val(slot_indices) as u64;
        // TODO: We currently allocate a new strips buffer for each render pass. A more efficient
        // approach would be to re-use buffers or slices of a larger buffer.
        resources.clear_slot_indices_buffer =
            Programs::make_clear_slot_indices_buffer(self.device, size);
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
    fn render_strips(
        &mut self,
        strips: &[GpuStrip],
        target_index: usize,
        load_op: crate::schedule::LoadOp,
    ) {
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

    /// Write image data to the atlas texture at the specified offset.
    fn write_to_atlas(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &wgpu::Texture,
        offset: [u32; 2],
        width: u32,
        height: u32,
    );
}

/// Implementation for `wgpu::Texture` - uses texture-to-texture copy
impl AtlasWriter for wgpu::Texture {
    fn width(&self) -> u32 {
        self.width()
    }

    fn height(&self) -> u32 {
        self.height()
    }

    fn write_to_atlas(
        &self,
        _device: &Device,
        _queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &wgpu::Texture,
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
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
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

    fn write_to_atlas(
        &self,
        _device: &Device,
        queue: &Queue,
        _encoder: &mut CommandEncoder,
        atlas_texture: &wgpu::Texture,
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
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            self.data_as_u8_slice(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Implementation for `Arc<Pixmap>`
impl AtlasWriter for alloc::sync::Arc<Pixmap> {
    fn width(&self) -> u32 {
        self.as_ref().width() as u32
    }

    fn height(&self) -> u32 {
        self.as_ref().height() as u32
    }

    fn write_to_atlas(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &wgpu::Texture,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        self.as_ref()
            .write_to_atlas(device, queue, encoder, atlas_texture, offset, width, height);
    }
}
