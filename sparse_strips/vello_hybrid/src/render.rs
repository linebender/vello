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

use std::fmt::Debug;

use bytemuck::{Pod, Zeroable};
use vello_common::tile::Tile;
use wgpu::{
    BindGroup, BindGroupLayout, BlendState, Buffer, ColorTargetState, ColorWrites, CommandEncoder,
    Device, PipelineCompilationOptions, Queue, RenderPass, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, Texture, TextureView, util::DeviceExt,
};

use crate::{scene::Scene, schedule::Schedule};

/// Dimensions of the rendering target
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RenderSize {
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
}

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

/// Contains all GPU resources needed for rendering
///
/// This struct contains the GPU resources that may be reallocated depending
/// on the scene. Resources that are created once at startup are simply in
/// `Renderer`.
#[derive(Debug)]
struct GpuResources {
    /// Buffer for strip data
    pub strips_buffer: Buffer,
    /// Texture for alpha values
    pub alphas_texture: Texture,
    /// Bind group for rendering
    pub render_bind_group: BindGroup,
    /// Buffer for config data
    pub config_buffer: Buffer,
    // Bind groups for rendering with clip buffers
    //pub clip_bind_groups: [BindGroup; 3],
}

/// GPU renderer for the hybrid rendering system
///
/// This struct contains GPU resources that are created once at startup and
/// are never reallocated or rebuilt.
#[derive(Debug)]
pub struct Renderer {
    /// Bind group layout for rendering
    pub render_bind_group_layout: BindGroupLayout,
    /// Pipeline for rendering
    pub render_pipeline: RenderPipeline,
    /// Bind group layout for clip draws
    pub clip_bind_group_layout: BindGroupLayout,
    /// Pipeline for rendering clip draws
    pub clip_pipeline: RenderPipeline,
    /// Clip temporary textures
    pub clip_textures: [Texture; 2],
    /// GPU resources for rendering (created during prepare)
    resources: Option<GpuResources>,

    /// Scratch buffer for staging alpha texture data.
    alpha_data: Vec<u8>,

    /// Dimensions of the rendering target
    render_size: RenderSize,
}

/// Contains the data needed for rendering
#[derive(Debug, Default)]
pub struct RenderData {
    /// GPU strips to be rendered
    pub strips: Vec<GpuStrip>,
    /// Alpha values used in rendering
    pub alphas: Vec<u8>,
}

/// Configuration for the GPU renderer
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct Config {
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
    /// Height of a strip in the rendering
    pub strip_height: u32,
    /// Number of trailing zeros in `alphas_tex_width` (log2 of width).
    /// Pre-calculated on CPU since downlevel targets do not support `firstTrailingBit`.
    pub alphas_tex_width_bits: u32,
}

/// Represents a GPU strip for rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct GpuStrip {
    /// X coordinate of the strip
    pub x: u16,
    /// Y coordinate of the strip
    pub y: u16,
    /// Width of the strip
    pub width: u16,
    /// Width of the portion where alpha blending should be applied.
    pub dense_width: u16,
    /// Column-index into the alpha texture where this strip's alpha values begin.
    ///
    /// There are [`Config::strip_height`] alpha values per column.
    pub col: u32,
    /// RGBA color value
    pub rgba: u32,
}

/// A struct containing references to the many objects needed to get work
/// scheduled onto the GPU.
pub(crate) struct RendererJunk<'a> {
    renderer: &'a mut Renderer,
    device: &'a Device,
    queue: &'a Queue,
    encoder: &'a mut CommandEncoder,
    view: &'a TextureView,
}

impl GpuStrip {
    /// Vertex attributes for the strip
    pub fn vertex_attributes() -> [wgpu::VertexAttribute; 4] {
        wgpu::vertex_attr_array![
            0 => Uint32,
            1 => Uint32,
            2 => Uint32,
            3 => Uint32,
        ]
    }
}

impl Renderer {
    /// Creates a new renderer
    ///
    /// The target parameter determines if we render to a window or headless
    pub fn new(device: &Device, render_target_config: &RenderTargetConfig) -> Self {
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sparse_strip_renderer.wgsl").into(),
            ),
        });
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
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
                ],
            });
        let clip_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<GpuStrip>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &GpuStrip::vertex_attributes(),
                }],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
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
        let clip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/sparse_strip_clip.wgsl").into(),
            ),
        });
        let clip_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&clip_bind_group_layout],
            push_constant_ranges: &[],
        });
        let clip_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&clip_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &clip_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<GpuStrip>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &GpuStrip::vertex_attributes(),
                }],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &clip_shader,
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
        let clip_textures = std::array::from_fn(|_| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("clip temp texture"),
                size: wgpu::Extent3d {
                    width: 256, // TODO: make configurable
                    height: 1024,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
        });

        Self {
            render_bind_group_layout,
            render_pipeline,
            clip_bind_group_layout,
            clip_pipeline,
            resources: None,
            alpha_data: Vec::new(),
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            clip_textures,
        }
    }

    fn make_strips_buffer(&self, device: &Device, required_strips_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Strips Buffer"),
            size: required_strips_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_config_buffer(
        &self,
        device: &Device,
        render_size: &RenderSize,
        max_texture_dimension_2d: u32,
    ) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&Config {
                width: render_size.width,
                height: render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn make_alphas_texture(
        &self,
        device: &Device,
        max_texture_dimension_2d: u32,
        alpha_texture_height: u32,
    ) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Alpha Texture"),
            size: wgpu::Extent3d {
                width: max_texture_dimension_2d,
                height: alpha_texture_height,
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

    fn make_render_bind_group(
        &self,
        device: &Device,
        alphas_texture: &Texture,
        config_buffer: &Buffer,
    ) -> BindGroup {
        let alphas_texture_view =
            alphas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &self.render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&alphas_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Prepare the GPU buffers for rendering, given alphas
    ///
    /// Does not guarantee that the strip buffer is big enough
    fn prepare_alphas(
        &mut self,
        device: &Device,
        queue: &Queue,
        alphas: &[u8],
        new_render_size: &RenderSize,
        est_strip_count: usize,
    ) {
        let required_strips_size = size_of::<GpuStrip>() as u64 * est_strip_count as u64;
        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        if self.resources.is_none() {
            let strips_buffer = self.make_strips_buffer(device, required_strips_size);
            let alpha_len = alphas.len();
            // There are 16 1-byte alpha values per texel.
            let alpha_texture_height =
                (u32::try_from(alpha_len).unwrap()).div_ceil(max_texture_dimension_2d * 16);

            assert!(
                alpha_texture_height <= max_texture_dimension_2d,
                "Alpha texture height exceeds max texture dimensions"
            );

            // Resize the alpha texture staging buffer.
            self.alpha_data.resize(
                (max_texture_dimension_2d * alpha_texture_height * 16) as usize,
                0,
            );
            // The alpha texture encodes 16 1-byte alpha values per texel, with 4 alpha values packed in each channel
            let alphas_texture =
                self.make_alphas_texture(device, max_texture_dimension_2d, alpha_texture_height);
            let config_buffer =
                self.make_config_buffer(device, new_render_size, max_texture_dimension_2d);

            let render_bind_group =
                self.make_render_bind_group(device, &alphas_texture, &config_buffer);
            self.resources = Some(GpuResources {
                strips_buffer,
                alphas_texture,
                render_bind_group,
                config_buffer,
            });
        } else {
            // Update existing resources as needed
            let alpha_len = alphas.len();
            // There are 16 1-byte alpha values per texel.
            let required_alpha_height =
                (u32::try_from(alpha_len).unwrap()).div_ceil(max_texture_dimension_2d * 16);
            let required_alpha_size = max_texture_dimension_2d * required_alpha_height * 16;

            let current_alpha_size = {
                let alphas_texture = &self.resources.as_ref().unwrap().alphas_texture;
                alphas_texture.width() * alphas_texture.height() * 16
            };
            if required_alpha_size > current_alpha_size {
                assert!(
                    required_alpha_height <= max_texture_dimension_2d,
                    "Alpha texture height exceeds max texture dimensions"
                );

                // Resize the alpha texture staging buffer.
                self.alpha_data.resize(
                    (max_texture_dimension_2d * required_alpha_height * 16) as usize,
                    0,
                );
                // The alpha texture encodes 16 1-byte alpha values per texel, with 4 alpha values packed in each channel
                let alphas_texture = self.make_alphas_texture(
                    device,
                    max_texture_dimension_2d,
                    required_alpha_height,
                );
                let config_buffer = &self.resources.as_ref().unwrap().config_buffer;
                let render_bind_group =
                    self.make_render_bind_group(device, &alphas_texture, config_buffer);
                let resources = self.resources.as_mut().unwrap();
                resources.alphas_texture = alphas_texture;
                resources.render_bind_group = render_bind_group;
            }
        }

        // Resources have been created by now.
        let resources = self.resources.as_ref().unwrap();

        // Update config buffer if dimensions changed.
        // We don't need to initialize a new config buffer because it's fixed size (uniform buffer).
        if self.render_size != *new_render_size {
            let config = Config {
                width: new_render_size.width,
                height: new_render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
            };
            queue.write_buffer(&resources.config_buffer, 0, bytemuck::bytes_of(&config));
            self.render_size = new_render_size.clone();
        }

        // Prepare alpha data for the texture with 16 1-byte alpha values per texel (4 per channel)
        let texture_width = resources.alphas_texture.width();
        let texture_height = resources.alphas_texture.height();
        assert!(
            alphas.len() <= (texture_width * texture_height * 16) as usize,
            "Alpha texture dimensions are too small to fit the alpha data"
        );
        // After this copy to `self.alpha_data`, there may be stale trailing alpha values. These
        // are not sampled, so can be left as-is.
        self.alpha_data[0..alphas.len()].copy_from_slice(alphas);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &resources.alphas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.alpha_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 16 bytes per RGBA32Uint texel (4 u32s × 4 bytes each)
                bytes_per_row: Some(texture_width * 16),
                rows_per_image: Some(texture_height),
            },
            wgpu::Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload the strip data
    fn upload_strips(&mut self, device: &Device, queue: &Queue, strips: &[GpuStrip]) {
        let required_strips_size = size_of_val(strips) as u64;

        if required_strips_size > self.resources.as_ref().unwrap().strips_buffer.size() {
            self.resources.as_mut().unwrap().strips_buffer =
                self.make_strips_buffer(device, required_strips_size);
        }

        // TODO: Explore using `write_buffer_with` to avoid copying the data twice
        queue.write_buffer(
            &self.resources.as_ref().unwrap().strips_buffer,
            0,
            bytemuck::cast_slice(strips),
        );
    }

    /// Prepare the GPU buffers for rendering
    pub fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        render_data: &RenderData,
        new_render_size: &RenderSize,
    ) {
        self.prepare_alphas(
            device,
            queue,
            &render_data.alphas,
            new_render_size,
            render_data.strips.len(),
        );
        self.upload_strips(device, queue, &render_data.strips);
    }

    /// Render `scene` into the provided render pass.
    ///
    /// You must call [`prepare`](Self::prepare) with this scene before
    /// calling `render`.
    /// The provided pass can be rendering to a surface, or to a "off-screen" buffer.
    pub fn render(&mut self, render_data: &RenderData, render_pass: &mut RenderPass<'_>) {
        // TODO: Consider API that forces the user to call `prepare` before `render`.
        // For example, `prepare` could return some struct that is consumed by `render`.
        let resources = self
            .resources
            .as_ref()
            .expect("`prepare` should be called before `render`");
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &resources.render_bind_group, &[]);
        render_pass.set_vertex_buffer(0, resources.strips_buffer.slice(..));
        let strips_to_draw = render_data.strips.len();
        render_pass.draw(0..4, 0..u32::try_from(strips_to_draw).unwrap());
    }

    /// Render `scene` into the provided command encoder.
    ///
    /// This method creates GPU resources as needed, and schedules potentially multiple
    /// render passes.
    pub fn render2(
        &mut self,
        scene: &Scene,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        view: &TextureView,
    ) {
        let render_data = scene.prepare_render_data();
        // For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        self.prepare_alphas(
            device,
            queue,
            &scene.alphas,
            render_size,
            render_data.strips.len(),
        );
        let mut junk = RendererJunk {
            renderer: self,
            device,
            queue,
            encoder,
            view,
        };
        // TODO: make this configurable, and make it match the allocation
        let n_slots = 1024;
        let mut schedule = Schedule::new(n_slots);
        schedule.do_scene(&mut junk, scene);
    }
}

impl RendererJunk<'_> {
    pub(crate) fn do_render_pass(&mut self, strips: &[GpuStrip], round: usize, _ix: usize) {
        self.renderer.upload_strips(self.device, self.queue, strips);
        let load = if round == 0 {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        } else {
            wgpu::LoadOp::Load
        };
        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("render to texture pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                // TODO: view is clip buffer[ix] for ix != 2
                view: self.view,
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
        let resources = self
            .renderer
            .resources
            .as_ref()
            .expect("`prepare` should be called before `render`");
        render_pass.set_pipeline(&self.renderer.render_pipeline);
        render_pass.set_bind_group(0, &resources.render_bind_group, &[]);
        render_pass.set_vertex_buffer(0, resources.strips_buffer.slice(..));
        let strips_to_draw = strips.len();
        render_pass.draw(0..4, 0..u32::try_from(strips_to_draw).unwrap());
    }
}
