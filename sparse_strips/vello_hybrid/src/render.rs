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
    BindGroup, BindGroupLayout, BlendState, Buffer, ColorTargetState, ColorWrites, Device,
    PipelineCompilationOptions, Queue, RenderPass, RenderPipeline, Texture, util::DeviceExt,
};

use crate::scene::Scene;

/// Parameters for the renderer
#[derive(Debug)]
pub struct RenderParams {
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
}

/// Options for the renderer
#[derive(Debug)]
pub struct RendererOptions {}

/// Contains all GPU resources needed for rendering
#[derive(Debug)]
struct GpuResources {
    /// Buffer for strip data
    pub strips_buffer: Buffer,
    /// Texture for alpha values
    pub alphas_texture: Texture,
    /// Bind group for rendering
    pub render_bind_group: BindGroup,
}

/// GPU renderer for the hybrid rendering system
#[derive(Debug)]
pub struct Renderer {
    /// Bind group layout for rendering
    pub render_bind_group_layout: BindGroupLayout,
    /// Pipeline for rendering
    pub render_pipeline: RenderPipeline,
    /// GPU resources for rendering (created during prepare)
    resources: Option<GpuResources>,
}

/// Contains the data needed for rendering
#[derive(Debug, Default)]
pub struct RenderData {
    /// GPU strips to be rendered
    pub strips: Vec<GpuStrip>,
    /// Alpha values used in rendering
    pub alphas: Vec<u32>,
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
    /// Align to 16 bytes for WebGL2 compatibility
    pub _padding: u32,
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
    /// Index into the alpha texture where this strip's alpha values begin.
    pub col: u32,
    /// RGBA color value
    pub rgba: u32,
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
    pub fn new(device: &Device, _options: &RendererOptions) -> Self {
        let format = wgpu::TextureFormat::Bgra8Unorm;

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
                    format,
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

        Self {
            render_bind_group_layout,
            render_pipeline,
            resources: None,
        }
    }

    /// Prepare the GPU buffers for rendering
    pub fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        render_params: &RenderParams,
    ) {
        let render_data = scene.prepare_render_data();
        let required_strips_size = size_of::<GpuStrip>() as u64 * render_data.strips.len() as u64;

        let (needs_new_strips_buffer, needs_new_alpha_texture) = match &self.resources {
            Some(resources) => {
                let strips_too_small = required_strips_size > resources.strips_buffer.size();

                let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
                let alpha_len = render_data.alphas.len();
                // 4 alpha values u32 each per texel
                let required_alpha_height =
                    (u32::try_from(alpha_len).unwrap()).div_ceil(max_texture_dimension_2d * 4);
                let required_alpha_size = max_texture_dimension_2d * required_alpha_height * 4;

                let current_alpha_size =
                    resources.alphas_texture.width() * resources.alphas_texture.height() * 4;
                let alpha_too_small = required_alpha_size > current_alpha_size;

                (strips_too_small, alpha_too_small)
            }
            None => (true, true),
        };

        if needs_new_strips_buffer || needs_new_alpha_texture {
            let strips_buffer = if needs_new_strips_buffer {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Strips Buffer"),
                    size: required_strips_size,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            } else {
                // Reuse existing buffer if it's big enough
                self.resources
                    .as_ref()
                    .expect("Strips buffer not found")
                    .strips_buffer
                    .clone()
            };
            let (alphas_texture, render_bind_group) = if needs_new_alpha_texture {
                let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
                let alpha_len = render_data.alphas.len();
                // 4 alpha values u32 each per texel
                let alpha_texture_height =
                    (u32::try_from(alpha_len).unwrap()).div_ceil(max_texture_dimension_2d * 4);

                // Ensure dimensions don't exceed WebGL2 limits
                assert!(
                    alpha_texture_height <= max_texture_dimension_2d,
                    "Alpha texture height exceeds WebGL2 limit"
                );

                let alphas_texture = device.create_texture(&wgpu::TextureDescriptor {
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
                });
                let alphas_texture_view =
                    alphas_texture.create_view(&wgpu::TextureViewDescriptor::default());

                let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Config Buffer"),
                    contents: bytemuck::bytes_of(&Config {
                        width: render_params.width,
                        height: render_params.height,
                        strip_height: Tile::HEIGHT.into(),
                        _padding: 0,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Render Bind Group"),
                    layout: &self.render_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&alphas_texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: config_buf.as_entire_binding(),
                        },
                    ],
                });
                (alphas_texture, render_bind_group)
            } else {
                let resources = self.resources.as_ref().unwrap();
                (
                    resources.alphas_texture.clone(),
                    resources.render_bind_group.clone(),
                )
            };
            self.resources = Some(GpuResources {
                strips_buffer,
                alphas_texture,
                render_bind_group,
            });
        };

        // Now that we have resources, we can update the data
        if let Some(resources) = &self.resources {
            // TODO: Explore using `write_buffer_with` to avoid copying the data twice
            queue.write_buffer(
                &resources.strips_buffer,
                0,
                bytemuck::cast_slice(&render_data.strips),
            );

            // Prepare alpha data for the texture with 4 alpha values per texel
            let texture_width = resources.alphas_texture.width();
            let texture_height = resources.alphas_texture.height();
            assert!(
                render_data.alphas.len() <= (texture_width * texture_height * 4) as usize,
                "Alpha texture dimensions are too small to fit the alpha data"
            );
            let mut alpha_data = vec![0_u32; render_data.alphas.len()];
            alpha_data[..].copy_from_slice(&render_data.alphas);
            alpha_data.resize((texture_width * texture_height * 4) as usize, 0);

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &resources.alphas_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&alpha_data),
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
    }

    /// Render `scene` into the provided render pass.
    ///
    /// You must call [`prepare`](Self::prepare) with this scene before
    /// calling `render`.
    /// The provided pass can be rendering to a surface, or to a "off-screen" buffer.
    pub fn render(
        &mut self,
        scene: &Scene,
        render_pass: &mut RenderPass<'_>,
        _render_params: &RenderParams,
    ) {
        // If we don't have the required resources, return empty data
        let Some(resources) = &self.resources else {
            return;
        };
        let render_data = scene.prepare_render_data();
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &resources.render_bind_group, &[]);
        render_pass.set_vertex_buffer(0, resources.strips_buffer.slice(..));
        let strips_to_draw = render_data.strips.len();
        render_pass.draw(0..4, 0..u32::try_from(strips_to_draw).unwrap());
    }
}
