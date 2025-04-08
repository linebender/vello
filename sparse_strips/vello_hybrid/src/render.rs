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

        Self {
            render_bind_group_layout,
            render_pipeline,
            resources: None,
            alpha_data: Vec::new(),
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
        }
    }

    /// Prepare the GPU buffers for rendering
    pub fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        new_render_size: &RenderSize,
    ) {
        let render_data = scene.prepare_render_data();
        let required_strips_size = size_of::<GpuStrip>() as u64 * render_data.strips.len() as u64;
        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;

        let (needs_new_strips_buffer, needs_new_alpha_texture, needs_new_config) =
            match &self.resources {
                Some(resources) => {
                    let strips_too_small = required_strips_size > resources.strips_buffer.size();

                    let alpha_len = render_data.alphas.len();
                    // There are 16 1-byte alpha values per texel.
                    let required_alpha_height =
                        (u32::try_from(alpha_len).unwrap()).div_ceil(max_texture_dimension_2d * 16);
                    let required_alpha_size = max_texture_dimension_2d * required_alpha_height * 16;

                    let current_alpha_size =
                        resources.alphas_texture.width() * resources.alphas_texture.height() * 16;
                    let alpha_too_small = required_alpha_size > current_alpha_size;

                    let dimensions_changed = self.render_size != *new_render_size;

                    (strips_too_small, alpha_too_small, dimensions_changed)
                }
                // self.resources is None if prepare has not been called yet
                None => (true, true, true),
            };

        if needs_new_strips_buffer || needs_new_alpha_texture {
            // Create strips buffer if it doesn't exist, or reuse existing strips buffer
            let strips_buffer = if needs_new_strips_buffer {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Strips Buffer"),
                    size: required_strips_size,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            } else {
                self.resources
                    .as_ref()
                    .expect("Strips buffer not initialized")
                    .strips_buffer
                    .clone()
            };

            // Create config buffer if it doesn't exist, or reuse existing config buffer
            let config_buffer = if self.resources.is_none() {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Config Buffer"),
                    contents: bytemuck::bytes_of(&Config {
                        width: new_render_size.width,
                        height: new_render_size.height,
                        strip_height: Tile::HEIGHT.into(),
                        alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
                    }),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
            } else {
                self.resources
                    .as_ref()
                    .expect("Config buffer not initialized")
                    .config_buffer
                    .clone()
            };

            // Create alpha texture if it doesn't exist, or reuse existing alpha texture
            let (alphas_texture, render_bind_group) = if needs_new_alpha_texture {
                let alpha_len = render_data.alphas.len();
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
                            resource: config_buffer.as_entire_binding(),
                        },
                    ],
                });
                (alphas_texture, render_bind_group)
            } else {
                let resources = self.resources.as_ref().expect("Resources not initialized");
                (
                    resources.alphas_texture.clone(),
                    resources.render_bind_group.clone(),
                )
            };

            self.resources = Some(GpuResources {
                strips_buffer,
                alphas_texture,
                render_bind_group,
                config_buffer,
            });
        };

        // Update config buffer if dimensions changed and config buffer exists.
        // We don't need to initialize a new config buffer because it's fixed size (uniform buffer).
        if needs_new_config && self.resources.is_some() {
            let config = Config {
                width: new_render_size.width,
                height: new_render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: max_texture_dimension_2d.trailing_zeros(),
            };
            queue.write_buffer(
                &self.resources.as_ref().unwrap().config_buffer,
                0,
                bytemuck::bytes_of(&config),
            );
            self.render_size = new_render_size.clone();
        }

        // Resources are created in above blocks.
        let resources = self.resources.as_ref().unwrap();

        // TODO: Explore using `write_buffer_with` to avoid copying the data twice
        queue.write_buffer(
            &resources.strips_buffer,
            0,
            bytemuck::cast_slice(&render_data.strips),
        );

        // Prepare alpha data for the texture with 16 1-byte alpha values per texel (4 per channel)
        let texture_width = resources.alphas_texture.width();
        let texture_height = resources.alphas_texture.height();
        assert!(
            render_data.alphas.len() <= (texture_width * texture_height * 16) as usize,
            "Alpha texture dimensions are too small to fit the alpha data"
        );
        // After this copy to `self.alpha_data`, there may be stale trailing alpha values. These
        // are not sampled, so can be left as-is.
        self.alpha_data[0..render_data.alphas.len()].copy_from_slice(&render_data.alphas);

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

    /// Render `scene` into the provided render pass.
    ///
    /// You must call [`prepare`](Self::prepare) with this scene before
    /// calling `render`.
    /// The provided pass can be rendering to a surface, or to a "off-screen" buffer.
    pub fn render(&mut self, scene: &Scene, render_pass: &mut RenderPass<'_>) {
        // TODO: Consider API that forces the user to call `prepare` before `render`.
        // For example, `prepare` could return some struct that is consumed by `render`.
        let resources = &self
            .resources
            .as_ref()
            .expect("`prepare` should be called before `render`");
        let render_data = scene.prepare_render_data();
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &resources.render_bind_group, &[]);
        render_pass.set_vertex_buffer(0, resources.strips_buffer.slice(..));
        let strips_to_draw = render_data.strips.len();
        render_pass.draw(0..4, 0..u32::try_from(strips_to_draw).unwrap());
    }
}
