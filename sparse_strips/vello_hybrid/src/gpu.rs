// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The GPU parts of a hybrid CPU/GPU rendering engine.

use bytemuck::{Pod, Zeroable};
use peniko::BrushRef;
use wgpu::{
    BindGroupLayout, BlendState, ColorTargetState, ColorWrites, Device, PipelineCompilationOptions,
    RenderPipeline, TextureFormat,
};

use crate::{
    render::RenderContext,
    wide_tile::{Cmd, STRIP_HEIGHT, WIDE_TILE_WIDTH},
};

/// Resources common to GPU renders.
pub struct GpuSession {
    pub render_bind_group_layout: BindGroupLayout,
    pub render_pipeline: RenderPipeline, /*  */
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct Strip {
    x: u16,
    y: u16,
    width: u16,
    dense_width: u16,
    col: u32,
    rgba: u32,
}

/// A render context for a single frame.
///
/// This will eventually get a `RenderCtx` trait impl.
pub struct GpuRenderCtx {
    // At the moment, we take the entire cpu-sparse render context,
    // but we might split that up.
    inner: RenderContext,
}

/// The buffers from a render.
///
/// This being a struct is based on a model where all the buffers are uploaded
/// up front. That will be replaced by the "submit early and often" model.
pub struct GpuRenderBufs {
    pub strips: Vec<Strip>,
    pub alphas: Vec<u32>,
}

impl GpuSession {
    pub fn new(device: &Device, format: TextureFormat) -> Self {
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("../shader/render.wgsl").into()),
        });
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                buffers: &[],
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
        }
    }
}

impl GpuRenderCtx {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            inner: RenderContext::new(width, height),
        }
    }

    pub fn harvest(&self) -> GpuRenderBufs {
        let mut strips = Vec::new();
        let width_tiles = (self.inner.width).div_ceil(WIDE_TILE_WIDTH);
        let height_tiles = (self.inner.height).div_ceil(STRIP_HEIGHT);
        for y in 0..height_tiles {
            for x in 0..width_tiles {
                let tile = &self.inner.tiles[y * width_tiles + x];
                let tile_x = x * WIDE_TILE_WIDTH;
                let tile_y = y * STRIP_HEIGHT;
                let bg = tile.bg.to_rgba8().to_u32();
                if bg != 0 {
                    let strip = Strip {
                        x: tile_x as u16,
                        y: tile_y as u16,
                        width: WIDE_TILE_WIDTH as u16,
                        dense_width: 0,
                        col: 0,
                        rgba: bg,
                    };
                    strips.push(strip);
                }
                for cmd in &tile.cmds {
                    match cmd {
                        Cmd::Fill(fill) => {
                            let strip = Strip {
                                x: (tile_x as u32 + fill.x) as u16,
                                y: tile_y as u16,
                                width: fill.width as u16,
                                dense_width: 0,
                                col: 0,
                                rgba: fill.color.to_rgba8().to_u32(),
                            };
                            strips.push(strip);
                        }
                        Cmd::Strip(cmd_strip) => {
                            let strip = Strip {
                                x: (tile_x as u32 + cmd_strip.x) as u16,
                                y: tile_y as u16,
                                width: cmd_strip.width as u16,
                                dense_width: cmd_strip.width as u16,
                                col: cmd_strip.alpha_ix as u32,
                                rgba: cmd_strip.color.to_rgba8().to_u32(),
                            };
                            strips.push(strip);
                        }
                    }
                }
            }
        }
        GpuRenderBufs {
            strips,
            alphas: self.inner.alphas.clone(),
        }
    }
}

// This block will eventually turn into an impl of RenderCtx.
impl GpuRenderCtx {
    pub fn fill(&mut self, path: &crate::common::Path, brush: BrushRef<'_>) {
        self.inner.fill(path, brush);
    }

    pub fn stroke(
        &mut self,
        path: &crate::common::Path,
        stroke: &peniko::kurbo::Stroke,
        brush: BrushRef<'_>,
    ) {
        self.inner.stroke(path, stroke, brush);
    }
}
