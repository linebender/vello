// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The GPU parts of a hybrid CPU/GPU rendering engine.
//!
use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupLayout, BlendState, Buffer, ColorTargetState, ColorWrites, Device,
    PipelineCompilationOptions, RenderPipeline, TextureFormat, util::DeviceExt,
};

pub struct Renderer {
    pub render_bind_group_layout: BindGroupLayout,
    pub render_pipeline: RenderPipeline,
    pub render_bind_group: BindGroup,
    pub config_buf: Buffer,
    pub strips_buf: Buffer,
    pub alpha_buf: Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub strip_height: u32,
}

pub struct GpuRenderBuffers {
    pub strips: Vec<GpuStrip>,
    pub alphas: Vec<u32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct GpuStrip {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub dense_width: u16,
    pub col: u32,
    pub rgba: u32,
}

impl Renderer {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        config: &Config,
        bufs: &GpuRenderBuffers,
    ) -> Self {
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render.wgsl").into()),
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

        let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("config"),
            contents: bytemuck::bytes_of(config),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let strips_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("strip"),
            contents: bytemuck::cast_slice(&bufs.strips),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let alpha_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("alpha"),
            contents: bytemuck::cast_slice(&bufs.alphas),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: alpha_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: strips_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            render_bind_group_layout,
            render_pipeline,
            render_bind_group,
            config_buf,
            strips_buf,
            alpha_buf,
        }
    }

    pub fn prepare(&self, device: &Device, config: &Config, bufs: &GpuRenderBuffers) {
        // TODO: update buffers
    }

    pub fn render(
        &self,
        device: &Device,
        surface: &wgpu::Surface,
        queue: &wgpu::Queue,
        bufs: &GpuRenderBuffers,
    ) {
        let frame = surface
            .get_current_texture()
            .expect("error getting texture from swap chain");

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.render_bind_group, &[]);
            let n_strips = bufs.strips.len().try_into().expect("too many strips");
            rpass.draw(0..4, 0..n_strips);
        }
        queue.submit(Some(encoder.finish()));
        frame.present();
    }
}
