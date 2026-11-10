// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The clear pass.

use crate::render::wgpu::RenderTargetConfig;
use crate::render::wgpu::passes::RendererContext;
use crate::{ClearSettings, render::common::GpuClearInstance, target::DrawPassTarget};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::geometry::RectU16;
use wgpu::{
    ColorTargetState, ColorWrites, CommandEncoder, Device, PipelineCompilationOptions,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, TextureView, util::DeviceExt,
};

/// Pipelines for clearing rectangles of layer textures and the root target.
#[derive(Debug)]
pub(crate) struct ClearPipelines {
    /// Layer-clear pipeline.
    pub(crate) layer: RenderPipeline,
    /// User-target rectangle-clear pipeline.
    pub(crate) root: RenderPipeline,
}

impl ClearPipelines {
    pub(crate) fn new(
        device: &Device,
        render_target_config: &RenderTargetConfig,
        pipeline_cache: Option<&wgpu::PipelineCache>,
    ) -> Self {
        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clear Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::CLEAR.into()),
        });

        let clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Clear Pipeline Layout"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });

        let clear_vertex_state = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuClearInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
                0 => Uint32x2,
                1 => Uint32x2,
                2 => Uint32x2,
                3 => Uint32,
            ],
        };
        let create_clear_pipeline = |label, format| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&clear_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &clear_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(clear_vertex_state.clone())],
                    compilation_options: PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &clear_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format,
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
                multiview_mask: None,
                cache: pipeline_cache,
            })
        };
        let clear_pipeline =
            create_clear_pipeline("Clear Pipeline", wgpu::TextureFormat::Rgba8Unorm);
        let root_clear_pipeline =
            create_clear_pipeline("Root Clear Pipeline", render_target_config.format);

        Self {
            layer: clear_pipeline,
            root: root_clear_pipeline,
        }
    }
}

impl RendererContext<'_> {
    pub(crate) fn clear_full_target(
        encoder: &mut CommandEncoder,
        view: &TextureView,
        color: wgpu::Color,
    ) {
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Clear Target"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
    }

    pub(crate) fn clear_pass_inner(&mut self, target: DrawPassTarget, settings: ClearSettings<'_>) {
        let color = settings.clear_color();

        let view = match target {
            DrawPassTarget::Root(_) => self.view,
            DrawPassTarget::Layer(target) => self.resources.layer_view(target),
        };
        let rects = match settings {
            ClearSettings::Viewport { .. } => {
                Self::clear_full_target(self.encoder, view, clear_color(color));

                return;
            }
            ClearSettings::Rects { rects, .. } => rects,
        };

        let (target_size, pipeline) = match target {
            DrawPassTarget::Root(_) => (
                [
                    self.resources.render_size.width,
                    self.resources.render_size.height,
                ],
                &self.programs.clear.root,
            ),
            DrawPassTarget::Layer(_) => {
                let texture_size = self.texture_size();
                (
                    [texture_size.width(), texture_size.height()],
                    &self.programs.clear.layer,
                )
            }
        };
        let bounds = RectU16::new(0, 0, target_size[0], target_size[1]);
        let target_size = target_size.map(u32::from);
        let color = color.premultiply().to_rgba8().to_u32();
        self.scratch_buffers.clear_instances.clear();
        self.scratch_buffers.clear_instances.extend(
            rects
                .iter()
                .map(|rect| rect.intersect(bounds))
                .filter(|rect| !rect.is_empty())
                .map(|rect| GpuClearInstance {
                    origin: [u32::from(rect.x0), u32::from(rect.y0)],
                    size: [u32::from(rect.width()), u32::from(rect.height())],
                    target_size,
                    color,
                }),
        );

        if self.scratch_buffers.clear_instances.is_empty() {
            return;
        }

        let clear_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Clear Buffer"),
                contents: bytemuck::cast_slice(&self.scratch_buffers.clear_instances),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Clear Rects"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(pipeline);
        render_pass.set_vertex_buffer(0, clear_buffer.slice(..));
        render_pass.draw(
            0..4,
            0..u32::try_from(self.scratch_buffers.clear_instances.len()).unwrap(),
        );
    }
}
pub(crate) fn clear_color(color: AlphaColor<Srgb>) -> wgpu::Color {
    let [r, g, b, a] = color.premultiply().components;
    wgpu::Color {
        r: f64::from(r),
        g: f64::from(g),
        b: f64::from(b),
        a: f64::from(a),
    }
}
