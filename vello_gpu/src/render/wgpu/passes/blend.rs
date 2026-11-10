// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The blend pass.

use crate::render::wgpu::passes::copy::encode_copy_pass;
use crate::render::wgpu::passes::create_texture_op_pipeline;
use crate::render::wgpu::passes::{Programs, RendererContext};
use crate::util::RangedSlice;
use crate::{
    blend::{BlendStrip, GpuBlendInstance},
    schedule::round::BlendOp,
    target::{BlendPassBindings, TextureParity},
};
use wgpu::{
    BindGroup, BindGroupLayout, Device, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, TextureView, TextureViewDescriptor, util::DeviceExt,
};

/// Bind group layout and pipeline of the blend pass.
#[derive(Debug)]
pub(crate) struct BlendPipeline {
    /// Blend bind group layout.
    pub(crate) bind_group_layout: BindGroupLayout,
    /// Blend pipeline.
    pub(crate) pipeline: RenderPipeline,
}

impl BlendPipeline {
    pub(crate) fn new(device: &Device, pipeline_cache: Option<&wgpu::PipelineCache>) -> Self {
        let blend_layer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blend Layer Textures Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        let blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blend Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::BLEND.into()),
        });
        let blend_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Blend Pipeline Layout"),
                bind_group_layouts: &[Some(&blend_layer_bind_group_layout)],
                immediate_size: 0,
            });
        let blend_vertex_state = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuBlendInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
                0 => Uint32,
                1 => Uint32,
                2 => Uint32,
                3 => Uint32,
                4 => Uint32,
                5 => Uint32,
                6 => Uint32,
                7 => Uint32,
            ],
        };
        let blend_pipeline = create_texture_op_pipeline(
            device,
            "Blend Pipeline",
            &blend_shader,
            &blend_pipeline_layout,
            &blend_vertex_state,
            pipeline_cache,
        );

        Self {
            bind_group_layout: blend_layer_bind_group_layout,
            pipeline: blend_pipeline,
        }
    }
}

impl Programs {
    fn create_blend_layer_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        layer_texture_views: [&TextureView; 2],
        alphas_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blend Layer Textures Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(layer_texture_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(layer_texture_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(alphas_texture_view),
                },
            ],
        })
    }
}

impl RendererContext<'_> {
    pub(crate) fn blend_pass_inner(
        &mut self,
        blends: RangedSlice<'_, BlendOp>,
        blend_strips: &[BlendStrip],
        bindings: BlendPassBindings,
    ) {
        let texture_size = self.texture_size();
        if !self
            .resources
            .blend_layer_bind_groups
            .contains_key(&bindings)
        {
            let alphas_texture_view = self
                .resources
                .alphas_texture
                .texture()
                .create_view(&TextureViewDescriptor::default());
            let bind_group = Programs::create_blend_layer_bind_group(
                self.device,
                &self.programs.blend.bind_group_layout,
                self.resources.layer_views([
                    bindings.layer_id(TextureParity::Even),
                    bindings.layer_id(TextureParity::Odd),
                ]),
                &alphas_texture_view,
            );
            self.resources
                .blend_layer_bind_groups
                .insert(bindings, bind_group);
        }
        let blend_layer_bind_group = &self.resources.blend_layer_bind_groups[&bindings];

        self.scratch_buffers.blend_instances.clear();

        for blend in blends.iter() {
            if let Some(range) = &blend.clip_strips {
                self.scratch_buffers.blend_instances.extend(
                    blend_strips[usize::try_from(range.start).unwrap()
                        ..usize::try_from(range.end).unwrap()]
                        .iter()
                        .copied()
                        .map(|strip| GpuBlendInstance::new(blend, Some(strip), texture_size)),
                );
            } else {
                self.scratch_buffers
                    .blend_instances
                    .push(GpuBlendInstance::new(blend, None, texture_size));
            }
        }

        let instance_count = u32::try_from(self.scratch_buffers.blend_instances.len()).unwrap();
        let instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Blend Instances Buffer"),
                contents: bytemuck::cast_slice(&self.scratch_buffers.blend_instances),
                usage: wgpu::BufferUsages::VERTEX,
            });

        {
            let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Blend To Scratch"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: self.resources.scratch_view(),
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
            render_pass.set_pipeline(&self.programs.blend.pipeline);
            render_pass.set_bind_group(0, blend_layer_bind_group, &[]);
            render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
            render_pass.draw(0..4, 0..instance_count);
        }

        self.scratch_buffers.copy_instances.clear();
        self.scratch_buffers.copy_instances.extend(
            self.scratch_buffers
                .blend_instances
                .iter()
                .copied()
                .map(GpuBlendInstance::copy_from_scratch),
        );
        encode_copy_pass(
            self.device,
            self.encoder,
            &self.programs.copy.pipeline,
            &self.scratch_buffers.copy_instances,
            &self.resources.scratch_copy_bind_group,
            self.resources.layer_view(bindings.target()),
            "Copy Blend Scratch To Layer",
        );
    }
}
