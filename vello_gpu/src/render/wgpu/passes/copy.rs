// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The copy pass.

use crate::copy::GpuCopyInstance;
use crate::render::wgpu::passes::Programs;
use crate::render::wgpu::passes::create_texture_op_pipeline;
use wgpu::{
    BindGroup, BindGroupLayout, CommandEncoder, Device, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, TextureView, util::DeviceExt,
};

/// Bind group layout and pipeline of the copy pass.
#[derive(Debug)]
pub(crate) struct CopyPipeline {
    /// Copy bind group layout.
    pub(crate) bind_group_layout: BindGroupLayout,
    /// Copy pipeline.
    pub(crate) pipeline: RenderPipeline,
}

impl CopyPipeline {
    pub(crate) fn new(device: &Device, pipeline_cache: Option<&wgpu::PipelineCache>) -> Self {
        let copy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Copy Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::COPY.into()),
        });
        let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Copy Pipeline Layout"),
            bind_group_layouts: &[Some(&copy_bind_group_layout)],
            immediate_size: 0,
        });
        let copy_vertex_state = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuCopyInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
                0 => Uint32,
                1 => Uint32,
                2 => Uint32,
                3 => Uint32,
            ],
        };
        let copy_pipeline = create_texture_op_pipeline(
            device,
            "Copy Pipeline",
            &copy_shader,
            &copy_pipeline_layout,
            &copy_vertex_state,
            pipeline_cache,
        );

        Self {
            bind_group_layout: copy_bind_group_layout,
            pipeline: copy_pipeline,
        }
    }
}

impl Programs {
    pub(crate) fn create_copy_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        source_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Copy Source Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source_texture_view),
            }],
        })
    }
}

pub(crate) fn encode_copy_pass(
    device: &Device,
    encoder: &mut CommandEncoder,
    copy_pipeline: &RenderPipeline,
    instances: &[GpuCopyInstance],
    source_bind_group: &BindGroup,
    output_view: &TextureView,
    label: &'static str,
) {
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: output_view,
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
    render_pass.set_pipeline(copy_pipeline);
    render_pass.set_bind_group(0, source_bind_group, &[]);
    render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
    render_pass.draw(0..4, 0..u32::try_from(instances.len()).unwrap());
}
