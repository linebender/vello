// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, Device, Queue, TextureView,
};

use super::OutputSize;

#[derive(Debug)]
pub(crate) struct BlurPipeline {
    module: wgpu::ShaderModule,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl BlurPipeline {
    pub(crate) fn new(device: &Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pseudo Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_shaders::SHADERS.pseudo_blur.wgsl.code),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Pseudo Blur"),
            layout: None,
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        Self {
            module,
            pipeline,
            bind_group_layout,
        }
    }
    pub(crate) fn blur_into(
        &self,
        device: &Device,
        queue: &Queue,
        source: &TextureView,
        target: &TextureView,
        dimensions: OutputSize,
    ) {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Pseudo Blur pipeline"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Pseudo Blur Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("Pseudo Blur Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(source),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(target),
                    },
                ],
            });
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(
                dimensions.width.div_ceil(64),
                dimensions.height.div_ceil(4),
                1,
            );
        }
        // TODO: Don't submit after every item
        queue.submit([encoder.finish()]);
    }
}
