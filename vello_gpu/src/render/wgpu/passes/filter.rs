// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The filter pass.

use crate::render::wgpu::passes::copy::encode_copy_pass;
use crate::render::wgpu::passes::{Programs, RendererContext};
use crate::render::wgpu::resources::GpuResources;
use crate::{
    filter::{FilterContext, FilterInstanceData, FilterPassPlan},
    target::{FilterPassBindings, LayerTextureId, TextureParity},
};
use core::fmt::Debug;
use wgpu::{
    BindGroup, BindGroupLayout, CommandEncoder, Device, Extent3d, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, Sampler, Texture, TextureView, TextureViewDescriptor,
    util::DeviceExt,
};

#[derive(Debug)]
pub(crate) struct FilterPairBindGroups {
    inputs: [BindGroup; 2],
    copy_source: BindGroup,
}

impl Programs {
    pub(crate) fn create_filter_data_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Filter Data Texture"),
            size: Extent3d {
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

    pub(crate) fn create_filter_base_bind_group(
        device: &Device,
        filter_bind_group_layout: &BindGroupLayout,
        filter_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Filter Base Bind Group"),
            layout: filter_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(filter_texture_view),
            }],
        })
    }

    pub(crate) fn maybe_resize_filter_tex(
        &mut self,
        device: &Device,
        resource_texture_dimension_2d: u32,
        filter_context: &FilterContext,
    ) {
        let Some(required_filter_height) =
            filter_context.required_filter_data_height(resource_texture_dimension_2d)
        else {
            return;
        };
        debug_assert!(
            self.resources.filter_data_texture.width() == resource_texture_dimension_2d,
            "Filter texture width must match resource texture dimensions"
        );
        let current_filter_height = self.resources.filter_data_texture.height();
        if required_filter_height > current_filter_height {
            let required_filter_size =
                (resource_texture_dimension_2d * required_filter_height) << 4;
            self.filter_data.resize(required_filter_size as usize, 0);

            let filter_texture = Self::create_filter_data_texture(
                device,
                resource_texture_dimension_2d,
                required_filter_height,
            );
            self.resources.filter_data_texture = filter_texture;
            self.resources.filter_base_bind_group = Self::create_filter_base_bind_group(
                device,
                &self.filter_bind_group_layout,
                &self
                    .resources
                    .filter_data_texture
                    .create_view(&TextureViewDescriptor::default()),
            );
        }
    }

    pub(crate) fn upload_filter_texture(
        &mut self,
        encoder: &mut CommandEncoder,
        filter_context: &FilterContext,
    ) {
        if filter_context.is_empty() {
            return;
        }
        let texture = &self.resources.filter_data_texture;
        let rows = filter_context.total_texels().div_ceil(texture.width());
        let data = &mut self.filter_data[..(rows * texture.width() * 16) as usize];

        filter_context.serialize_to_buffer(data);

        self.staging_belt.upload_texture(encoder, texture, data, 16);
    }
}

impl RendererContext<'_> {
    pub(crate) fn filter_pass_inner(
        &mut self,
        plan: &FilterPassPlan,
        bindings: FilterPassBindings,
    ) {
        let resources = &self.programs.resources;
        if !self
            .programs
            .filter_pair_bind_groups
            .contains_key(&bindings)
        {
            let target = bindings.target();
            let temporary = bindings.temporary();
            let layer_ids = match target.texture_parity {
                TextureParity::Even => [target, temporary],
                TextureParity::Odd => [temporary, target],
            };
            let layer_views = resources.layer_views(layer_ids);
            let inputs = layer_views.map(|view| {
                create_filter_input_bind_group(
                    self.device,
                    &self.programs.filter_input_bind_group_layouts[0],
                    &self.programs.filter_sampler,
                    view,
                )
            });
            let copy_source = Programs::create_copy_bind_group(
                self.device,
                &self.programs.copy_bind_group_layout,
                resources.layer_view(target),
            );
            self.programs.filter_pair_bind_groups.insert(
                bindings,
                FilterPairBindGroups {
                    inputs,
                    copy_source,
                },
            );
        }
        let filter_pair_bind_groups = &self.programs.filter_pair_bind_groups[&bindings];

        if let Some(copy_pass) = plan.copy_pass() {
            encode_copy_pass(
                self.device,
                self.encoder,
                &self.programs.copy_pipeline,
                copy_pass,
                &filter_pair_bind_groups.copy_source,
                resources.scratch_view(),
                "Filter Copy Pass",
            );
        }

        for (step_index, instances) in plan.steps().enumerate() {
            let input = bindings.input(step_index);
            let output = bindings.output(step_index);
            encode_filter_pass(
                self.device,
                self.encoder,
                resources,
                &self.programs.filter_pipeline,
                instances,
                &filter_pair_bind_groups.inputs[input.texture_parity.get_parity()],
                &resources.filter_original_bind_group,
                output,
            );
        }
    }
}

pub(crate) fn encode_filter_pass(
    device: &Device,
    encoder: &mut CommandEncoder,
    resources: &GpuResources,
    filter_pipeline: &RenderPipeline,
    instances: &[FilterInstanceData],
    input_bind_group: &BindGroup,
    original_texture_bind_group: &BindGroup,
    output: LayerTextureId,
) {
    let output_view = resources.layer_view(output);
    let instance_count = u32::try_from(instances.len()).unwrap();
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Filter Instances Buffer"),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("Apply Filter Pass"),
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
    render_pass.set_pipeline(filter_pipeline);
    render_pass.set_bind_group(0, &resources.filter_base_bind_group, &[]);
    render_pass.set_bind_group(1, input_bind_group, &[]);
    render_pass.set_bind_group(2, original_texture_bind_group, &[]);
    render_pass.set_vertex_buffer(0, instance_buffer.slice(..));
    render_pass.draw(0..4, 0..instance_count);
}

pub(crate) fn create_filter_input_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    sampler: &Sampler,
    texture_view: &TextureView,
) -> BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Filter Input Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

pub(crate) fn create_filter_original_texture_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    layer_texture_view: &TextureView,
) -> BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Filter Original Texture Bind Group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(layer_texture_view),
        }],
    })
}
