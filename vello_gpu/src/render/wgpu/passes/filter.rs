// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The filter pass.

use crate::RenderError;
use crate::render::wgpu::passes::copy::encode_copy_pass;
use crate::render::wgpu::passes::{Programs, RendererContext};
use crate::render::wgpu::resources::GpuResources;
use crate::{
    filter::{FilterContext, FilterInstanceData, FilterPassPlan},
    target::{FilterPassBindings, LayerTextureId, TextureParity},
};
use core::fmt::Debug;
use wgpu::{
    BindGroup, BindGroupLayout, ColorTargetState, ColorWrites, CommandEncoder, Device,
    PipelineCompilationOptions, RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline,
    Sampler, TextureView, util::DeviceExt,
};

/// Bind group layouts, sampler and pipeline of the filter pass.
#[derive(Debug)]
pub(crate) struct FilterPipeline {
    /// Bind group layout for filter data texture.
    pub(crate) bind_group_layout: BindGroupLayout,
    /// Filter input layouts.
    pub(crate) input_bind_group_layouts: [BindGroupLayout; 2],
    /// Filter sampler.
    pub(crate) sampler: Sampler,
    /// Pipeline for applying filter effects.
    pub(crate) pipeline: RenderPipeline,
}

impl FilterPipeline {
    pub(crate) fn new(device: &Device, pipeline_cache: Option<&wgpu::PipelineCache>) -> Self {
        let filter_texture_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let filter_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Filter Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let filter_input_bind_group_layouts = [
            // Input texture and linear sampler.
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Filter Input Bind Group Layout"),
                entries: &[
                    filter_texture_entry,
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            }),
            // The original layer texture.
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Filter Original Texture Bind Group Layout"),
                entries: &[filter_texture_entry],
            }),
        ];

        let filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_gpu_shaders::wgsl::FILTER.into()),
        });
        let filter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Filter Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&filter_bind_group_layout),
                    Some(&filter_input_bind_group_layouts[0]),
                    Some(&filter_input_bind_group_layouts[1]),
                ],
                immediate_size: 0,
            });
        let filter_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Filter Pipeline"),
            layout: Some(&filter_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &filter_shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<FilterInstanceData>() as u64,
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
                        8 => Uint32,
                    ],
                })],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &filter_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
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
            cache: pipeline_cache,
            multiview_mask: None,
        });
        let filter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Filter Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            bind_group_layout: filter_bind_group_layout,
            input_bind_group_layouts: filter_input_bind_group_layouts,
            sampler: filter_sampler,
            pipeline: filter_pipeline,
        }
    }
}

#[derive(Debug)]
pub(crate) struct FilterPairBindGroups {
    inputs: [BindGroup; 2],
    copy_source: BindGroup,
}

impl GpuResources {
    pub(crate) fn maybe_resize_filter_tex(
        &mut self,
        device: &Device,
        programs: &Programs,
        filter_context: &FilterContext,
    ) -> Result<(), RenderError> {
        let layout = &programs.filter.bind_group_layout;
        self.filter_data_texture.ensure_bytes(
            filter_context.as_bytes().len(),
            |format, width, height| {
                Ok::<_, RenderError>(Self::create_bound_data_texture(
                    device,
                    "Filter Data",
                    layout,
                    format,
                    width,
                    height,
                ))
            },
        )?;

        Ok(())
    }

    pub(crate) fn upload_filter_texture(
        &mut self,
        encoder: &mut CommandEncoder,
        filter_context: &FilterContext,
    ) {
        if filter_context.is_empty() {
            return;
        }
        self.filter_data_texture
            .upload(&mut self.staging_belt, encoder, filter_context.as_bytes());
    }
}

impl RendererContext<'_> {
    pub(crate) fn filter_pass_inner(
        &mut self,
        plan: &FilterPassPlan,
        bindings: FilterPassBindings,
    ) {
        if !self
            .resources
            .filter_pair_bind_groups
            .contains_key(&bindings)
        {
            let target = bindings.target();
            let temporary = bindings.temporary();
            let layer_ids = match target.texture_parity {
                TextureParity::Even => [target, temporary],
                TextureParity::Odd => [temporary, target],
            };
            let layer_views = self.resources.layer_views(layer_ids);
            let inputs = layer_views.map(|view| {
                create_filter_input_bind_group(
                    self.device,
                    &self.programs.filter.input_bind_group_layouts[0],
                    &self.programs.filter.sampler,
                    view,
                )
            });
            let copy_source = Programs::create_copy_bind_group(
                self.device,
                &self.programs.copy.bind_group_layout,
                self.resources.layer_view(target),
            );
            self.resources.filter_pair_bind_groups.insert(
                bindings,
                FilterPairBindGroups {
                    inputs,
                    copy_source,
                },
            );
        }
        let resources = &*self.resources;
        let filter_pair_bind_groups = &resources.filter_pair_bind_groups[&bindings];

        if let Some(copy_pass) = plan.copy_pass() {
            encode_copy_pass(
                self.device,
                self.encoder,
                &self.programs.copy.pipeline,
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
                &self.programs.filter.pipeline,
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
    render_pass.set_bind_group(0, &resources.filter_data_texture.texture().bind_group, &[]);
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
