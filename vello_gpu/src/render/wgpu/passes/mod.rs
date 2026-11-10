// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pipeline creation and execution of the scheduled render passes.

pub(crate) mod blend;
pub(crate) mod clear;
pub(crate) mod copy;
pub(crate) mod filter;
pub(crate) mod strip;

use crate::draw::{ExternalTextureBindings, ExternalTextureRun};
use crate::render::wgpu::RenderTargetConfig;
use crate::render::wgpu::bindings::TextureBindings;
use crate::render::wgpu::passes::blend::BlendPipeline;
use crate::render::wgpu::passes::clear::{ClearPipelines, clear_color};
use crate::render::wgpu::passes::copy::CopyPipeline;
use crate::render::wgpu::passes::filter::FilterPipeline;
use crate::render::wgpu::passes::strip::StripPipelines;
use crate::render::wgpu::resources::GpuResources;
use crate::util::RangedSlice;
use crate::{
    ClearSettings, GpuStrip, TargetInit,
    blend::BlendStrip,
    filter::FilterPassPlan,
    render::common::ScratchBuffers,
    schedule::{Backend, round::BlendOp},
    target::{
        BlendPassBindings, DrawPassBindings, DrawPassTarget, FilterPassBindings, LayerTextureId,
        RootTarget,
    },
};
use core::fmt::Debug;
use hashbrown::HashMap;
use vello_common::color::AlphaColor;
use vello_common::geometry::{RectU16, SizeU16};
use wgpu::{
    BindGroup, BindGroupLayout, ColorTargetState, ColorWrites, CommandEncoder, Device,
    PipelineCompilationOptions, RenderPipeline, TextureView,
};

/// The pipelines and bind group layouts of the render passes.
#[derive(Debug)]
pub(crate) struct Programs {
    /// Bind group layouts and pipelines of the strip pass.
    pub(crate) strip: StripPipelines,
    /// Clear pipelines.
    pub(crate) clear: ClearPipelines,
    /// Bind group layouts, sampler and pipeline of the filter pass.
    pub(crate) filter: FilterPipeline,
    /// Bind group layout and pipeline of the blend pass.
    pub(crate) blend: BlendPipeline,
    /// Bind group layout and pipeline of the copy pass.
    pub(crate) copy: CopyPipeline,
    /// Bind group layout for encoded paints
    pub(crate) encoded_paints_bind_group_layout: BindGroupLayout,
    /// Bind group layout for gradient texture
    pub(crate) gradient_bind_group_layout: BindGroupLayout,
}

impl Programs {
    pub(crate) fn new(
        device: &Device,
        render_target_config: &RenderTargetConfig,
        pipeline_cache: Option<&wgpu::PipelineCache>,
    ) -> Self {
        let encoded_paints_bind_group_layout =
            Self::create_encoded_paints_bind_group_layout(device);
        let gradient_bind_group_layout = Self::create_gradient_bind_group_layout(device);
        let strip = StripPipelines::new(
            device,
            render_target_config,
            &encoded_paints_bind_group_layout,
            &gradient_bind_group_layout,
            pipeline_cache,
        );

        Self {
            strip,
            clear: ClearPipelines::new(device, render_target_config, pipeline_cache),
            filter: FilterPipeline::new(device, pipeline_cache),
            blend: BlendPipeline::new(device, pipeline_cache),
            copy: CopyPipeline::new(device, pipeline_cache),
            encoded_paints_bind_group_layout,
            gradient_bind_group_layout,
        }
    }

    pub(crate) fn create_encoded_paints_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Encoded Paints Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        })
    }

    pub(crate) fn create_gradient_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gradient Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        })
    }
}

/// Create a pipeline that draws instanced quads into an intermediate texture without blending.
fn create_texture_op_pipeline(
    device: &Device,
    label: &str,
    shader_module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    vertex_state: &wgpu::VertexBufferLayout<'_>,
    pipeline_cache: Option<&wgpu::PipelineCache>,
) -> RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader_module,
            entry_point: Some("vs_main"),
            buffers: &[Some(vertex_state.clone())],
            compilation_options: PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader_module,
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
        multiview_mask: None,
        cache: pipeline_cache,
    })
}

/// A struct containing references to the many objects needed to get work
/// scheduled onto the GPU.
pub(crate) struct RendererContext<'a> {
    pub(crate) programs: &'a Programs,
    pub(crate) resources: &'a mut GpuResources,
    pub(crate) device: &'a Device,
    pub(crate) encoder: &'a mut CommandEncoder,
    pub(crate) view: &'a TextureView,
    pub(crate) depth_view: Option<&'a TextureView>,
    pub(crate) texture_bindings: &'a TextureBindings,
    pub(crate) external_texture_bind_groups: HashMap<ExternalTextureBindings, BindGroup>,
    pub(crate) scratch_buffers: &'a mut ScratchBuffers,
    pub(crate) root_load_op: wgpu::LoadOp<wgpu::Color>,
}

impl RendererContext<'_> {
    pub(crate) fn init_root_clear(&mut self, target_init: TargetInit<'_>, root_target: RootTarget) {
        self.root_load_op = match target_init {
            TargetInit::SrcOver => wgpu::LoadOp::Load,
            TargetInit::Clear(ClearSettings::Viewport { color }) => {
                wgpu::LoadOp::Clear(clear_color(color))
            }
            TargetInit::Clear(clear @ ClearSettings::Rects { .. }) => {
                self.clear_pass_inner(DrawPassTarget::Root(root_target), clear);

                wgpu::LoadOp::Load
            }
        };
    }

    pub(crate) fn finish_root_clear(&mut self) {
        // If the load mode was set to clear but no actual drawing instructions were
        // emitted, we still need to start a pass to clear the output.
        if let wgpu::LoadOp::Clear(color) = self.take_root_load_op() {
            Self::clear_full_target(self.encoder, self.view, color);
        }
    }

    fn take_root_load_op(&mut self) -> wgpu::LoadOp<wgpu::Color> {
        core::mem::replace(&mut self.root_load_op, wgpu::LoadOp::Load)
    }

    fn texture_size(&self) -> SizeU16 {
        self.resources.texture_size
    }
}

impl Backend for RendererContext<'_> {
    type Error = core::convert::Infallible;

    fn opaque_draw_pass(
        &mut self,
        strips: &[GpuStrip],
        external_texture_runs: &[ExternalTextureRun],
    ) -> Result<(), Self::Error> {
        self.strip_pass_inner(
            strips,
            RangedSlice::empty(),
            external_texture_runs,
            DrawPassTarget::Root(RootTarget::UserSurface),
            None,
        );

        Ok(())
    }

    fn draw_pass(
        &mut self,
        strips: RangedSlice<'_, GpuStrip>,
        external_texture_runs: &[ExternalTextureRun],
        bindings: DrawPassBindings,
    ) -> Result<(), Self::Error> {
        self.strip_pass_inner(
            &[],
            strips,
            external_texture_runs,
            bindings.target,
            bindings.child,
        );

        Ok(())
    }

    fn blend_pass(
        &mut self,
        blends: RangedSlice<'_, BlendOp>,
        blend_strips: &[BlendStrip],
        bindings: BlendPassBindings,
    ) -> Result<(), Self::Error> {
        self.blend_pass_inner(blends, blend_strips, bindings);

        Ok(())
    }

    fn filter_pass(
        &mut self,
        plan: &FilterPassPlan,
        bindings: FilterPassBindings,
    ) -> Result<(), Self::Error> {
        self.filter_pass_inner(plan, bindings);

        Ok(())
    }

    fn clear_pass(&mut self, target: LayerTextureId, rects: &[RectU16]) -> Result<(), Self::Error> {
        self.clear_pass_inner(
            DrawPassTarget::Layer(target),
            ClearSettings::Rects {
                color: AlphaColor::TRANSPARENT,
                rects,
            },
        );

        Ok(())
    }
}
