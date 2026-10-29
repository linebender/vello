// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pipeline creation and execution of the scheduled render passes.

pub(crate) mod blend;
pub(crate) mod clear;
pub(crate) mod copy;
pub(crate) mod filter;
pub(crate) mod strip;

use crate::draw::{EXTERNAL_TEXTURE_SLOT_COUNT, ExternalTextureBindings, ExternalTextureRun};
use crate::render::wgpu::RenderTargetConfig;
use crate::render::wgpu::bindings::TextureBindings;
use crate::render::wgpu::passes::blend::BlendPipeline;
use crate::render::wgpu::passes::clear::{ClearPipelines, clear_color};
use crate::render::wgpu::passes::copy::CopyPipeline;
use crate::render::wgpu::passes::filter::{
    FilterPairBindGroups, FilterPipeline, create_filter_original_texture_bind_group,
};
use crate::render::wgpu::passes::strip::{StripPipelines, StripTargetKind};
use crate::render::wgpu::resources::{GpuResources, StripBufferArena, UploadStagingBelt};
use crate::util::RangedSlice;
use crate::{
    ClearSettings, GpuStrip, LayersConfig, RenderSize, TargetInit,
    blend::BlendStrip,
    filter::FilterPassPlan,
    paint::ExternalSampler,
    render::common::ScratchBuffers,
    schedule::{Backend, round::BlendOp},
    target::{
        BlendPassBindings, DrawPassBindings, DrawPassTarget, FilterPassBindings, LayerTextureId,
        RootTarget,
    },
};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use hashbrown::HashMap;
use vello_common::color::AlphaColor;
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::AtlasConfig;
use wgpu::{
    BindGroup, BindGroupLayout, ColorTargetState, ColorWrites, CommandEncoder, Device,
    PipelineCompilationOptions, RenderPipeline, Sampler, TextureView, TextureViewDescriptor,
};

/// Defines the GPU resources and pipelines for rendering.
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
    /// Whether the depth buffer has been cleared this frame.
    pub(crate) depth_cleared_this_frame: bool,
    /// Bind group layout for encoded paints
    pub(crate) encoded_paints_bind_group_layout: BindGroupLayout,
    /// Bind group layout for gradient texture
    pub(crate) gradient_bind_group_layout: BindGroupLayout,
    /// Native image samplers shared by external texture slots.
    external_samplers: HashMap<ExternalSampler, Sampler>,
    /// Strip bind group cache.
    pub(crate) strip_layer_bind_groups:
        HashMap<(StripTargetKind, Option<LayerTextureId>), BindGroup>,
    /// Blend bind group cache.
    pub(crate) blend_layer_bind_groups: HashMap<BlendPassBindings, BindGroup>,
    /// Filter bind group cache.
    pub(crate) filter_pair_bind_groups: HashMap<FilterPassBindings, FilterPairBindGroups>,
    /// GPU resources for rendering (created during prepare)
    pub(crate) resources: GpuResources,
    /// Arena holding all [`GpuStrip`] data.
    pub(crate) strips_arena: StripBufferArena,
    /// Shared staging memory for buffer and texture uploads.
    pub(crate) staging_belt: UploadStagingBelt,
    /// Dimensions of the rendering target
    pub(crate) render_size: RenderSize,
    /// Offset of the scene within the rendering target.
    pub(crate) scene_offset: (i32, i32),
    /// Scratch buffer for staging encoded paints texture data.
    pub(crate) encoded_paints_data: Vec<u8>,
    /// Scratch buffer for staging filter data texture data.
    filter_data: Vec<u8>,
}

impl Programs {
    pub(crate) fn new(
        device: &Device,
        image_cache: &ImageCache,
        render_target_config: &RenderTargetConfig,
        layer_config: LayersConfig,
        resource_texture_dimension_2d: u32,
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
        let clear = ClearPipelines::new(device, render_target_config, pipeline_cache);
        let filter = FilterPipeline::new(device, pipeline_cache);
        let blend = BlendPipeline::new(device, pipeline_cache);
        let copy = CopyPipeline::new(device, pipeline_cache);

        let texture_size = layer_config.min_texture_size;

        // The empty bind group needs a sampler even before any external image is drawn.
        let mut external_samplers = HashMap::new();
        external_samplers.insert(
            ExternalSampler::DEFAULT,
            Self::create_external_sampler(device, ExternalSampler::DEFAULT),
        );

        let layer_textures: [Vec<TextureView>; 2] = core::array::from_fn(|_| Vec::new());
        let scratch_texture = None;
        let placeholder_external_texture_view = Self::create_placeholder_external_texture(device);
        let filter_original_bind_group = create_filter_original_texture_bind_group(
            device,
            &filter.input_bind_group_layouts[1],
            &placeholder_external_texture_view,
        );
        let layer_config_buffer = Self::create_config_buffer(
            device,
            u32::from(texture_size.width()),
            u32::from(texture_size.height()),
            resource_texture_dimension_2d,
        );

        const INITIAL_ALPHA_TEXTURE_HEIGHT: u32 = 1;
        let alphas_texture = Self::create_alphas_texture(
            device,
            resource_texture_dimension_2d,
            INITIAL_ALPHA_TEXTURE_HEIGHT,
        );
        let view_config_buffer = Self::create_config_buffer(
            device,
            u32::from(render_target_config.width),
            u32::from(render_target_config.height),
            resource_texture_dimension_2d,
        );

        let AtlasConfig {
            atlas_size: (atlas_width, atlas_height),
            initial_atlas_count,
            ..
        } = image_cache.atlas_manager().config();
        let atlas_size = (*atlas_width, *atlas_height);
        let atlas_textures: Vec<_> = (0..*initial_atlas_count)
            .map(|_| Self::create_atlas_texture(device, *atlas_width, *atlas_height))
            .collect();
        let atlas_texture_views = atlas_textures
            .iter()
            .map(|texture| texture.create_view(&TextureViewDescriptor::default()))
            .collect();
        let empty_external_texture_bind_group = Self::create_external_texture_bind_group(
            device,
            &strip.external_texture_bind_group_layout,
            [&placeholder_external_texture_view; EXTERNAL_TEXTURE_SLOT_COUNT],
            [&external_samplers[&ExternalSampler::DEFAULT]; EXTERNAL_TEXTURE_SLOT_COUNT],
            strip
                .wgpu_external_textures
                .then_some(wgpu::BindingResource::TextureView(
                    &placeholder_external_texture_view,
                )),
        );

        const INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT: u32 = 1;
        let encoded_paints_data = vec![
            0;
            ((resource_texture_dimension_2d * INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT) << 4)
                as usize
        ];
        let encoded_paints_texture = Self::create_encoded_paints_texture(
            device,
            resource_texture_dimension_2d,
            INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT,
        );
        let encoded_paints_bind_group = Self::create_encoded_paints_bind_group(
            device,
            &encoded_paints_bind_group_layout,
            &encoded_paints_texture.create_view(&TextureViewDescriptor::default()),
        );

        const INITIAL_GRADIENT_TEXTURE_HEIGHT: u32 = 1;
        let gradient_texture = Self::create_gradient_texture(
            device,
            resource_texture_dimension_2d,
            INITIAL_GRADIENT_TEXTURE_HEIGHT,
        );
        let gradient_bind_group = Self::create_gradient_bind_group(
            device,
            &gradient_bind_group_layout,
            &gradient_texture.create_view(&TextureViewDescriptor::default()),
        );

        // TODO: We really should deduplicate handling of this this with encoded paints texture.
        const INITIAL_FILTER_TEXTURE_HEIGHT: u32 = 1;
        let filter_data = vec![
            0_u8;
            ((resource_texture_dimension_2d * INITIAL_FILTER_TEXTURE_HEIGHT) << 4)
                as usize
        ];
        let filter_data_texture = Self::create_filter_data_texture(
            device,
            resource_texture_dimension_2d,
            INITIAL_FILTER_TEXTURE_HEIGHT,
        );
        let filter_base_bind_group = Self::create_filter_base_bind_group(
            device,
            &filter.bind_group_layout,
            &filter_data_texture.create_view(&TextureViewDescriptor::default()),
        );
        let scratch_copy_bind_group = Self::create_copy_bind_group(
            device,
            &copy.bind_group_layout,
            &placeholder_external_texture_view,
        );

        let resources = GpuResources {
            layer_textures,
            scratch_texture,
            filter_original_bind_group,
            scratch_copy_bind_group,
            layer_config_buffer,
            alphas_texture,
            resource_texture_dimension_2d,
            atlas_textures,
            atlas_texture_views,
            atlas_size,
            placeholder_external_texture_view,
            empty_external_texture_bind_group,
            encoded_paints_texture,
            encoded_paints_bind_group,
            gradient_texture,
            gradient_bind_group,
            filter_data_texture,
            filter_base_bind_group,
            texture_size,
            view_config_buffer,
        };

        Self {
            strip,
            clear,
            filter,
            blend,
            copy,
            depth_cleared_this_frame: false,
            encoded_paints_bind_group_layout,
            gradient_bind_group_layout,
            resources,
            strips_arena: StripBufferArena::new(device),
            staging_belt: UploadStagingBelt::new(device),
            encoded_paints_data,
            filter_data,
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            scene_offset: (0, 0),
            external_samplers,
            strip_layer_bind_groups: HashMap::new(),
            blend_layer_bind_groups: HashMap::new(),
            filter_pair_bind_groups: HashMap::new(),
        }
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
    pub(crate) programs: &'a mut Programs,
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
        self.programs.resources.texture_size
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
