// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU rendering module for the sparse strips CPU/GPU rendering engine.
//!
//! This module provides the GPU-side implementation of the hybrid rendering system.
//! It handles:
//! - GPU resource management (buffers, textures, pipelines)
//! - Surface/window management and presentation
//! - Shader execution and rendering
//!
//! The hybrid approach combines CPU-side path processing with efficient GPU rendering
//! to balance flexibility and performance.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

use crate::render::common::IMAGE_PADDING;
use crate::util::{IntRect, IntSize};
use crate::{
    GpuStrip, RenderError, RenderSettings, RenderSize, Resources,
    direct::{DirectStrips, DirectTarget, ExternalTextureRun, RenderOrigin},
    filter::{FilterContext, FilterInstanceData, FilterPassState, FilterPassTarget, GpuFilterData},
    gradient_cache::GradientRampCache,
    render::{
        Config,
        common::{
            GPU_BLURRED_ROUNDED_RECT_SIZE_TEXELS, GPU_ENCODED_IMAGE_SIZE_TEXELS,
            GPU_LINEAR_GRADIENT_SIZE_TEXELS, GPU_RADIAL_GRADIENT_SIZE_TEXELS,
            GPU_SWEEP_GRADIENT_SIZE_TEXELS, GpuBlurredRoundedRect, GpuEncodedImage,
            GpuEncodedPaint, GpuLinearGradient, GpuRadialGradient, GpuSweepGradient,
            normalize_atlas_config, pack_image_offset, pack_image_params, pack_image_size,
            pack_radial_kind_and_swapped, pack_texture_width_and_extend_mode, pack_tint,
        },
        layer_filter::{FilterPassSource, LayerFilterPlan, filter_pass_target},
        scheduler::{
            LayerSampleExtent, LayerScheduleRenderer, LayerScheduler, ROOT_LAYER_IDX,
            ScheduledLayerOp, ScheduledLayerTarget, aligned_layer_source_texture_id,
            backdrop_texture_id, filter_scratch_texture_id, flatten_draw_batches,
            layer_blend_scissor, layer_can_be_sampled_directly, layer_encoded_paint,
            layer_sample_command_with_paint_idx, layer_sample_encoded_paint,
            layer_target_origin_and_size, layer_texture_id, root_needs_offscreen_layer,
            root_sample_command_with_paint_idx, root_texture_id,
        },
    },
    scene::{FastStripCommand, RecordedCommand, RecordedLayer, RecordedLayerId, Scene},
};
use alloc::vec::Vec;
use alloc::{sync::Arc, vec};
use bytemuck::{Pod, Zeroable};
use core::fmt::Debug;
#[cfg(feature = "text")]
use glifo::PendingClearRect;
use hashbrown::{HashMap, hash_map::Entry};
use vello_common::image_cache::{ImageCache, ImageResource};
use vello_common::multi_atlas::{AtlasConfig, AtlasId};
use vello_common::render_graph::LayerId;
use vello_common::{
    TextureId,
    encode::{
        EncodedBlurredRoundedRectangle, EncodedExternalTexture, EncodedGradient, EncodedKind,
        EncodedPaint, MAX_GRADIENT_LUT_SIZE, RadialKind,
    },
    geometry::RectU16,
    paint::ImageSource,
    peniko,
    pixmap::Pixmap,
    tile::Tile,
};
use wgpu::{
    BindGroup, BindGroupLayout, BlendState, Buffer, ColorTargetState, ColorWrites, CommandEncoder,
    Device, Extent3d, PipelineCompilationOptions, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, Sampler, Texture, TextureView, TextureViewDescriptor,
    util::DeviceExt,
};

/// Placeholder value for uninitialized GPU encoded paints.
const GPU_PAINT_PLACEHOLDER: GpuEncodedPaint = GpuEncodedPaint::LinearGradient(GpuLinearGradient {
    texture_width_and_extend_mode: 0,
    gradient_start: 0,
    transform: [0.0; 6],
});

const EXTERNAL_IMAGE_SOURCE_FLAG: u32 = 1 << 14;

/// Options for the renderer
#[derive(Debug)]
pub struct RenderTargetConfig {
    /// Format of the rendering target
    pub format: wgpu::TextureFormat,
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
}

/// Runtime bindings for [externally owned textures](`TextureId`) sampled by texture-rect draws.
#[derive(Debug, Default, Clone)]
pub struct TextureBindings {
    views: HashMap<TextureId, TextureView>,
}

impl TextureBindings {
    /// Create an empty binding map.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a texture binding.
    ///
    /// The [`TextureView`] must fit the following binding type.
    ///
    /// ```ignore
    /// wgpu::BindGroupLayoutEntry {
    ///     binding: 1,
    ///     visibility: wgpu::ShaderStages::FRAGMENT,
    ///     ty: wgpu::BindingType::Texture {
    ///         sample_type: wgpu::TextureSampleType::Float { filterable: false },
    ///         view_dimension: wgpu::TextureViewDimension::D2,
    ///         multisampled: false,
    ///     },
    ///     count: None,
    /// }
    /// ```
    ///
    /// This means the view must be a non-array 2D view of a float-sampleable texture (e.g. integer
    /// formats are rejected by wgpu at bind time), the underlying texture must include
    /// [`wgpu::TextureUsages::TEXTURE_BINDING`], and only mip level 0 is read.
    #[inline]
    pub fn insert(&mut self, texture_id: TextureId, view: TextureView) {
        self.views.insert(texture_id, view);
    }

    /// Get a texture binding.
    #[inline]
    fn get(&self, texture_id: TextureId) -> Option<&TextureView> {
        self.views.get(&texture_id)
    }

    /// Remove a texture binding.
    ///
    /// This returns the removed [`TextureView`] binding if it existed.
    #[inline]
    pub fn remove(&mut self, texture_id: TextureId) -> Option<TextureView> {
        self.views.remove(&texture_id)
    }
}

#[derive(Clone, Debug)]
struct LayerTarget {
    texture_id: TextureId,
    texture: Texture,
    view: TextureView,
    filter_input_bind_group: BindGroup,
    filter_original_bind_group: BindGroup,
    render_size: RenderSize,
    origin: RenderOrigin,
}

#[derive(Debug, Default)]
struct LayerTargetPool {
    targets: Vec<LayerTarget>,
}

impl LayerTargetPool {
    fn acquire(
        &mut self,
        device: &Device,
        programs: &Programs,
        texture_id: TextureId,
        render_size: RenderSize,
        origin: RenderOrigin,
    ) -> LayerTarget {
        if let Some(index) = self
            .targets
            .iter()
            .position(|target| target.render_size == render_size)
        {
            let mut target = self.targets.swap_remove(index);
            target.texture_id = texture_id;
            target.origin = origin;
            return target;
        }

        LayerTarget::new(device, programs, texture_id, render_size, origin)
    }

    fn release(&mut self, target: LayerTarget) {
        self.targets.push(target);
    }
}

impl LayerTarget {
    fn new(
        device: &Device,
        programs: &Programs,
        texture_id: TextureId,
        render_size: RenderSize,
        origin: RenderOrigin,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Opacity Layer Texture"),
            size: Extent3d {
                width: render_size.width.max(1),
                height: render_size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let filter_input_bind_group = create_filter_input_bind_group(
            device,
            &programs.filter_input_bind_group_layouts[0],
            &programs.resources.filter_atlas.sampler,
            &view,
        );
        let filter_original_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Layer Filter Original Bind Group"),
            layout: &programs.filter_input_bind_group_layouts[1],
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        Self {
            texture_id,
            texture,
            view,
            filter_input_bind_group,
            filter_original_bind_group,
            render_size,
            origin,
        }
    }
}

impl ScheduledLayerTarget for LayerTarget {
    #[inline]
    fn texture_id(&self) -> TextureId {
        self.texture_id
    }

    #[inline]
    fn width(&self) -> u16 {
        self.render_size.width.try_into().unwrap_or(u16::MAX)
    }

    #[inline]
    fn height(&self) -> u16 {
        self.render_size.height.try_into().unwrap_or(u16::MAX)
    }
}

#[derive(Debug)]
struct ScheduledPaintIndices {
    root: Option<usize>,
    output: Vec<Option<usize>>,
    content: Vec<Option<usize>>,
}

impl ScheduledPaintIndices {
    fn new(layer_count: usize) -> Self {
        Self {
            root: None,
            output: vec![None; layer_count],
            content: vec![None; layer_count],
        }
    }
}

fn root_needs_scheduled_offscreen(scene: &Scene, visible_layers: &[bool]) -> bool {
    scene.root(scene.root_id()).commands.iter().any(|command| {
        let RecordedCommand::Layer(layer_id) = command else {
            return false;
        };
        visible_layers[layer_id.as_usize()]
    })
}

struct WgpuLayerSchedule<'a, 'b> {
    renderer: &'a mut Renderer,
    scene: &'b Scene,
    device: &'b Device,
    queue: &'b Queue,
    encoder: &'a mut CommandEncoder,
    root_render_size: &'b RenderSize,
    root_target: DirectTarget,
    root_clear: bool,
    layer_render_size: RenderSize,
    root_view: &'b TextureView,
    image_cache: &'b ImageCache,
    texture_bindings: &'b TextureBindings,
    planned_layer_targets: Vec<Option<LayerTarget>>,
    planned_root_target: Option<LayerTarget>,
    paint_indices: ScheduledPaintIndices,
    extra_layer_targets: Vec<LayerTarget>,
    layer_filter_instances: Vec<FilterInstanceData>,
    shared_resources_uploaded: bool,
    layer_filter_data_offset: u32,
}

impl LayerScheduleRenderer<LayerTarget> for WgpuLayerSchedule<'_, '_> {
    fn create_layer_target(&mut self, layer_idx: usize, texture_id: TextureId) -> LayerTarget {
        if layer_idx == ROOT_LAYER_IDX {
            if let Some(target) = &self.planned_root_target {
                return target.clone();
            }
        } else if let Some(Some(target)) = self.planned_layer_targets.get(layer_idx) {
            return target.clone();
        }

        let (origin, render_size) =
            layer_target_origin_and_size(self.scene, layer_idx, &self.layer_render_size);
        let target =
            self.renderer
                .create_layer_target(self.device, texture_id, render_size, origin);
        self.extra_layer_targets.push(target.clone());
        target
    }

    fn render_layer(
        &mut self,
        layer_idx: usize,
        layer: &RecordedLayer,
        target: LayerTarget,
        batches: &[ScheduledLayerOp],
        encoded_paints: &mut Vec<EncodedPaint>,
        rendered_targets: &[Option<LayerTarget>],
    ) -> Result<LayerTarget, RenderError> {
        self.render_batches_to_layer(
            layer_idx,
            &target,
            batches,
            encoded_paints,
            rendered_targets,
        )?;
        if layer.filter.is_some() {
            return self.apply_filter_to_layer(layer_idx, layer, target);
        }
        Ok(target)
    }

    fn render_root(
        &mut self,
        batches: &[ScheduledLayerOp],
        encoded_paints: &mut Vec<EncodedPaint>,
        rendered_targets: &[Option<LayerTarget>],
    ) -> Result<(), RenderError> {
        if root_needs_offscreen_layer(batches) {
            let root_layer_target = self.create_layer_target(ROOT_LAYER_IDX, root_texture_id());
            self.render_batches_to_layer(
                ROOT_LAYER_IDX,
                &root_layer_target,
                batches,
                encoded_paints,
                rendered_targets,
            )?;

            let commands = [root_sample_command_with_paint_idx(
                self.root_render_size,
                self.paint_indices
                    .root
                    .expect("root sample paint must be preplanned"),
            )];

            let mut root_texture_bindings =
                self.texture_bindings_with_scheduled_layers(rendered_targets);
            root_texture_bindings
                .insert(root_layer_target.texture_id, root_layer_target.view.clone());
            return self.render_scene(
                &commands,
                self.root_render_size,
                self.root_view,
                encoded_paints,
                self.root_clear,
                self.root_target,
                RenderOrigin::default(),
                &root_texture_bindings,
            );
        }

        let commands = flatten_draw_batches(batches);
        let root_texture_bindings = self.texture_bindings_with_scheduled_layers(rendered_targets);
        self.render_scene(
            commands.as_slice(),
            self.root_render_size,
            self.root_view,
            encoded_paints,
            self.root_clear,
            self.root_target,
            RenderOrigin::default(),
            &root_texture_bindings,
        )
    }
}

impl WgpuLayerSchedule<'_, '_> {
    fn create_extra_layer_target(
        &mut self,
        texture_id: TextureId,
        render_size: RenderSize,
        origin: RenderOrigin,
    ) -> LayerTarget {
        let target =
            self.renderer
                .create_layer_target(self.device, texture_id, render_size, origin);
        self.extra_layer_targets.push(target.clone());
        target
    }

    fn finish(&mut self) {
        if !self.layer_filter_instances.is_empty() {
            self.queue.write_buffer(
                &self.renderer.programs.resources.filter_instance_buffer,
                0,
                bytemuck::cast_slice(&self.layer_filter_instances),
            );
        }

        for target in self
            .planned_layer_targets
            .iter_mut()
            .filter_map(Option::take)
        {
            self.renderer.release_layer_target(target);
        }
        if let Some(target) = self.planned_root_target.take() {
            self.renderer.release_layer_target(target);
        }
        for target in self.extra_layer_targets.drain(..) {
            self.renderer.release_layer_target(target);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_scene(
        &mut self,
        commands: &[FastStripCommand],
        render_size: &RenderSize,
        view: &TextureView,
        encoded_paints: &[EncodedPaint],
        clear: bool,
        target: DirectTarget,
        origin: RenderOrigin,
        texture_bindings: &TextureBindings,
    ) -> Result<(), RenderError> {
        let upload_shared_resources = !self.shared_resources_uploaded;
        let result = self.renderer.render_scene(
            self.scene,
            commands,
            self.device,
            self.queue,
            self.encoder,
            render_size,
            view,
            self.image_cache,
            encoded_paints,
            clear,
            target,
            origin,
            texture_bindings,
            upload_shared_resources,
        );
        if result.is_ok() && upload_shared_resources {
            self.shared_resources_uploaded = true;
        }
        result
    }

    fn texture_bindings_with_scheduled_layers(
        &self,
        rendered_targets: &[Option<LayerTarget>],
    ) -> TextureBindings {
        let mut bindings = self.texture_bindings.clone();
        for target in self.planned_layer_targets.iter().flatten() {
            bindings.insert(target.texture_id, target.view.clone());
        }
        if let Some(target) = &self.planned_root_target {
            bindings.insert(target.texture_id, target.view.clone());
        }
        for target in rendered_targets.iter().flatten() {
            bindings.insert(target.texture_id, target.view.clone());
        }
        bindings
    }

    fn render_batches_to_layer(
        &mut self,
        parent_idx: usize,
        target: &LayerTarget,
        batches: &[ScheduledLayerOp],
        encoded_paints: &mut Vec<EncodedPaint>,
        rendered_targets: &[Option<LayerTarget>],
    ) -> Result<(), RenderError> {
        if !batches
            .iter()
            .any(|batch| matches!(batch, ScheduledLayerOp::CompositeLayer(_)))
        {
            let commands = flatten_draw_batches(batches);
            let layer_texture_bindings =
                self.texture_bindings_with_scheduled_layers(rendered_targets);
            return self.render_scene(
                commands.as_slice(),
                &target.render_size,
                &target.view,
                encoded_paints,
                true,
                DirectTarget::AtlasLayer,
                target.origin,
                &layer_texture_bindings,
            );
        }

        let mut clear_next_draw = true;
        for batch in batches {
            match batch {
                ScheduledLayerOp::Draw(commands) => {
                    if commands.is_empty() {
                        continue;
                    }

                    let layer_texture_bindings =
                        self.texture_bindings_with_scheduled_layers(rendered_targets);
                    self.render_scene(
                        commands,
                        &target.render_size,
                        &target.view,
                        encoded_paints,
                        clear_next_draw,
                        DirectTarget::AtlasLayer,
                        target.origin,
                        &layer_texture_bindings,
                    )?;
                    clear_next_draw = false;
                }
                ScheduledLayerOp::CompositeLayer(layer_id) => {
                    if clear_next_draw {
                        Renderer::clear_view(self.encoder, &target.view);
                        self.renderer.record_render_pass();
                        clear_next_draw = false;
                    }
                    self.composite_layer(
                        parent_idx,
                        target,
                        *layer_id,
                        encoded_paints,
                        rendered_targets,
                    )?;
                }
            }
        }

        Ok(())
    }

    fn composite_layer(
        &mut self,
        parent_idx: usize,
        parent: &LayerTarget,
        layer_id: RecordedLayerId,
        encoded_paints: &mut Vec<EncodedPaint>,
        rendered_targets: &[Option<LayerTarget>],
    ) -> Result<(), RenderError> {
        let layer_idx = layer_id.as_usize();
        let layer = &self.scene.layers()[layer_idx];
        let child = rendered_targets[layer_idx]
            .as_ref()
            .expect("child layer must be rendered before it is composited");
        if layer_can_be_sampled_directly(layer) {
            let commands = layer_sample_command_with_paint_idx(
                layer,
                LayerSampleExtent::Output,
                self.paint_indices.output[layer_idx]
                    .expect("output layer sample paint must be preplanned"),
            );
            let commands = commands.as_slice();
            let layer_texture_bindings =
                self.texture_bindings_with_scheduled_layers(rendered_targets);
            return self.render_scene(
                commands,
                &parent.render_size,
                &parent.view,
                encoded_paints,
                false,
                DirectTarget::AtlasLayer,
                parent.origin,
                &layer_texture_bindings,
            );
        }

        let source = self.create_aligned_layer_source(
            parent,
            layer_idx,
            layer,
            child,
            encoded_paints,
            rendered_targets,
        )?;
        let backdrop = self.create_extra_layer_target(
            backdrop_texture_id(parent_idx, layer_idx),
            parent.render_size.clone(),
            parent.origin,
        );
        self.renderer
            .copy_layer_target(self.encoder, self.scene, parent, &backdrop);
        self.renderer.composite_layer(
            self.device,
            self.encoder,
            &parent.render_size,
            parent,
            &source,
            &backdrop,
            layer.blend_mode,
            layer.opacity,
            layer_blend_scissor(layer, parent.origin),
        );
        Ok(())
    }

    fn create_aligned_layer_source(
        &mut self,
        parent: &LayerTarget,
        layer_idx: usize,
        layer: &RecordedLayer,
        _child: &LayerTarget,
        encoded_paints: &mut Vec<EncodedPaint>,
        rendered_targets: &[Option<LayerTarget>],
    ) -> Result<LayerTarget, RenderError> {
        let aligned_source = self.create_extra_layer_target(
            aligned_layer_source_texture_id(layer_idx),
            parent.render_size.clone(),
            parent.origin,
        );
        let commands = layer_sample_command_with_paint_idx(
            layer,
            LayerSampleExtent::Content,
            self.paint_indices.content[layer_idx]
                .expect("content layer sample paint must be preplanned"),
        );
        let commands = commands.as_slice();
        let layer_texture_bindings = self.texture_bindings_with_scheduled_layers(rendered_targets);
        self.render_scene(
            commands,
            &aligned_source.render_size,
            &aligned_source.view,
            encoded_paints,
            true,
            DirectTarget::AtlasLayer,
            aligned_source.origin,
            &layer_texture_bindings,
        )?;
        Ok(aligned_source)
    }

    fn apply_filter_to_layer(
        &mut self,
        layer_idx: usize,
        layer: &RecordedLayer,
        source: LayerTarget,
    ) -> Result<LayerTarget, RenderError> {
        let filter_data = layer
            .filter
            .as_ref()
            .expect("filter target requested for a non-filter layer");
        let plan = LayerFilterPlan::new(filter_data, (source.width(), source.height()));
        let filter_data_offset = self.layer_filter_data_offset;
        self.layer_filter_data_offset += GpuFilterData::SIZE_TEXELS;
        self.renderer.programs.upload_layer_filter_data(
            self.device,
            self.queue,
            filter_data_offset,
            plan.gpu_filter(),
        );
        let final_target = self.create_extra_layer_target(
            source.texture_id,
            source.render_size.clone(),
            source.origin,
        );

        if !plan.uses_scratch() {
            let pass = plan
                .passes()
                .first()
                .expect("single-pass filters must schedule one pass");
            self.run_filter_pass(
                &source,
                &final_target,
                &source,
                filter_data_offset,
                pass.pass_kind,
                pass.src_size,
                pass.dst_size,
            );
            return Ok(final_target);
        }

        let scratch_0 = self.create_extra_layer_target(
            filter_scratch_texture_id(layer_idx, 0),
            source.render_size.clone(),
            source.origin,
        );
        let scratch_1 = self.create_extra_layer_target(
            filter_scratch_texture_id(layer_idx, 1),
            source.render_size.clone(),
            source.origin,
        );
        for pass in plan.passes() {
            self.run_scheduled_filter_pass(
                pass.input,
                pass.output,
                &source,
                &scratch_0,
                &scratch_1,
                &final_target,
                filter_data_offset,
                pass.pass_kind,
                pass.src_size,
                pass.dst_size,
            );
        }

        Ok(final_target)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_scheduled_filter_pass(
        &mut self,
        input: FilterPassSource,
        output: FilterPassSource,
        source: &LayerTarget,
        scratch_0: &LayerTarget,
        scratch_1: &LayerTarget,
        final_target: &LayerTarget,
        filter_data_offset: u32,
        pass_kind: u32,
        src_size: (u16, u16),
        dst_size: (u16, u16),
    ) {
        let input = filter_pass_target(input, source, scratch_0, scratch_1, final_target);
        let output = filter_pass_target(output, source, scratch_0, scratch_1, final_target);
        self.run_filter_pass(
            input,
            output,
            source,
            filter_data_offset,
            pass_kind,
            src_size,
            dst_size,
        );
    }

    fn run_filter_pass(
        &mut self,
        input: &LayerTarget,
        output: &LayerTarget,
        original: &LayerTarget,
        filter_data_offset: u32,
        pass_kind: u32,
        src_size: (u16, u16),
        dst_size: (u16, u16),
    ) {
        let instance = FilterInstanceData {
            src: IntRect::new([0, 0], [u32::from(src_size.0), u32::from(src_size.1)]),
            dest: IntRect::new([0, 0], [u32::from(dst_size.0), u32::from(dst_size.1)]),
            dest_atlas_size: IntSize([output.render_size.width, output.render_size.height]),
            filter_data_offset,
            original: IntRect::new(
                [0, 0],
                [u32::from(original.width()), u32::from(original.height())],
            ),
            pass_kind,
        };
        let instance_size = size_of::<FilterInstanceData>() as u64;
        let instance_offset = self.layer_filter_instances.len() as u64 * instance_size;
        self.layer_filter_instances.push(instance);
        debug_assert!(
            instance_offset + instance_size
                <= self
                    .renderer
                    .programs
                    .resources
                    .filter_instance_buffer_capacity
        );
        self.renderer.record_render_pass();
        let programs = &self.renderer.programs;

        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Layer Filter Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &output.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        let [x, y, width, height] =
            instance.scissor_rect([output.render_size.width, output.render_size.height]);
        render_pass.set_scissor_rect(x, y, width.max(1), height.max(1));
        render_pass.set_pipeline(&programs.filter_pipeline);
        render_pass.set_bind_group(0, &programs.resources.layer_filter_base_bind_group, &[]);
        render_pass.set_bind_group(1, &input.filter_input_bind_group, &[]);
        render_pass.set_bind_group(2, &original.filter_original_bind_group, &[]);
        render_pass.set_vertex_buffer(
            0,
            programs
                .resources
                .filter_instance_buffer
                .slice(instance_offset..instance_offset + instance_size),
        );
        render_pass.draw(0..4, 0..1);
    }
}

/// Vello Hybrid's Renderer.
#[derive(Debug)]
pub struct Renderer {
    /// Programs for rendering.
    programs: Programs,
    /// Encoded paints for storing encoded paints.
    encoded_paints: Vec<GpuEncodedPaint>,
    /// Stores the index (offset) of the encoded paints in the encoded paints texture.
    paint_idxs: Vec<u32>,
    /// Gradient cache for storing gradient ramps.
    gradient_cache: GradientRampCache,
    /// Context for GPU filter effects.
    filter_context: FilterContext,
    /// State used for constructing filter passes.
    filter_pass_state: FilterPassState,
    /// Diagnostic count of render passes issued for the current scene render.
    render_pass_count: usize,
    /// Reusable offscreen layer textures and bind groups.
    layer_target_pool: LayerTargetPool,
    dummy_image_cache: Option<ImageCache>,
    atlas_clear_scratch: Vec<u8>,
}

impl Renderer {
    /// Creates a new renderer.
    pub fn new(device: &Device, render_target_config: &RenderTargetConfig) -> Self {
        Self::new_with(device, render_target_config, RenderSettings::default())
    }

    /// Creates a new renderer with specific settings.
    pub fn new_with(
        device: &Device,
        render_target_config: &RenderTargetConfig,
        settings: RenderSettings,
    ) -> Self {
        super::common::maybe_warn_about_webgl_feature_conflict();

        let mut settings = settings;
        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        // When targeting wasm32 with a WebGL/GLES backend, we need to set
        // `initial_atlas_count` to 2. In WGPU's GLES backend, heuristics are used to decide
        // whether a texture should be treated as D2 or D2Array. However, this can cause a
        // mismatch: when depth_or_array_layers == 1, the backend assumes the texture is D2,
        // even if it was actually created as a D2Array. This issue only occurs with the GLES
        // backend.
        //
        // @see https://github.com/gfx-rs/wgpu/blob/61e5124eb9530d3b3865556a7da4fd320d03ddc5/wgpu-hal/src/gles/mod.rs#L470-L517
        // TODO: Can we somehow dynamically detect whether the WebGL backend was chosen, so that the
        // wgpu backend isn't affected by this?
        #[cfg(target_arch = "wasm32")]
        let min_initial_atlas_count = 2;
        #[cfg(not(target_arch = "wasm32"))]
        let min_initial_atlas_count = 1;
        normalize_atlas_config(
            &mut settings.atlas_config,
            max_texture_dimension_2d,
            device.limits().max_texture_array_layers,
            min_initial_atlas_count,
        );
        let image_cache = ImageCache::new_with_config(settings.atlas_config);
        // Estimate the maximum number of gradient cache entries based on the max texture dimension
        // and the maximum gradient LUT size - worst case scenario.
        let max_gradient_cache_size =
            max_texture_dimension_2d * max_texture_dimension_2d / MAX_GRADIENT_LUT_SIZE as u32;
        let gradient_cache = GradientRampCache::new(max_gradient_cache_size, settings.level);

        let filter_context = FilterContext::new(settings.atlas_config);
        Self {
            programs: Programs::new(
                device,
                &image_cache,
                &filter_context.image_cache,
                render_target_config,
            ),
            gradient_cache,
            encoded_paints: Vec::new(),
            paint_idxs: Vec::new(),
            filter_context,
            filter_pass_state: FilterPassState::default(),
            render_pass_count: 0,
            layer_target_pool: LayerTargetPool::default(),
            dummy_image_cache: Some(ImageCache::new_dummy()),
            atlas_clear_scratch: Vec::new(),
        }
    }

    #[inline]
    fn reset_render_pass_count(&mut self) {
        self.render_pass_count = 0;
    }

    #[inline]
    fn record_render_pass(&mut self) {
        self.render_pass_count += 1;
    }

    fn print_render_pass_count(&self) {
        #[cfg(feature = "std")]
        std::eprintln!("vello_hybrid render passes: {}", self.render_pass_count);
    }

    fn prepare_filter_textures(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        image_cache: &mut ImageCache,
    ) {
        // TODO: Maybe we can do the clear implicitly when using the textures for the first time.
        if !self.filter_context.filter_textures.is_empty() {
            self.render_pass_count += self.programs.resources.filter_atlas.views.len();
            for view in &self.programs.resources.filter_atlas.views {
                let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("Clear Filter Atlas Texture"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                });
            }
        }

        self.filter_context
            .deallocate_all_and_clear_context(image_cache);

        Programs::maybe_resize_atlas_texture_array(
            device,
            encoder,
            &mut self.programs.resources,
            &self.programs.atlas_bind_group_layout,
            image_cache.atlas_count() as u32,
        );
        self.programs.resources.filter_atlas.ensure_count(
            device,
            self.filter_context.image_cache.atlas_count() as u32,
            &self.programs.filter_input_bind_group_layouts[0],
            &self.programs.filter_input_bind_group_layouts[1],
        );
    }

    /// Render `scene`.
    ///
    /// Every [`TextureId`] referenced by the scene must have a binding; this returns
    /// [`RenderError::MissingTextureBinding`] otherwise. See [`TextureBindings::insert`] for the
    /// requirements on the bound texture views.
    ///
    /// To render without any texture bindings, you can pass an empty [`TextureBindings`].
    pub fn render(
        &mut self,
        scene: &Scene,
        resources: &mut Resources,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        view: &TextureView,
        texture_bindings: &TextureBindings,
    ) -> Result<(), RenderError> {
        #[cfg(feature = "text")]
        {
            resources.before_render(
                self,
                |renderer, glyph_renderer, atlas_count, atlas_config, atlas_id| {
                    renderer
                        .render_to_atlas(
                            glyph_renderer,
                            atlas_count,
                            atlas_config,
                            device,
                            queue,
                            atlas_id,
                            texture_bindings,
                        )
                        .expect("Failed to render glyphs to atlas");
                },
                |renderer, image_cache, upload, dst_x, dst_y| {
                    renderer.write_to_atlas(
                        image_cache,
                        device,
                        queue,
                        encoder,
                        upload.image_id,
                        &upload.pixmap,
                        Some([dst_x, dst_y]),
                    );
                },
            );
        }

        let mut encoded_paints = scene.encoded_paints.borrow_mut();
        let scene_paint_count = encoded_paints.len();

        self.reset_render_pass_count();
        self.prepare_filter_textures(device, encoder, &mut resources.image_cache);

        let result = self.render_scene_with_layers(
            scene,
            device,
            queue,
            encoder,
            render_size,
            view,
            &resources.image_cache,
            &mut encoded_paints,
            scene_paint_count,
            texture_bindings,
            DirectTarget::UserSurface,
            true,
        );

        encoded_paints.truncate(scene_paint_count);
        #[cfg(feature = "text")]
        resources.after_render(self, |renderer, rect| {
            clear_atlas_region(queue, renderer, rect);
        });
        self.print_render_pass_count();
        result
    }

    /// Render a `scene` directly into an atlas layer.
    ///
    /// This renders the scene's content into the specified atlas layer, which can then
    /// be sampled as an image in subsequent render passes. This is useful for rendering
    /// vector content (e.g., glyphs) into the atlas for later use as cached images.
    ///
    /// The scene should be sized to the atlas layer dimensions
    /// ([`AtlasConfig::atlas_size`]), with content positioned at the allocated offset
    /// coordinates from `ImageCache::allocate`.
    ///
    /// This method creates its own command encoder and submits immediately,
    /// ensuring atlas content is committed before any subsequent
    /// [`render`](Self::render) call (the two methods share GPU resources that
    /// are staged by `queue.write_*` and only applied on the next `queue.submit`).
    ///
    /// `texture_bindings` provides [externally bound textures](`TextureBindings`)
    /// referenced by the scene. Pass `&TextureBindings::new()` if the scene does
    /// not use any.
    #[doc(hidden)]
    pub fn render_to_atlas(
        &mut self,
        scene: &Scene,
        atlas_count: u32,
        atlas_config: AtlasConfig,
        device: &Device,
        queue: &Queue,
        atlas_id: AtlasId,
        texture_bindings: &TextureBindings,
    ) -> Result<(), RenderError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render to Atlas Encoder"),
        });

        Programs::maybe_resize_atlas_texture_array(
            device,
            &mut encoder,
            &mut self.programs.resources,
            &self.programs.atlas_bind_group_layout,
            atlas_count,
        );

        let (atlas_width, atlas_height) = atlas_config.atlas_size;
        let atlas_render_size = RenderSize {
            width: atlas_width,
            height: atlas_height,
        };

        let layer_view =
            self.programs
                .resources
                .atlas_texture_array
                .create_view(&TextureViewDescriptor {
                    label: Some("Atlas Layer Render View"),
                    format: Some(wgpu::TextureFormat::Rgba8Unorm),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    base_array_layer: atlas_id.as_u32(),
                    array_layer_count: Some(1),
                    usage: None,
                });

        // Swap in the stub atlas bind group to avoid the read-write conflict:
        // the real atlas texture is used as the render target (COLOR_TARGET), so it
        // cannot also be bound as a shader resource (TEXTURE_BINDING) in the same pass.
        core::mem::swap(
            &mut self.programs.resources.atlas_bind_group,
            &mut self.programs.resources.stub_atlas_bind_group,
        );
        core::mem::swap(
            &mut self.programs.resources.atlas_texture_array_view,
            &mut self.programs.resources.stub_atlas_texture_array_view,
        );

        let mut encoded_paints = scene.encoded_paints.borrow_mut();
        let scene_paint_count = encoded_paints.len();
        let dummy_image_cache = self
            .dummy_image_cache
            .take()
            .expect("dummy image cache must exist");
        let result = self.render_scene_with_layers(
            scene,
            device,
            queue,
            &mut encoder,
            &atlas_render_size,
            &layer_view,
            &dummy_image_cache,
            &mut encoded_paints,
            scene_paint_count,
            texture_bindings,
            DirectTarget::AtlasLayer,
            false,
        );
        encoded_paints.truncate(scene_paint_count);
        self.dummy_image_cache = Some(dummy_image_cache);

        // Restore the real atlas bind group.
        core::mem::swap(
            &mut self.programs.resources.atlas_bind_group,
            &mut self.programs.resources.stub_atlas_bind_group,
        );
        core::mem::swap(
            &mut self.programs.resources.atlas_texture_array_view,
            &mut self.programs.resources.stub_atlas_texture_array_view,
        );

        // Submit immediately so the atlas content is committed before subsequent
        // render() calls overwrite the shared alpha/config/paint resources.
        queue.submit(Some(encoder.finish()));

        result
    }

    /// Shared render pipeline: prepares GPU resources, renders direct strips against
    /// the provided `view` at `render_size`, and maintains caches.
    ///
    /// When `clear` is true the render target is cleared to transparent black
    /// before drawing (normal frame rendering).
    fn render_scene_with_layers(
        &mut self,
        scene: &Scene,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        view: &TextureView,
        image_cache: &ImageCache,
        encoded_paints: &mut Vec<EncodedPaint>,
        scene_paint_count: usize,
        texture_bindings: &TextureBindings,
        root_target: DirectTarget,
        root_clear: bool,
    ) -> Result<(), RenderError> {
        let scheduler = LayerScheduler::new(scene, scene_paint_count);
        let layer_render_size = RenderSize {
            width: u32::from(scene.width),
            height: u32::from(scene.height),
        };
        let visible_layers = scheduler.visible_layers();
        let (planned_layer_targets, planned_root_target, paint_indices) = self
            .plan_scheduled_layer_paints(
                scene,
                device,
                encoded_paints,
                &layer_render_size,
                &visible_layers,
            );
        let filter_instance_count =
            Self::scheduled_layer_filter_pass_count(scene, &layer_render_size, &visible_layers);
        self.programs.ensure_filter_instance_buffer_capacity(
            device,
            filter_instance_count as u64 * size_of::<FilterInstanceData>() as u64,
        );
        let scheduler = LayerScheduler::new(scene, encoded_paints.len());
        let mut schedule = WgpuLayerSchedule {
            renderer: self,
            scene,
            device,
            queue,
            encoder,
            root_render_size: render_size,
            root_target,
            root_clear,
            layer_render_size,
            root_view: view,
            image_cache,
            texture_bindings,
            planned_layer_targets,
            planned_root_target,
            paint_indices,
            extra_layer_targets: Vec::new(),
            layer_filter_instances: Vec::new(),
            shared_resources_uploaded: false,
            layer_filter_data_offset: 0,
        };
        let result = scheduler.render(encoded_paints, &mut schedule);
        schedule.finish();
        result?;
        Ok(())
    }

    fn plan_scheduled_layer_paints(
        &mut self,
        scene: &Scene,
        device: &Device,
        encoded_paints: &mut Vec<EncodedPaint>,
        layer_render_size: &RenderSize,
        visible_layers: &[bool],
    ) -> (
        Vec<Option<LayerTarget>>,
        Option<LayerTarget>,
        ScheduledPaintIndices,
    ) {
        let mut planned_targets = vec![None; scene.layers().len()];
        let mut paint_indices = ScheduledPaintIndices::new(scene.layers().len());

        for (layer_idx, layer) in scene.layers().iter().enumerate() {
            if !visible_layers[layer_idx] {
                continue;
            }

            let (origin, render_size) =
                layer_target_origin_and_size(scene, layer_idx, layer_render_size);
            let target =
                self.create_layer_target(device, layer_texture_id(layer_idx), render_size, origin);

            paint_indices.output[layer_idx] = Some(encoded_paints.len());
            encoded_paints.push(layer_sample_encoded_paint(
                layer,
                &target,
                layer.opacity,
                LayerSampleExtent::Output,
            ));

            paint_indices.content[layer_idx] = Some(encoded_paints.len());
            encoded_paints.push(layer_sample_encoded_paint(
                layer,
                &target,
                1.0,
                LayerSampleExtent::Content,
            ));

            planned_targets[layer_idx] = Some(target);
        }

        let planned_root_target =
            root_needs_scheduled_offscreen(scene, visible_layers).then(|| {
                self.create_layer_target(
                    device,
                    root_texture_id(),
                    layer_render_size.clone(),
                    RenderOrigin::default(),
                )
            });
        if let Some(target) = &planned_root_target {
            paint_indices.root = Some(encoded_paints.len());
            encoded_paints.push(layer_encoded_paint(
                target,
                1.0,
                RectU16::new(0, 0, target.width(), target.height()),
                vello_common::kurbo::Affine::IDENTITY,
            ));
        }

        (planned_targets, planned_root_target, paint_indices)
    }

    fn scheduled_layer_filter_pass_count(
        scene: &Scene,
        layer_render_size: &RenderSize,
        visible_layers: &[bool],
    ) -> usize {
        scene
            .layers()
            .iter()
            .enumerate()
            .filter(|(layer_idx, _)| visible_layers[*layer_idx])
            .filter_map(|(layer_idx, layer)| {
                let filter = layer.filter.as_ref()?;
                let (_, render_size) =
                    layer_target_origin_and_size(scene, layer_idx, layer_render_size);
                let plan = LayerFilterPlan::new(
                    filter,
                    (
                        render_size.width.try_into().unwrap_or(u16::MAX),
                        render_size.height.try_into().unwrap_or(u16::MAX),
                    ),
                );
                Some(plan.passes().len())
            })
            .sum()
    }

    fn create_layer_target(
        &mut self,
        device: &Device,
        texture_id: TextureId,
        render_size: RenderSize,
        origin: RenderOrigin,
    ) -> LayerTarget {
        self.layer_target_pool
            .acquire(device, &self.programs, texture_id, render_size, origin)
    }

    fn release_layer_target(&mut self, target: LayerTarget) {
        self.layer_target_pool.release(target);
    }

    fn copy_layer_target(
        &self,
        encoder: &mut CommandEncoder,
        _scene: &Scene,
        src: &LayerTarget,
        dst: &LayerTarget,
    ) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &src.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &dst.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width: src.render_size.width.min(dst.render_size.width).max(1),
                height: src.render_size.height.min(dst.render_size.height).max(1),
                depth_or_array_layers: 1,
            },
        );
    }

    fn composite_layer(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        dst: &LayerTarget,
        src: &LayerTarget,
        backdrop: &LayerTarget,
        blend_mode: peniko::BlendMode,
        opacity: f32,
        scissor_bbox: Option<RectU16>,
    ) {
        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Layer Blend Config"),
            contents: bytemuck::bytes_of(&LayerBlendConfig {
                mix_mode: blend_mode.mix as u32,
                compose_mode: blend_mode.compose as u32,
                opacity,
                _padding: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let config_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Layer Blend Config Bind Group"),
            layout: &self.programs.layer_blend_config_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: config_buffer.as_entire_binding(),
            }],
        });
        let src_bind_group = self
            .programs
            .create_layer_blend_texture_bind_group(device, &src.view);
        let backdrop_bind_group = self
            .programs
            .create_layer_blend_texture_bind_group(device, &backdrop.view);

        self.record_render_pass();
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Layer Blend Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &dst.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&self.programs.layer_blend_pipeline);
        render_pass.set_bind_group(0, &config_bind_group, &[]);
        render_pass.set_bind_group(1, &src_bind_group, &[]);
        render_pass.set_bind_group(2, &backdrop_bind_group, &[]);
        if let Some(scissor_bbox) = scissor_bbox {
            render_pass.set_scissor_rect(
                u32::from(scissor_bbox.x0),
                u32::from(scissor_bbox.y0),
                u32::from(scissor_bbox.width()).max(1),
                u32::from(scissor_bbox.height()).max(1),
            );
        }
        render_pass.set_viewport(
            0.0,
            0.0,
            render_size.width as f32,
            render_size.height as f32,
            0.0,
            1.0,
        );
        render_pass.draw(0..4, 0..1);
    }

    fn render_scene(
        &mut self,
        scene: &Scene,
        commands: &[FastStripCommand],
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        render_size: &RenderSize,
        view: &TextureView,
        image_cache: &ImageCache,
        encoded_paints: &[EncodedPaint],
        clear: bool,
        target: DirectTarget,
        origin: RenderOrigin,
        texture_bindings: &TextureBindings,
        upload_shared_resources: bool,
    ) -> Result<(), RenderError> {
        self.programs.depth_cleared_this_frame = false;
        if upload_shared_resources {
            self.prepare_gpu_encoded_paints(encoded_paints, image_cache, texture_bindings)?;
        }
        // TODO: For the time being, we upload the entire alpha buffer as one big chunk. As a future
        // refinement, we could have a bounded alpha buffer, and break draws when the alpha
        // buffer fills.
        self.programs.prepare(
            device,
            queue,
            &mut self.gradient_cache,
            &self.encoded_paints,
            &mut scene.strip_storage.borrow_mut().alphas,
            render_size,
            origin,
            target,
            &self.paint_idxs,
            &self.filter_context,
            upload_shared_resources,
        );

        let target_x_limit = origin
            .x
            .saturating_add(render_size.width.try_into().unwrap_or(u16::MAX));
        let strips = DirectStrips::from_commands(
            scene,
            commands,
            target,
            target_x_limit,
            &self.paint_idxs,
            encoded_paints,
        );
        if strips.is_empty() {
            if clear {
                Self::clear_view(encoder, view);
                self.record_render_pass();
            }
            self.gradient_cache.maintain();
            return Ok(());
        }

        let mut ctx = RendererContext {
            programs: &mut self.programs,
            device,
            queue,
            encoder,
            view,
            image_cache,
            filter_context: &self.filter_context,
            filter_pass_state: &mut self.filter_pass_state,
            texture_bindings,
            external_paint_source_bind_groups: HashMap::new(),
            render_pass_count: &mut self.render_pass_count,
        };
        ctx.do_strip_render_pass(
            strips.opaque(),
            strips.alpha(),
            strips.external_texture_runs(),
            target,
            if clear {
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
            } else {
                wgpu::LoadOp::Load
            },
        );
        self.gradient_cache.maintain();

        Ok(())
    }

    /// Clear the view to transparent black.
    // TODO: Investigate adding tests for the clear_view behavior.
    fn clear_view(encoder: &mut CommandEncoder, view: &TextureView) {
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Clear View"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
    }

    /// Upload image to cache and atlas in one step. Returns the `ImageId`.
    ///
    /// It's used when an image is not already in the cache.
    ///
    /// This is a convenience method that:
    /// 1. Reserves space in the image cache
    /// 2. Writes the image data directly to the atlas
    /// 3. Returns the `ImageId` for use in rendering
    pub fn upload_image<T: AtlasWriter>(
        &mut self,
        resources: &mut Resources,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        writer: &T,
    ) -> vello_common::paint::ImageId {
        self.upload_image_with(
            &mut resources.image_cache,
            device,
            queue,
            encoder,
            writer,
            IMAGE_PADDING,
        )
    }

    pub(crate) fn upload_image_with<T: AtlasWriter>(
        &mut self,
        image_cache: &mut ImageCache,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        writer: &T,
        padding: u16,
    ) -> vello_common::paint::ImageId {
        let width = writer.width();
        let height = writer.height();
        let image_id = image_cache.allocate(width, height, padding).unwrap();
        self.write_to_atlas(image_cache, device, queue, encoder, image_id, writer, None);
        image_id
    }

    /// Write pixel data to an existing atlas allocation.
    ///
    /// Unlike [`upload_image`](Self::upload_image), this does not allocate space in the image
    /// cache. The `image_id` must have been previously allocated (e.g. via
    /// `ImageCache::allocate`). This is useful for uploading CPU-side pixel data (such as
    /// bitmap font glyphs) to a pre-allocated atlas region.
    ///
    /// If `offset_override` is `Some`, the provided offset is used instead of the
    /// allocator-assigned position. Pass `None` to use the default atlas offset.
    pub(crate) fn write_to_atlas<T: AtlasWriter>(
        &mut self,
        image_cache: &ImageCache,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        image_id: vello_common::paint::ImageId,
        writer: &T,
        offset_override: Option<[u32; 2]>,
    ) {
        let image_resource = image_cache.get(image_id).expect("Image resource not found");

        Programs::maybe_resize_atlas_texture_array(
            device,
            encoder,
            &mut self.programs.resources,
            &self.programs.atlas_bind_group_layout,
            image_cache.atlas_count() as u32,
        );
        let offset = offset_override.unwrap_or([
            image_resource.offset[0] as u32,
            image_resource.offset[1] as u32,
        ]);
        writer.write_to_atlas_layer(
            device,
            queue,
            encoder,
            &self.programs.resources.atlas_texture_array,
            image_resource.atlas_id.as_u32(),
            offset,
            writer.width(),
            writer.height(),
        );
    }

    /// Destroy an image from the cache and clear the allocated slot in the atlas.
    pub fn destroy_image(
        &mut self,
        resources: &mut Resources,
        _device: &Device,
        queue: &Queue,
        _encoder: &mut CommandEncoder,
        image_id: vello_common::paint::ImageId,
    ) {
        if let Some(image_resource) = resources.image_cache.deallocate(image_id) {
            let padding = image_resource.padding as u32;

            self.clear_atlas_region(
                queue,
                image_resource.atlas_id,
                [
                    image_resource.offset[0] as u32 - padding,
                    image_resource.offset[1] as u32 - padding,
                ],
                image_resource.width as u32 + padding * 2,
                image_resource.height as u32 + padding * 2,
            );
        }
    }

    /// Returns a reference to the underlying atlas texture array.
    ///
    /// This is a 2D array texture (`TextureViewDimension::D2Array`) containing all
    /// atlas layers used by the image cache. Each layer holds cached image data
    /// (e.g., rasterised glyphs) that the renderer samples during draw calls.
    pub fn atlas_texture(&self) -> &Texture {
        &self.programs.resources.atlas_texture_array
    }

    /// Clear a specific region of the atlas texture.
    fn clear_atlas_region(
        &mut self,
        queue: &Queue,
        atlas_id: AtlasId,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        if width == 0 || height == 0 {
            return;
        }

        let byte_count = width as usize * height as usize * 4;
        self.atlas_clear_scratch.resize(byte_count, 0);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.programs.resources.atlas_texture_array,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: offset[0],
                    y: offset[1],
                    z: atlas_id.as_u32(),
                },
                aspect: wgpu::TextureAspect::All,
            },
            &self.atlas_clear_scratch[..byte_count],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: None,
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn prepare_gpu_encoded_paints(
        &mut self,
        encoded_paints: &[EncodedPaint],
        image_cache: &ImageCache,
        texture_bindings: &TextureBindings,
    ) -> Result<(), RenderError> {
        self.encoded_paints
            .resize_with(encoded_paints.len(), || GPU_PAINT_PLACEHOLDER);
        self.paint_idxs.resize(encoded_paints.len() + 1, 0);

        let mut current_idx = 0;
        for (encoded_paint_idx, paint) in encoded_paints.iter().enumerate() {
            self.paint_idxs[encoded_paint_idx] = current_idx;
            match paint {
                EncodedPaint::Image(img) => {
                    if let ImageSource::OpaqueId { id: image_id, .. } = img.source {
                        let image_resource: Option<&ImageResource> = image_cache.get(image_id);
                        if let Some(image_resource) = image_resource {
                            let image_paint = self.encode_image_paint(img, image_resource);
                            self.encoded_paints[encoded_paint_idx] = image_paint;
                            current_idx += GPU_ENCODED_IMAGE_SIZE_TEXELS;
                        }
                    }
                }
                EncodedPaint::ExternalTexture(img) => {
                    if texture_bindings.get(img.texture_id).is_none() {
                        return Err(RenderError::MissingTextureBinding(img.texture_id));
                    }
                    let image_paint = self.encode_external_texture_paint(img);
                    self.encoded_paints[encoded_paint_idx] = image_paint;
                    current_idx += GPU_ENCODED_IMAGE_SIZE_TEXELS;
                }
                EncodedPaint::Gradient(gradient) => {
                    let (gradient_start, gradient_width) =
                        self.gradient_cache.get_or_create_ramp(gradient);
                    let gradient_paint: GpuEncodedPaint =
                        self.encode_gradient_paint(gradient, gradient_width, gradient_start);
                    let gradient_size_texels = match &gradient_paint {
                        GpuEncodedPaint::LinearGradient(_) => GPU_LINEAR_GRADIENT_SIZE_TEXELS,
                        GpuEncodedPaint::RadialGradient(_) => GPU_RADIAL_GRADIENT_SIZE_TEXELS,
                        GpuEncodedPaint::SweepGradient(_) => GPU_SWEEP_GRADIENT_SIZE_TEXELS,
                        _ => unreachable!("encode_gradient_for_gpu only returns gradient types"),
                    };
                    self.encoded_paints[encoded_paint_idx] = gradient_paint;
                    current_idx += gradient_size_texels;
                }
                EncodedPaint::BlurredRoundedRect(blurred_rect) => {
                    self.encoded_paints[encoded_paint_idx] =
                        Self::encode_blurred_rounded_rect_paint(blurred_rect);
                    current_idx += GPU_BLURRED_ROUNDED_RECT_SIZE_TEXELS;
                }
            }
        }
        self.paint_idxs[encoded_paints.len()] = current_idx;
        Ok(())
    }

    fn encode_image_paint(
        &self,
        image: &vello_common::encode::EncodedImage,
        image_resource: &ImageResource,
    ) -> GpuEncodedPaint {
        let transform = image.transform.as_coeffs().map(|x| x as f32);
        let image_size = pack_image_size(image_resource.width, image_resource.height);
        let image_offset = pack_image_offset(image_resource.offset[0], image_resource.offset[1]);
        let image_params = pack_image_params(
            image.sampler.quality as u32,
            image.sampler.x_extend as u32,
            image.sampler.y_extend as u32,
            image_resource.atlas_id.as_u32(),
        );
        let (tint, tint_mode) = pack_tint(image.tint);

        GpuEncodedPaint::Image(GpuEncodedImage {
            image_params,
            image_size,
            image_offset,
            transform,
            tint,
            tint_mode,
            image_padding: image_resource.padding as u32,
        })
    }

    fn encode_external_texture_paint(&self, image: &EncodedExternalTexture) -> GpuEncodedPaint {
        let transform = image.transform.as_coeffs().map(|x| x as f32);
        let region = image.source_region;
        let image_size = pack_image_size(region.width(), region.height());
        let image_offset = pack_image_offset(region.x0, region.y0);
        let image_params = pack_image_params(
            image.sampler.quality as u32,
            image.sampler.x_extend as u32,
            image.sampler.y_extend as u32,
            0,
        ) | EXTERNAL_IMAGE_SOURCE_FLAG;
        let (tint, tint_mode) = pack_tint(image.tint);

        GpuEncodedPaint::Image(GpuEncodedImage {
            image_params,
            image_size,
            image_offset,
            transform,
            tint,
            tint_mode,
            image_padding: 0,
        })
    }

    fn encode_gradient_paint(
        &self,
        gradient: &EncodedGradient,
        gradient_width: u32,
        gradient_start: u32,
    ) -> GpuEncodedPaint {
        let transform = gradient.transform.as_coeffs().map(|x| x as f32);
        let extend_mode = match gradient.extend {
            peniko::Extend::Pad => 0,
            peniko::Extend::Repeat => 1,
            peniko::Extend::Reflect => 2,
        };
        let texture_width_and_extend_mode =
            pack_texture_width_and_extend_mode(gradient_width, extend_mode);

        match &gradient.kind {
            EncodedKind::Linear(_) => GpuEncodedPaint::LinearGradient(GpuLinearGradient {
                texture_width_and_extend_mode,
                gradient_start,
                transform,
            }),
            EncodedKind::Radial(radial) => {
                let (kind, bias, scale, fp0, fp1, fr1, f_focal_x, f_is_swapped, scaled_r0_squared) =
                    match radial {
                        RadialKind::Radial { bias, scale } => {
                            (0, *bias, *scale, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
                        }
                        RadialKind::Strip { scaled_r0_squared } => {
                            (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, *scaled_r0_squared)
                        }
                        RadialKind::Focal {
                            focal_data,
                            fp0,
                            fp1,
                        } => (
                            2,
                            *fp0,
                            *fp1,
                            *fp0,
                            *fp1,
                            focal_data.fr1,
                            focal_data.f_focal_x,
                            focal_data.f_is_swapped as u32,
                            0.0,
                        ),
                    };
                GpuEncodedPaint::RadialGradient(GpuRadialGradient {
                    texture_width_and_extend_mode,
                    gradient_start,
                    transform,
                    kind_and_f_is_swapped: pack_radial_kind_and_swapped(kind, f_is_swapped),
                    bias,
                    scale,
                    fp0,
                    fp1,
                    fr1,
                    f_focal_x,
                    scaled_r0_squared,
                })
            }
            EncodedKind::Sweep(sweep) => GpuEncodedPaint::SweepGradient(GpuSweepGradient {
                texture_width_and_extend_mode,
                gradient_start,
                transform,
                start_angle: sweep.start_angle,
                inv_angle_delta: sweep.inv_angle_delta,
                _padding: [0, 0],
            }),
        }
    }

    fn encode_blurred_rounded_rect_paint(rect: &EncodedBlurredRoundedRectangle) -> GpuEncodedPaint {
        GpuEncodedPaint::BlurredRoundedRect(GpuBlurredRoundedRect {
            transform: rect.transform.as_coeffs().map(|x| x as f32),
            color: rect.color.as_premul_rgba8().to_u32(),
            _padding0: 0,
            params0: [
                rect.exponent,
                rect.recip_exponent,
                rect.scale,
                rect.std_dev_inv,
            ],
            params1: [rect.min_edge, rect.w, rect.h, rect.r1],
            size: [rect.width, rect.height],
            _padding1: [0, 0],
        })
    }
}

#[cfg(feature = "text")]
fn clear_atlas_region(queue: &Queue, renderer: &mut Renderer, rect: &PendingClearRect) {
    renderer.clear_atlas_region(
        queue,
        AtlasId::new(rect.page_index),
        [rect.x as u32, rect.y as u32],
        rect.width as u32,
        rect.height as u32,
    );
}

/// Defines the GPU resources and pipelines for rendering.
#[derive(Debug)]
struct Programs {
    /// Pipelines for rendering strips to intermediate textures (depth test OFF, depth write OFF, blending ON).
    /// The first pipeline should be used for color attachments in the native pixel format,
    /// the second for color attachments in RGBA8.
    intermediate_strip_pipelines: [RenderPipeline; 2],
    /// Alpha pipelines for rendering strips to Output targets (depth test ON, depth write OFF, blending ON).
    alpha_strip_pipelines: [RenderPipeline; 2],
    /// Opaque pipelines for rendering strips to Output targets (depth test ON, depth write ON, blending OFF).
    opaque_strip_pipelines: [RenderPipeline; 2],
    /// Depth texture for early-z rejection on the Output target.
    depth_texture: Texture,
    /// View for the depth texture.
    depth_texture_view: TextureView,
    /// Whether the depth buffer has been cleared this frame.
    depth_cleared_this_frame: bool,
    /// Bind group layout for strip draws
    strip_bind_group_layout: BindGroupLayout,
    /// Bind group layout for encoded paints
    encoded_paints_bind_group_layout: BindGroupLayout,
    /// Bind group layout for gradient texture
    gradient_bind_group_layout: BindGroupLayout,
    /// Bind group layout for atlas textures
    atlas_bind_group_layout: BindGroupLayout,
    /// Bind group layout for filter data texture.
    filter_bind_group_layout: BindGroupLayout,
    /// Pipeline for applying filter effects.
    filter_pipeline: RenderPipeline,
    /// Pipeline for compositing a rendered layer over a parent backdrop.
    layer_blend_pipeline: RenderPipeline,
    /// Bind group layouts for filter input.
    filter_input_bind_group_layouts: [BindGroupLayout; 2],
    /// Bind group layout for layer blend configuration.
    layer_blend_config_bind_group_layout: BindGroupLayout,
    /// Bind group layout for layer blend input textures.
    layer_blend_texture_bind_group_layout: BindGroupLayout,
    /// GPU resources for rendering (created during prepare)
    resources: GpuResources,
    /// Dimensions of the rendering target
    render_size: RenderSize,
    /// Scratch buffer for staging encoded paints texture data.
    encoded_paints_data: Vec<u8>,
    /// Scratch buffer for staging filter data texture data.
    filter_data: Vec<u8>,
    /// Stable view config buffers/bind groups for render passes recorded into one encoder.
    view_config_cache: Vec<ViewConfigCacheEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ViewConfigKey {
    render_size: RenderSize,
    origin: RenderOrigin,
}

#[derive(Clone, Debug)]
struct ViewConfigCacheEntry {
    key: ViewConfigKey,
    buffer: Buffer,
    bind_group: BindGroup,
}

#[derive(Debug)]
struct FilterAtlasState {
    textures: Vec<Texture>,
    views: Vec<TextureView>,
    input_bind_groups: Vec<BindGroup>,
    original_bind_groups: Vec<BindGroup>,
    sampler: Sampler,
    atlas_size: (u32, u32),
}

impl FilterAtlasState {
    fn new(device: &Device, atlas_size: (u32, u32)) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Filter Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            textures: Vec::new(),
            views: Vec::new(),
            input_bind_groups: Vec::new(),
            original_bind_groups: Vec::new(),
            sampler,
            atlas_size,
        }
    }

    fn ensure_count(
        &mut self,
        device: &Device,
        required_count: u32,
        input_layout: &BindGroupLayout,
        original_layout: &BindGroupLayout,
    ) {
        let current_count = self.textures.len() as u32;

        if required_count <= current_count {
            return;
        }
        let (width, height) = self.atlas_size;

        for _ in current_count..required_count {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Filter Atlas Texture"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            let input_bg =
                create_filter_input_bind_group(device, input_layout, &self.sampler, &view);
            let original_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: original_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                }],
            });
            self.textures.push(texture);
            self.views.push(view);
            self.input_bind_groups.push(input_bg);
            self.original_bind_groups.push(original_bg);
        }
    }
}

/// Contains all GPU resources needed for rendering
#[derive(Debug)]
struct GpuResources {
    /// Buffer for [`GpuStrip`] data
    strips_buffer: Buffer,
    /// Texture for alpha values.
    alphas_texture: Texture,
    /// Textures for atlas data (multiple atlases supported)
    atlas_texture_array: Texture,
    /// View for atlas texture array
    atlas_texture_array_view: TextureView,
    /// Bind group for paint sources: an atlas textures as texture array plus an external texture.
    atlas_bind_group: BindGroup,
    /// Transparent 1x1 placeholder texture in case no external texture is bound by the user.
    placeholder_external_texture_view: TextureView,
    /// Filter atlas textures and their associated views/bind groups.
    /// Lazily allocated: stays empty until the first scene with filters.
    filter_atlas: FilterAtlasState,
    /// Texture for encoded paints
    encoded_paints_texture: Texture,
    /// Bind group for encoded paints
    encoded_paints_bind_group: BindGroup,
    /// Texture for gradient lookup table
    gradient_texture: Texture,
    /// Bind group for gradient texture
    gradient_bind_group: BindGroup,
    /// Texture holding serialized `GpuFilterData` for all filter layers.
    filter_data_texture: Texture,
    /// Bind group for the filter data texture.
    filter_base_bind_group: BindGroup,
    /// Texture holding the single `GpuFilterData` used by scheduled layer filters.
    layer_filter_data_texture: Texture,
    /// Current height of the scheduled layer filter data texture.
    layer_filter_data_texture_height: u32,
    /// Bind group for scheduled layer filter data.
    layer_filter_base_bind_group: BindGroup,

    /// Config buffer for rendering wide tile commands into the view texture.
    view_config_buffer: Buffer,

    /// Buffer holding `FilterInstanceData` for a single filter draw call.
    filter_instance_buffer: Buffer,
    filter_instance_buffer_capacity: u64,
    /// Bind group for strip rendering.
    strip_bind_group: BindGroup,

    /// Placeholder paint-source bind group with a 1x1 dummy atlas texture, used during
    /// `render_to_atlas` to avoid a read-write conflict on the real atlas texture.
    stub_atlas_bind_group: BindGroup,
    /// Placeholder atlas texture view used when creating external paint-source bind groups
    /// during `render_to_atlas`.
    stub_atlas_texture_array_view: TextureView,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct LayerBlendConfig {
    mix_mode: u32,
    compose_mode: u32,
    opacity: f32,
    _padding: u32,
}

impl GpuStrip {
    /// Vertex attributes for the strip
    pub fn vertex_attributes() -> [wgpu::VertexAttribute; 6] {
        wgpu::vertex_attr_array![
            0 => Uint32,
            1 => Uint32,
            2 => Uint32,
            3 => Uint32,
            4 => Uint32,
            5 => Uint32,
        ]
    }
}

impl Programs {
    fn new(
        device: &Device,
        image_cache: &ImageCache,
        filter_texture_cache: &ImageCache,
        render_target_config: &RenderTargetConfig,
    ) -> Self {
        let strip_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Strip Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let atlas_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Paint Source Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
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
                ],
            });

        let encoded_paints_bind_group_layout =
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
            });

        let gradient_bind_group_layout =
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
            });

        let strip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Strip Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::RENDER_STRIPS.into()),
        });

        let strip_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Strip Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&strip_bind_group_layout),
                    Some(&atlas_bind_group_layout),
                    Some(&encoded_paints_bind_group_layout),
                    Some(&gradient_bind_group_layout),
                ],
                immediate_size: 0,
            });

        let depth_format = wgpu::TextureFormat::Depth24Plus;
        let strip_formats = [render_target_config.format, wgpu::TextureFormat::Rgba8Unorm];

        let strip_vertex_state = wgpu::VertexBufferLayout {
            array_stride: size_of::<GpuStrip>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &GpuStrip::vertex_attributes(),
        };

        let create_strip_pipelines =
            |label, blend, depth_stencil: Option<wgpu::DepthStencilState>| -> [RenderPipeline; 2] {
                core::array::from_fn(|i| {
                    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some(label),
                        layout: Some(&strip_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &strip_shader,
                            entry_point: Some("vs_main"),
                            buffers: core::slice::from_ref(&strip_vertex_state),
                            compilation_options: PipelineCompilationOptions::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &strip_shader,
                            entry_point: Some("fs_main"),
                            targets: &[Some(ColorTargetState {
                                format: strip_formats[i],
                                blend,
                                write_mask: ColorWrites::ALL,
                            })],
                            compilation_options: PipelineCompilationOptions::default(),
                        }),
                        primitive: wgpu::PrimitiveState {
                            topology: wgpu::PrimitiveTopology::TriangleStrip,
                            ..Default::default()
                        },
                        depth_stencil: depth_stencil.clone(),
                        multisample: wgpu::MultisampleState::default(),
                        multiview_mask: None,
                        cache: None,
                    })
                })
            };

        let depth_stencil = |depth_write_enabled| wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: Some(depth_write_enabled),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        // Intermediate pipelines: depth test OFF, depth write OFF, blending ON.
        let intermediate_strip_pipelines = create_strip_pipelines(
            "Strip Intermediate Pipeline",
            Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            None,
        );
        // Alpha pipelines: depth test ON (LessEqual), depth write OFF, blending ON.
        let alpha_strip_pipelines = create_strip_pipelines(
            "Strip Alpha Pipeline",
            Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            Some(depth_stencil(false)),
        );
        // Opaque pipelines: depth test ON (LessEqual), depth write ON, blending OFF.
        let opaque_strip_pipelines =
            create_strip_pipelines("Strip Opaque Pipeline", None, Some(depth_stencil(true)));

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
        let layer_blend_config_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Layer Blend Config Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let layer_blend_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Layer Blend Texture Bind Group Layout"),
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
            // The original texture.
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Filter Original Bind Group Layout"),
                entries: &[filter_texture_entry],
            }),
        ];

        let filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Filter Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::FILTERS.into()),
        });
        let layer_blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Layer Blend Shader"),
            source: wgpu::ShaderSource::Wgsl(vello_sparse_shaders::wgsl::LAYER_BLEND.into()),
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
        let layer_blend_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Layer Blend Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&layer_blend_config_bind_group_layout),
                    Some(&layer_blend_texture_bind_group_layout),
                    Some(&layer_blend_texture_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let filter_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Filter Pipeline"),
            layout: Some(&filter_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &filter_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<FilterInstanceData>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Uint32x2,
                        1 => Uint32x2,
                        2 => Uint32x2,
                        3 => Uint32x2,
                        4 => Uint32x2,
                        5 => Uint32,
                        6 => Uint32x2,
                        7 => Uint32x2,
                        8 => Uint32,
                    ],
                }],
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
            cache: None,
            multiview_mask: None,
        });
        let layer_blend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Layer Blend Pipeline"),
            layout: Some(&layer_blend_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &layer_blend_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &layer_blend_shader,
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
            cache: None,
            multiview_mask: None,
        });

        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        const INITIAL_ALPHA_TEXTURE_HEIGHT: u32 = 1;
        let alphas_texture = Self::create_alphas_texture(
            device,
            max_texture_dimension_2d,
            INITIAL_ALPHA_TEXTURE_HEIGHT,
        );
        let view_config_buffer = Self::create_config_buffer(
            device,
            &RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
            max_texture_dimension_2d,
        );

        let AtlasConfig {
            atlas_size: (atlas_width, atlas_height),
            initial_atlas_count,
            ..
        } = image_cache.atlas_manager().config();
        let (atlas_texture_array, atlas_texture_array_view) = Self::create_atlas_texture_array(
            device,
            *atlas_width,
            *atlas_height,
            *initial_atlas_count as u32,
        );
        let placeholder_external_texture_view = Self::create_placeholder_external_texture(device);
        let atlas_bind_group = Self::create_paint_source_bind_group(
            device,
            &atlas_bind_group_layout,
            &atlas_texture_array_view,
            &placeholder_external_texture_view,
        );

        // Create a 1x1 stub atlas texture array for use during render_to_atlas.
        // This avoids the read-write conflict that occurs when the real atlas is both
        // a shader input (bind group) and render target in the same pass.
        let (_stub_atlas_texture, stub_atlas_texture_array_view) =
            Self::create_atlas_texture_array(device, 1, 1, 1);
        let stub_atlas_bind_group = Self::create_paint_source_bind_group(
            device,
            &atlas_bind_group_layout,
            &stub_atlas_texture_array_view,
            &placeholder_external_texture_view,
        );

        const INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT: u32 = 1;
        let encoded_paints_data = vec![
            0;
            ((max_texture_dimension_2d * INITIAL_ENCODED_PAINTS_TEXTURE_HEIGHT) << 4)
                as usize
        ];
        let encoded_paints_texture = Self::create_encoded_paints_texture(
            device,
            max_texture_dimension_2d,
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
            max_texture_dimension_2d,
            INITIAL_GRADIENT_TEXTURE_HEIGHT,
        );
        let gradient_bind_group = Self::create_gradient_bind_group(
            device,
            &gradient_bind_group_layout,
            &gradient_texture.create_view(&TextureViewDescriptor::default()),
        );

        let AtlasConfig {
            atlas_size: (filter_atlas_width, filter_atlas_height),
            ..
        } = filter_texture_cache.atlas_manager().config();
        let filter_atlas_size = (*filter_atlas_width, *filter_atlas_height);

        // TODO: We really should deduplicate handling of this this with encoded paints texture.
        const INITIAL_FILTER_TEXTURE_HEIGHT: u32 = 1;
        let filter_data =
            vec![0_u8; ((max_texture_dimension_2d * INITIAL_FILTER_TEXTURE_HEIGHT) << 4) as usize];
        let filter_data_texture = Self::create_filter_data_texture(
            device,
            max_texture_dimension_2d,
            INITIAL_FILTER_TEXTURE_HEIGHT,
        );
        let filter_base_bind_group = Self::create_filter_base_bind_group(
            device,
            &filter_bind_group_layout,
            &filter_data_texture.create_view(&TextureViewDescriptor::default()),
        );
        let layer_filter_data_texture =
            Self::create_filter_data_texture(device, GpuFilterData::SIZE_TEXELS, 1);
        let layer_filter_base_bind_group = Self::create_filter_base_bind_group(
            device,
            &filter_bind_group_layout,
            &layer_filter_data_texture.create_view(&TextureViewDescriptor::default()),
        );
        let layer_filter_data_texture_height = 1;

        let filter_atlas = FilterAtlasState::new(device, filter_atlas_size);

        let strip_bind_group = Self::create_strip_bind_group(
            device,
            &strip_bind_group_layout,
            &alphas_texture.create_view(&TextureViewDescriptor::default()),
            &view_config_buffer,
        );

        let filter_instance_buffer_capacity = size_of::<FilterInstanceData>() as u64;
        let resources = GpuResources {
            strips_buffer: Self::create_strips_buffer(device, 0),
            filter_instance_buffer: Self::create_filter_instance_buffer(
                device,
                filter_instance_buffer_capacity,
            ),
            filter_instance_buffer_capacity,
            strip_bind_group,
            alphas_texture,
            atlas_texture_array,
            atlas_texture_array_view,
            atlas_bind_group,
            placeholder_external_texture_view,
            filter_atlas,
            stub_atlas_bind_group,
            stub_atlas_texture_array_view,
            encoded_paints_texture,
            encoded_paints_bind_group,
            gradient_texture,
            gradient_bind_group,
            filter_data_texture,
            filter_base_bind_group,
            layer_filter_data_texture,
            layer_filter_data_texture_height,
            layer_filter_base_bind_group,
            view_config_buffer,
        };

        let depth_texture = Self::create_depth_texture(
            device,
            render_target_config.width,
            render_target_config.height,
        );
        let depth_texture_view = depth_texture.create_view(&TextureViewDescriptor::default());

        Self {
            intermediate_strip_pipelines,
            alpha_strip_pipelines,
            opaque_strip_pipelines,
            depth_texture,
            depth_texture_view,
            depth_cleared_this_frame: false,
            strip_bind_group_layout,
            encoded_paints_bind_group_layout,
            gradient_bind_group_layout,
            atlas_bind_group_layout,
            filter_bind_group_layout,
            filter_pipeline,
            layer_blend_pipeline,
            filter_input_bind_group_layouts,
            layer_blend_config_bind_group_layout,
            layer_blend_texture_bind_group_layout,
            resources,
            encoded_paints_data,
            filter_data,
            view_config_cache: Vec::new(),
            render_size: RenderSize {
                width: render_target_config.width,
                height: render_target_config.height,
            },
        }
    }

    #[inline]
    fn create_depth_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    fn create_strips_buffer(device: &Device, required_strips_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Strips Buffer"),
            size: required_strips_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_filter_instance_buffer(device: &Device, required_size: u64) -> Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Filter Instance Buffer"),
            size: required_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn grow_buffer_capacity(required_size: u64) -> u64 {
        required_size.max(1).next_power_of_two()
    }

    fn ensure_filter_instance_buffer_capacity(&mut self, device: &Device, required_size: u64) {
        if required_size <= self.resources.filter_instance_buffer_capacity {
            return;
        }

        let capacity = Self::grow_buffer_capacity(required_size);
        self.resources.filter_instance_buffer =
            Self::create_filter_instance_buffer(device, capacity);
        self.resources.filter_instance_buffer_capacity = capacity;
    }

    fn ensure_layer_filter_data_texture_height(&mut self, device: &Device, required_height: u32) {
        if required_height <= self.resources.layer_filter_data_texture_height {
            return;
        }

        let height = required_height.next_power_of_two();
        self.resources.layer_filter_data_texture =
            Self::create_filter_data_texture(device, GpuFilterData::SIZE_TEXELS, height);
        self.resources.layer_filter_base_bind_group = Self::create_filter_base_bind_group(
            device,
            &self.filter_bind_group_layout,
            &self
                .resources
                .layer_filter_data_texture
                .create_view(&TextureViewDescriptor::default()),
        );
        self.resources.layer_filter_data_texture_height = height;
    }

    fn upload_layer_filter_data(
        &mut self,
        device: &Device,
        queue: &Queue,
        texel_offset: u32,
        gpu_filter: GpuFilterData,
    ) {
        let row = texel_offset / GpuFilterData::SIZE_TEXELS;
        self.ensure_layer_filter_data_texture_height(device, row + 1);
        let data = [gpu_filter];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.resources.layer_filter_data_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: row, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(GpuFilterData::SIZE_TEXELS * 16),
                rows_per_image: Some(1),
            },
            Extent3d {
                width: GpuFilterData::SIZE_TEXELS,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    fn create_config_buffer(
        device: &Device,
        render_size: &RenderSize,
        alpha_texture_width: u32,
    ) -> Buffer {
        Self::create_config_buffer_with_origin(
            device,
            render_size,
            alpha_texture_width,
            RenderOrigin::default(),
        )
    }

    fn create_config_buffer_with_origin(
        device: &Device,
        render_size: &RenderSize,
        alpha_texture_width: u32,
        origin: RenderOrigin,
    ) -> Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&Config {
                width: render_size.width,
                height: render_size.height,
                strip_height: Tile::HEIGHT.into(),
                alphas_tex_width_bits: alpha_texture_width.trailing_zeros(),
                encoded_paints_tex_width_bits: alpha_texture_width.trailing_zeros(),
                strip_offset_x: -i32::from(origin.x),
                strip_offset_y: -i32::from(origin.y),
                negate_ndc: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn create_alphas_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Alpha Texture"),
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

    fn create_atlas_texture_array(
        device: &Device,
        width: u32,
        height: u32,
        atlas_count: u32,
    ) -> (Texture, TextureView) {
        // See the comment in `Renderer::new_with`. On WASM, we need to set this to at
        // least 2 so it works with the wgpu WebGL backend.
        #[cfg(target_arch = "wasm32")]
        let depth_or_array_layers = atlas_count.max(2);
        #[cfg(not(target_arch = "wasm32"))]
        let depth_or_array_layers = atlas_count;

        let atlas_texture_array = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atlas Texture Array"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let atlas_texture_array_view = atlas_texture_array.create_view(&TextureViewDescriptor {
            label: Some("Atlas Texture Array View"),
            format: None,
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(atlas_count),
            usage: None,
        });

        (atlas_texture_array, atlas_texture_array_view)
    }

    fn create_filter_data_texture(device: &Device, width: u32, height: u32) -> Texture {
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

    fn create_filter_base_bind_group(
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

    fn create_placeholder_external_texture(device: &Device) -> TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Placeholder External Texture"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        texture.create_view(&TextureViewDescriptor::default())
    }

    fn create_paint_source_bind_group(
        device: &Device,
        atlas_bind_group_layout: &BindGroupLayout,
        atlas_texture_array_view: &TextureView,
        external_texture_view: &TextureView,
    ) -> BindGroup {
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(atlas_texture_array_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(external_texture_view),
            },
        ];
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Paint Source Bind Group"),
            layout: atlas_bind_group_layout,
            entries: &entries,
        })
    }

    fn create_encoded_paints_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Encoded Paints Texture"),
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

    fn create_encoded_paints_bind_group(
        device: &Device,
        encoded_paints_bind_group_layout: &BindGroupLayout,
        encoded_paints_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Encoded Paints Bind Group"),
            layout: encoded_paints_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(encoded_paints_texture_view),
            }],
        })
    }

    fn create_gradient_texture(device: &Device, width: u32, height: u32) -> Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gradient Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    fn create_gradient_bind_group(
        device: &Device,
        gradient_bind_group_layout: &BindGroupLayout,
        gradient_texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gradient Bind Group"),
            layout: gradient_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(gradient_texture_view),
            }],
        })
    }

    fn create_strip_bind_group(
        device: &Device,
        strip_bind_group_layout: &BindGroupLayout,
        alphas_texture_view: &TextureView,
        config_buffer: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Strip Bind Group"),
            layout: strip_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(alphas_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Prepare GPU buffers for rendering, given alphas.
    ///
    /// Specifically, updates the alpha texture with `alphas` and the config buffer when
    /// the rendering size changes.
    fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        gradient_cache: &mut GradientRampCache,
        encoded_paints: &[GpuEncodedPaint],
        alphas: &mut Vec<u8>,
        new_render_size: &RenderSize,
        origin: RenderOrigin,
        target: DirectTarget,
        paint_idxs: &[u32],
        filter_context: &FilterContext,
        upload_shared_resources: bool,
    ) {
        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        if upload_shared_resources {
            self.maybe_resize_alphas_tex(device, max_texture_dimension_2d, alphas.len());
            self.maybe_resize_encoded_paints_tex(device, max_texture_dimension_2d, paint_idxs);
            self.maybe_resize_filter_tex(device, max_texture_dimension_2d, filter_context);
        }
        self.maybe_update_config_buffer(
            device,
            max_texture_dimension_2d,
            new_render_size,
            origin,
            target,
        );

        if upload_shared_resources {
            self.upload_alpha_texture(queue, alphas);
            self.upload_encoded_paints_texture(queue, encoded_paints);
            self.upload_filter_texture(queue, filter_context);
        }

        if upload_shared_resources && gradient_cache.has_changed() {
            self.maybe_resize_gradient_tex(device, max_texture_dimension_2d, gradient_cache);
            self.upload_gradient_texture(queue, gradient_cache);
            gradient_cache.mark_synced();
        }
    }

    fn maybe_resize_filter_tex(
        &mut self,
        device: &Device,
        max_texture_dimension_2d: u32,
        filter_context: &FilterContext,
    ) {
        let Some(required_filter_height) =
            filter_context.required_filter_data_height(max_texture_dimension_2d)
        else {
            return;
        };
        debug_assert!(
            self.resources.filter_data_texture.width() == max_texture_dimension_2d,
            "Filter texture width must match max texture dimensions"
        );
        let current_filter_height = self.resources.filter_data_texture.height();
        if required_filter_height > current_filter_height {
            let required_filter_size = (max_texture_dimension_2d * required_filter_height) << 4;
            self.filter_data.resize(required_filter_size as usize, 0);

            let filter_texture = Self::create_filter_data_texture(
                device,
                max_texture_dimension_2d,
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

    /// Update the alpha texture size if needed.
    fn maybe_resize_alphas_tex(
        &mut self,
        device: &Device,
        max_texture_dimension_2d: u32,
        alphas_len: usize,
    ) {
        let required_alpha_height = u32::try_from(alphas_len)
            .unwrap()
            // There are 16 1-byte alpha values per texel.
            .div_ceil(max_texture_dimension_2d << 4);
        debug_assert!(
            self.resources.alphas_texture.width() == max_texture_dimension_2d,
            "Alpha texture width must match max texture dimensions"
        );
        let current_alpha_height = self.resources.alphas_texture.height();
        if required_alpha_height > current_alpha_height {
            // We need to resize the alpha texture to fit the new alpha data.
            assert!(
                required_alpha_height <= max_texture_dimension_2d,
                "Alpha texture height exceeds max texture dimensions"
            );

            // The alpha texture encodes 16 1-byte alpha values per texel, with 4 alpha values packed in each channel
            let alphas_texture = Self::create_alphas_texture(
                device,
                max_texture_dimension_2d,
                required_alpha_height,
            );
            self.resources.alphas_texture = alphas_texture;
            self.view_config_cache.clear();

            // Since the alpha texture has changed, update the strip bind group.
            self.resources.strip_bind_group = Self::create_strip_bind_group(
                device,
                &self.strip_bind_group_layout,
                &self
                    .resources
                    .alphas_texture
                    .create_view(&TextureViewDescriptor::default()),
                &self.resources.view_config_buffer,
            );
        }
    }

    /// Create encoded paint metadata for the current pass.
    ///
    /// WGPU command buffers only execute after all scheduled layer passes have been recorded.
    /// Layer compositing appends temporary external-texture paints at reused indices, so rewriting a
    /// single shared encoded-paints texture would make earlier passes observe later paint metadata.
    /// Use a fresh texture/bind group per prepare call, just like the per-pass view config buffer.
    fn maybe_resize_encoded_paints_tex(
        &mut self,
        device: &Device,
        max_texture_dimension_2d: u32,
        paint_idxs: &[u32],
    ) {
        let required_texels = paint_idxs.last().unwrap();
        let required_encoded_paints_height =
            required_texels.div_ceil(max_texture_dimension_2d).max(1);
        assert!(
            required_encoded_paints_height <= max_texture_dimension_2d,
            "Encoded paints texture height exceeds max texture dimensions"
        );
        let required_encoded_paints_size =
            (max_texture_dimension_2d * required_encoded_paints_height) << 4;
        self.encoded_paints_data
            .resize(required_encoded_paints_size as usize, 0);

        self.resources.encoded_paints_texture = Self::create_encoded_paints_texture(
            device,
            max_texture_dimension_2d,
            required_encoded_paints_height,
        );
        self.resources.encoded_paints_bind_group = Self::create_encoded_paints_bind_group(
            device,
            &self.encoded_paints_bind_group_layout,
            &self
                .resources
                .encoded_paints_texture
                .create_view(&TextureViewDescriptor::default()),
        );
    }

    /// Update the gradient texture size if needed.
    fn maybe_resize_gradient_tex(
        &mut self,
        device: &Device,
        max_texture_dimension_2d: u32,
        gradient_cache: &GradientRampCache,
    ) {
        let gradient_pixels = (gradient_cache.luts_size() / 4) as u32; // 4 bytes per RGBA8 pixel
        let required_gradient_height = gradient_pixels.div_ceil(max_texture_dimension_2d);
        debug_assert!(
            self.resources.gradient_texture.width() == max_texture_dimension_2d,
            "Gradient texture width must match max texture dimensions"
        );
        let current_gradient_height = self.resources.gradient_texture.height();
        if required_gradient_height > current_gradient_height {
            assert!(
                required_gradient_height <= max_texture_dimension_2d,
                "Gradient texture height exceeds max texture dimensions"
            );
            let gradient_texture = Self::create_gradient_texture(
                device,
                max_texture_dimension_2d,
                required_gradient_height,
            );
            self.resources.gradient_texture = gradient_texture;

            // Since the gradient texture has changed, we need to update the gradient bind group.
            self.resources.gradient_bind_group = Self::create_gradient_bind_group(
                device,
                &self.gradient_bind_group_layout,
                &self
                    .resources
                    .gradient_texture
                    .create_view(&TextureViewDescriptor::default()),
            );
        }
    }

    /// Select a stable config buffer for this pass.
    ///
    /// Multiple render passes are recorded before the queue is submitted, so each distinct
    /// target size/origin needs its own uniform buffer rather than rewriting one shared buffer.
    fn maybe_update_config_buffer(
        &mut self,
        device: &Device,
        max_texture_dimension_2d: u32,
        new_render_size: &RenderSize,
        origin: RenderOrigin,
        target: DirectTarget,
    ) {
        let key = ViewConfigKey {
            render_size: new_render_size.clone(),
            origin,
        };
        if let Some(entry) = self.view_config_cache.iter().find(|entry| entry.key == key) {
            self.resources.view_config_buffer = entry.buffer.clone();
            self.resources.strip_bind_group = entry.bind_group.clone();
        } else {
            let buffer = Self::create_config_buffer_with_origin(
                device,
                new_render_size,
                max_texture_dimension_2d,
                origin,
            );
            let bind_group = Self::create_strip_bind_group(
                device,
                &self.strip_bind_group_layout,
                &self
                    .resources
                    .alphas_texture
                    .create_view(&TextureViewDescriptor::default()),
                &buffer,
            );
            self.view_config_cache.push(ViewConfigCacheEntry {
                key,
                buffer: buffer.clone(),
                bind_group: bind_group.clone(),
            });
            self.resources.view_config_buffer = buffer;
            self.resources.strip_bind_group = bind_group;
        }

        if target == DirectTarget::UserSurface && self.render_size != *new_render_size {
            self.depth_texture =
                Self::create_depth_texture(device, new_render_size.width, new_render_size.height);
            self.depth_texture_view = self
                .depth_texture
                .create_view(&TextureViewDescriptor::default());

            self.render_size = new_render_size.clone();
        }
    }

    /// Resize the texture array to accommodate more atlases.
    fn maybe_resize_atlas_texture_array(
        device: &Device,
        encoder: &mut CommandEncoder,
        resources: &mut GpuResources,
        atlas_bind_group_layout: &BindGroupLayout,
        required_atlas_count: u32,
    ) {
        let Extent3d {
            width,
            height,
            depth_or_array_layers: current_atlas_count,
        } = resources.atlas_texture_array.size();
        if required_atlas_count > current_atlas_count {
            // Create new texture array with more layers
            let (new_atlas_texture_array, new_atlas_texture_array_view) =
                Self::create_atlas_texture_array(device, width, height, required_atlas_count);

            // Copy existing atlas data from old texture array to new one
            Self::copy_atlas_texture_data(
                encoder,
                &resources.atlas_texture_array,
                &new_atlas_texture_array,
                current_atlas_count,
                width,
                height,
            );

            // Update the bind group with the new texture array view
            let new_atlas_bind_group = Self::create_paint_source_bind_group(
                device,
                atlas_bind_group_layout,
                &new_atlas_texture_array_view,
                &resources.placeholder_external_texture_view,
            );

            // Replace the old resources
            resources.atlas_texture_array = new_atlas_texture_array;
            resources.atlas_texture_array_view = new_atlas_texture_array_view;
            resources.atlas_bind_group = new_atlas_bind_group;
        }
    }

    fn create_external_paint_source_bind_group(
        &self,
        device: &Device,
        external_texture_view: &TextureView,
    ) -> BindGroup {
        Self::create_paint_source_bind_group(
            device,
            &self.atlas_bind_group_layout,
            &self.resources.atlas_texture_array_view,
            external_texture_view,
        )
    }

    fn create_layer_blend_texture_bind_group(
        &self,
        device: &Device,
        texture_view: &TextureView,
    ) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Layer Blend Texture Bind Group"),
            layout: &self.layer_blend_texture_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(texture_view),
            }],
        })
    }

    /// Copy texture data from the old atlas texture array to a new one.
    /// This is necessary when resizing the texture array to preserve existing atlas data.
    fn copy_atlas_texture_data(
        encoder: &mut CommandEncoder,
        old_atlas_texture_array: &Texture,
        new_atlas_texture_array: &Texture,
        layer_count_to_copy: u32,
        width: u32,
        height: u32,
    ) {
        // Copy all layers from old texture array to new texture array
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: old_atlas_texture_array,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: new_atlas_texture_array,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: layer_count_to_copy,
            },
        );
    }

    /// Upload alpha data to the texture.
    fn upload_alpha_texture(&mut self, queue: &Queue, alphas: &mut Vec<u8>) {
        if alphas.is_empty() {
            return;
        }

        let texture_width = self.resources.alphas_texture.width();
        let texture_height = self.resources.alphas_texture.height();
        let total_size = texture_width as usize * texture_height as usize * 16;

        let original_len = alphas.len();

        // Temporarily pad the length of the alphas to the texture size before uploading.
        alphas.resize(total_size, 0);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.resources.alphas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            alphas,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 16 bytes per RGBA32Uint texel (4 u32s × 4 bytes each), which is equivalent to
                // a bit shift of 4.
                bytes_per_row: Some(texture_width << 4),
                rows_per_image: Some(texture_height),
            },
            Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
        );

        // Truncate back to the original size.
        alphas.truncate(original_len);
    }

    /// Upload encoded paints to the texture.
    fn upload_encoded_paints_texture(&mut self, queue: &Queue, encoded_paints: &[GpuEncodedPaint]) {
        let encoded_paints_texture = &self.resources.encoded_paints_texture;
        let encoded_paints_texture_width = encoded_paints_texture.width();
        let encoded_paints_texture_height = encoded_paints_texture.height();

        GpuEncodedPaint::serialize_to_buffer(encoded_paints, &mut self.encoded_paints_data);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: encoded_paints_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.encoded_paints_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 16 bytes per RGBA32Uint texel (4 u32s × 4 bytes each), equivalent to bit shift of 4
                bytes_per_row: Some(encoded_paints_texture_width << 4),
                rows_per_image: Some(encoded_paints_texture_height),
            },
            Extent3d {
                width: encoded_paints_texture_width,
                height: encoded_paints_texture_height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn upload_filter_texture(&mut self, queue: &Queue, filter_context: &FilterContext) {
        if filter_context.is_empty() {
            return;
        }

        let filter_texture = &self.resources.filter_data_texture;
        let width = filter_texture.width();
        let height = filter_texture.height();

        filter_context.serialize_to_buffer(&mut self.filter_data);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: filter_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.filter_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width << 4),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload gradient data to the texture.
    fn upload_gradient_texture(&mut self, queue: &Queue, gradient_cache: &mut GradientRampCache) {
        let gradient_texture = &self.resources.gradient_texture;
        let gradient_texture_width = gradient_texture.width();
        let gradient_texture_height = gradient_texture.height();

        // Upload the gradient LUT data
        if !gradient_cache.is_empty() {
            let total_capacity = (gradient_texture_width * gradient_texture_height * 4) as usize;

            // Take ownership of the luts to avoid copying, then resize for texture padding
            let mut luts = gradient_cache.take_luts();
            let old_luts_len = luts.len();
            luts.resize(total_capacity, 0);

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: gradient_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &luts,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // 4 bytes per RGBA8 pixel
                    bytes_per_row: Some(gradient_texture_width << 2),
                    rows_per_image: Some(gradient_texture_height),
                },
                Extent3d {
                    width: gradient_texture_width,
                    height: gradient_texture_height,
                    depth_or_array_layers: 1,
                },
            );

            // Restore the luts back to the cache
            luts.truncate(old_luts_len);
            gradient_cache.restore_luts(luts);
        }
    }

    /// Uploads two strip slices (opaque then alpha) into a single GPU buffer.
    fn upload_strip_pair(
        &mut self,
        device: &Device,
        queue: &Queue,
        opaque_strips: &[GpuStrip],
        alpha_strips: &[GpuStrip],
    ) {
        let opaque_bytes = size_of_val(opaque_strips) as u64;
        let alpha_bytes = size_of_val(alpha_strips) as u64;
        let total = opaque_bytes + alpha_bytes;
        self.resources.strips_buffer = Self::create_strips_buffer(device, total);
        // TODO: Consider using a staging belt to avoid an extra staging buffer allocation.
        let mut buffer_view = queue
            .write_buffer_with(&self.resources.strips_buffer, 0, total.try_into().unwrap())
            .expect("Capacity handled in creation");
        buffer_view
            .slice(..opaque_bytes as usize)
            .copy_from_slice(bytemuck::cast_slice(opaque_strips));
        buffer_view
            .slice(opaque_bytes as usize..)
            .copy_from_slice(bytemuck::cast_slice(alpha_strips));
    }
}

/// A struct containing references to the many objects needed to get work
/// scheduled onto the GPU.
struct RendererContext<'a> {
    programs: &'a mut Programs,
    device: &'a Device,
    queue: &'a Queue,
    encoder: &'a mut CommandEncoder,
    view: &'a TextureView,
    image_cache: &'a ImageCache,
    filter_context: &'a FilterContext,
    filter_pass_state: &'a mut FilterPassState,
    texture_bindings: &'a TextureBindings,
    external_paint_source_bind_groups: HashMap<TextureId, BindGroup>,
    render_pass_count: &'a mut usize,
}

impl RendererContext<'_> {
    fn external_paint_source_bind_group_for_texture(
        &mut self,
        texture_id: TextureId,
    ) -> &BindGroup {
        match self.external_paint_source_bind_groups.entry(texture_id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let texture_view = self
                    .texture_bindings
                    .get(texture_id)
                    .expect("external texture bindings were validated during paint preparation");
                let bind_group = self
                    .programs
                    .create_external_paint_source_bind_group(self.device, texture_view);
                entry.insert(bind_group)
            }
        }
    }

    /// Render the strips to the specified render target.
    fn do_strip_render_pass(
        &mut self,
        opaque_strips: &[GpuStrip],
        alpha_strips: &[GpuStrip],
        external_texture_runs: &[ExternalTextureRun],
        target: DirectTarget,
        load: wgpu::LoadOp<wgpu::Color>,
    ) {
        if opaque_strips.is_empty() && alpha_strips.is_empty() {
            return;
        }
        // TODO: We currently allocate a new strips buffer for each render pass. A more efficient
        // approach would be to re-use buffers or slices of a larger buffer.
        self.programs
            .upload_strip_pair(self.device, self.queue, opaque_strips, alpha_strips);
        let opaque_count = opaque_strips.len() as u32;
        let alpha_count = alpha_strips.len() as u32;

        // Create bind groups for all external textures passed in by the user that are used this
        // pass.
        for run in external_texture_runs {
            self.external_paint_source_bind_group_for_texture(run.texture_id);
        }

        let view = self.view;
        let bind_group = &self.programs.resources.strip_bind_group;
        let scissor_rect: Option<[u32; 4]> = None;

        let pipeline_idx = if target == DirectTarget::AtlasLayer {
            1
        } else {
            0
        };

        let is_final_view = target == DirectTarget::UserSurface;

        let depth_stencil_attachment = if is_final_view {
            let depth_load = if self.programs.depth_cleared_this_frame {
                wgpu::LoadOp::Load
            } else {
                self.programs.depth_cleared_this_frame = true;
                wgpu::LoadOp::Clear(1.0)
            };
            Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.programs.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: depth_load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            })
        } else {
            None
        };

        *self.render_pass_count += 1;
        let mut render_pass = self.encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render to Texture Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        if let Some([x, y, width, height]) = scissor_rect {
            render_pass.set_scissor_rect(x, y, width, height);
        }

        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_bind_group(2, &self.programs.resources.encoded_paints_bind_group, &[]);
        render_pass.set_bind_group(3, &self.programs.resources.gradient_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.programs.resources.strips_buffer.slice(..));

        if opaque_count > 0 {
            // Opaque pass
            debug_assert!(
                is_final_view,
                "Direct strip rendering only allows the final view to have opaque strips"
            );
            render_pass.set_pipeline(&self.programs.opaque_strip_pipelines[pipeline_idx]);
            render_pass.set_bind_group(1, &self.programs.resources.atlas_bind_group, &[]);
            render_pass.draw(0..4, 0..opaque_count);
        }

        if alpha_count > 0 {
            // Alpha pass
            if is_final_view {
                render_pass.set_pipeline(&self.programs.alpha_strip_pipelines[pipeline_idx]);
            } else {
                render_pass.set_pipeline(&self.programs.intermediate_strip_pipelines[pipeline_idx]);
            }

            let alpha_start = opaque_count;
            if external_texture_runs.is_empty() {
                render_pass.set_bind_group(1, &self.programs.resources.atlas_bind_group, &[]);
                render_pass.draw(0..4, alpha_start..alpha_start + alpha_count);
            } else {
                // Each run is drawn with a different external texture binding. Runs go from
                // `run.strips_start` to the next run's `strips_start`; gaps are ordinary atlas
                // draws.
                let mut current = 0;
                for (i, run) in external_texture_runs.iter().enumerate() {
                    let start = u32::try_from(run.strips_start).unwrap();
                    if start > current {
                        render_pass.set_bind_group(
                            1,
                            &self.programs.resources.atlas_bind_group,
                            &[],
                        );
                        render_pass.draw(0..4, alpha_start + current..alpha_start + start);
                    }

                    let end = external_texture_runs
                        .get(i + 1)
                        .map_or(alpha_count, |next| {
                            u32::try_from(next.strips_start).unwrap()
                        });
                    let paint_source_bind_group = self
                        .external_paint_source_bind_groups
                        .get(&run.texture_id)
                        .unwrap();
                    render_pass.set_bind_group(1, paint_source_bind_group, &[]);
                    render_pass.draw(0..4, alpha_start + start..alpha_start + end);
                    current = end;
                }

                if current < alpha_count {
                    render_pass.set_bind_group(1, &self.programs.resources.atlas_bind_group, &[]);
                    render_pass.draw(0..4, alpha_start + current..alpha_start + alpha_count);
                }
            }
        }
    }
}

impl RendererContext<'_> {
    #[allow(
        dead_code,
        reason = "Filter passes will be reattached once layer roots exist."
    )]
    fn apply_filter(&mut self, layer_id: LayerId) {
        let filter_atlas = &self.programs.resources.filter_atlas;
        self.filter_context.build_filter_passes(
            self.filter_pass_state,
            &layer_id,
            self.image_cache,
            |atlas_idx| {
                let size = filter_atlas.textures[atlas_idx as usize].size();
                [size.width, size.height]
            },
            || {
                let size = self.programs.resources.atlas_texture_array.size();
                [size.width, size.height]
            },
        );

        let filter_passes = self.filter_pass_state.filter_passes();
        if filter_passes.is_empty() {
            return;
        }

        let instances = self.filter_pass_state.instances();
        let instance_stride = size_of::<FilterInstanceData>() as u64;
        let total_size = instances.len() as u64 * instance_stride;
        // TODO: Reuse buffer (https://github.com/linebender/vello/pull/1494#discussion_r2937890819)
        self.programs.resources.filter_instance_buffer =
            Programs::create_filter_instance_buffer(self.device, total_size);
        self.queue.write_buffer(
            &self.programs.resources.filter_instance_buffer,
            0,
            bytemuck::cast_slice(instances),
        );

        let programs = &self.programs;
        let encoder = &mut self.encoder;
        let filter_atlas = &programs.resources.filter_atlas;
        for (i, pass) in filter_passes.iter().enumerate() {
            let input_bg = &filter_atlas.input_bind_groups[pass.input_atlas_idx as usize];
            // If this is `None`, it's unused, so we can just pass anything here.
            let original_idx = pass.original_atlas_idx.unwrap_or(pass.input_atlas_idx) as usize;
            let original_bg = &filter_atlas.original_bind_groups[original_idx];

            let (output_view, target_width, target_height) = match &pass.output {
                FilterPassTarget::FilterAtlas(idx) => {
                    let size = filter_atlas.textures[*idx as usize].size();
                    (&filter_atlas.views[*idx as usize], size.width, size.height)
                }
                FilterPassTarget::MainAtlas(idx) => {
                    let size = programs.resources.atlas_texture_array.size();
                    (
                        &create_atlas_layer_view(&programs.resources.atlas_texture_array, *idx),
                        size.width,
                        size.height,
                    )
                }
            };

            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Apply Filter Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            let instance = &instances[i];
            let [x, y, width, height] = instance.scissor_rect([target_width, target_height]);
            render_pass.set_scissor_rect(x, y, width, height);
            render_pass.set_pipeline(&programs.filter_pipeline);
            render_pass.set_bind_group(0, &programs.resources.filter_base_bind_group, &[]);
            render_pass.set_bind_group(1, input_bg, &[]);
            render_pass.set_bind_group(2, original_bg, &[]);
            render_pass.set_vertex_buffer(
                0,
                programs
                    .resources
                    .filter_instance_buffer
                    .slice((i as u64 * instance_stride)..((i as u64 + 1) * instance_stride)),
            );
            render_pass.draw(0..4, 0..1);
        }
    }
}

fn create_filter_input_bind_group(
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

fn create_atlas_layer_view(atlas: &Texture, layer: u32) -> TextureView {
    atlas.create_view(&TextureViewDescriptor {
        label: Some("Atlas Layer View"),
        format: None,
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: layer,
        array_layer_count: Some(1),
        usage: None,
    })
}

/// Trait for types that can write image data directly to the atlas texture.
///
/// This allows efficient uploading from different sources:
/// - `Pixmap`: Direct upload without intermediate texture
/// - `Texture`: Texture-to-texture copy
/// - Custom implementations for other image sources
pub trait AtlasWriter {
    /// Get the width of the image.
    fn width(&self) -> u32;
    /// Get the height of the image.
    fn height(&self) -> u32;

    /// Write image data to a specific layer of an atlas texture array at the specified offset.
    fn write_to_atlas_layer(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    );
}

/// Implementation for `wgpu::Texture` - uses texture-to-texture copy
impl AtlasWriter for Texture {
    fn width(&self) -> u32 {
        self.width()
    }

    fn height(&self) -> u32 {
        self.height()
    }

    fn write_to_atlas_layer(
        &self,
        _device: &Device,
        _queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: self,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: offset[0],
                    y: offset[1],
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Implementation for `Pixmap` - direct upload to atlas
impl AtlasWriter for Pixmap {
    fn width(&self) -> u32 {
        self.width() as u32
    }

    fn height(&self) -> u32 {
        self.height() as u32
    }

    fn write_to_atlas_layer(
        &self,
        _device: &Device,
        queue: &Queue,
        _encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: offset[0],
                    y: offset[1],
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            self.data_as_u8_slice(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Implementation for `Arc<Pixmap>`
impl AtlasWriter for Arc<Pixmap> {
    fn width(&self) -> u32 {
        self.as_ref().width() as u32
    }

    fn height(&self) -> u32 {
        self.as_ref().height() as u32
    }

    fn write_to_atlas_layer(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        atlas_texture: &Texture,
        layer: u32,
        offset: [u32; 2],
        width: u32,
        height: u32,
    ) {
        self.as_ref().write_to_atlas_layer(
            device,
            queue,
            encoder,
            atlas_texture,
            layer,
            offset,
            width,
            height,
        );
    }
}
