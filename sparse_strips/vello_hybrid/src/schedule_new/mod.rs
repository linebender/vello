// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! New scheduler implementation for `vello_hybrid`.
//!
//! This first slice supports root-level draws and regular property-less layers.

mod builder;
mod round;
mod timeline;

use self::builder::ScheduleBuilder;
pub(crate) use self::round::BlendOp;
use self::round::Schedule;
use crate::{GpuStrip, RenderError, Scene};
use alloc::vec::Vec;
use vello_common::TextureId;
use vello_common::encode::EncodedPaint;
use vello_common::geometry::RectU16;
use vello_common::render_graph::LayerId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootRenderTarget {
    /// The root render target is the user-provided surface.
    UserSurface,
    /// The root render target is the user-provided surface and samples layer texture 0.
    UserSurfaceFromLayer0,
    /// The root render target is an atlas layer.
    AtlasLayer,
    /// The root render target is an atlas layer and samples layer texture 0.
    AtlasLayerFromLayer0,
}

/// Specifies the target for a strip render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum StripPassRenderTarget {
    /// Render to the root output target.
    Root(RootRenderTarget),
    /// Render to an allocated region in one of the layer atlas textures.
    Layer(LayerTextureRegion),
    /// Render to a whole layer atlas texture.
    LayerAtlas(usize),
    /// Render to a layer in the filter atlas.
    FilterLayer(LayerId),
    /// Render to one of the slot textures used for clipping/blending.
    SlotTexture(u8),
}

/// A rectangular region in one of the layer atlas textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerTextureRegion {
    /// Layer texture index, currently `0` or `1`.
    pub(crate) texture_index: usize,
    /// X coordinate in the layer texture.
    pub(crate) x: u32,
    /// Y coordinate in the layer texture.
    pub(crate) y: u32,
    /// Width of the region.
    pub(crate) width: u32,
    /// Height of the region.
    pub(crate) height: u32,
    /// Bounds of this layer in viewport coordinates.
    pub(crate) scene_bbox: RectU16,
}

/// Specifies a run of strips inside a draw that can be drawn with the same external texture
/// binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExternalTextureRun {
    pub(crate) texture_id: TextureId,
    /// Start index of the strip range for this run. The end is implicitly the start of the next
    /// run, or, for the last run, the total number of strips.
    pub(crate) strips_start: usize,
}

pub(crate) trait RendererBackend {
    /// Return the dimensions of each layer atlas texture.
    fn layer_texture_size(&self) -> (u32, u32);

    /// Clear rectangular regions in layer atlas textures to transparent black.
    fn clear_layer_regions(&mut self, regions: &[LayerTextureRegion]);

    /// Execute a render pass for strips, split into opaque and alpha passes.
    fn render_strips(
        &mut self,
        opaque_strips: &[GpuStrip],
        alpha_strips: &[GpuStrip],
        external_texture_runs: &[ExternalTextureRun],
        target: StripPassRenderTarget,
        load_op: LoadOp,
    );

    /// Apply non-default blend layer operations.
    fn blend_layers(&mut self, blends: &[BlendOp]);
}

/// Backend agnostic enum that specifies the operation to perform to the output attachment at the
/// start of a render pass.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoadOp {
    Load,
    Clear,
}

/// Render the supported subset of a scene through the new recorder-based scheduler.
pub(crate) fn render_scene<R: RendererBackend>(
    renderer: &mut R,
    scene: &Scene,
    root_output_target: RootRenderTarget,
    paint_idxs: &[u32],
    encoded_paints: &[EncodedPaint],
) -> Result<(), RenderError> {
    let strip_storage = scene.strip_storage.borrow();
    let mut builder = ScheduleBuilder::new(
        scene,
        &strip_storage,
        root_output_target,
        paint_idxs,
        encoded_paints,
        renderer.layer_texture_size(),
    );
    let schedule = builder.build()?;
    print_schedule_debug_stats(scene, &schedule);
    execute_schedule(renderer, &schedule);
    Ok(())
}

fn print_schedule_debug_stats(scene: &Scene, schedule: &Schedule) {
    let non_empty_rounds = schedule
        .rounds
        .iter()
        .filter(|round| {
            !round.passes.is_empty()
                || !round.blends.is_empty()
                || !round.clear_layer_regions.is_empty()
        })
        .count();
    let total_passes = schedule
        .rounds
        .iter()
        .map(|round| strip_pass_count(round).total)
        .sum::<usize>();
    let total_blends = schedule
        .rounds
        .iter()
        .map(|round| round.blends.len())
        .sum::<usize>();
    let total_clear_passes = schedule.rounds.iter().map(clear_pass_count).sum::<usize>();
    let total_clear_rects = schedule
        .rounds
        .iter()
        .map(|round| round.clear_layer_regions.len())
        .sum::<usize>();

    eprintln!("vello_hybrid schedule debug begin");
    eprintln!(
        "vello_hybrid schedule summary: rounds={} non_empty_rounds={} passes={} blends={} clear_passes={} clear_rects={} scene={}x{} layers={} root_cmds={} root_child_layers={} root_is_blend_target={} layer_texture={}x{} allocation_attempts={} allocation_retries={} allocation_retry_events={}",
        schedule.rounds.len(),
        non_empty_rounds,
        total_passes,
        total_blends,
        total_clear_passes,
        total_clear_rects,
        scene.width,
        scene.height,
        schedule.debug.layer_count,
        schedule.debug.root_cmd_count,
        schedule.debug.root_child_layer_count,
        schedule.debug.root_is_blend_target,
        schedule.debug.layer_texture_size.0,
        schedule.debug.layer_texture_size.1,
        schedule.debug.allocation_attempts,
        schedule.debug.allocation_retries,
        schedule.debug.allocation_retry_events.len(),
    );

    for (depth, count) in &schedule.debug.depth_counts {
        eprintln!("vello_hybrid schedule depth: depth={depth} layers={count}");
    }

    for event in &schedule.debug.allocation_retry_events {
        eprintln!(
            "vello_hybrid schedule alloc-retry: texture={} earliest_round={} allocated_round={} attempts={} bbox={}x{} at {},{}",
            event.texture_index,
            event.earliest_round,
            event.allocated_round,
            event.attempts,
            event.bbox.width(),
            event.bbox.height(),
            event.bbox.x0,
            event.bbox.y0,
        );
    }

    for layer in &schedule.debug.scheduled_layers {
        eprintln!(
            "vello_hybrid schedule layer: id={} depth={} texture={} allocated_round={} ready_round={} atlas={},{} cmds={} batches={} child_layers={} bbox={}x{} at {},{} clip={} default_blend={} destructive_blend={} opacity={:.3}",
            layer.layer_id,
            layer.depth,
            layer.texture_index,
            layer.allocated_round,
            layer.ready_round,
            layer.atlas_x,
            layer.atlas_y,
            layer.command_count,
            layer.batch_count,
            layer.child_layer_count,
            layer.bbox.width(),
            layer.bbox.height(),
            layer.bbox.x0,
            layer.bbox.y0,
            layer.has_clip,
            layer.has_default_blend,
            layer.is_destructive_blend,
            layer.opacity,
        );
    }

    for (round_idx, round) in schedule.rounds.iter().enumerate() {
        if round.passes.is_empty()
            && round.blends.is_empty()
            && round.clear_layer_regions.is_empty()
        {
            continue;
        }

        let strip_passes = strip_pass_count(round);
        let mut opaque_strips = 0usize;
        let mut alpha_strips = 0usize;
        let mut external_texture_runs = 0usize;
        for pass in &round.passes {
            opaque_strips += pass.draw.opaque.len();
            alpha_strips += pass.draw.alpha.len();
            external_texture_runs += pass.draw.external_texture_runs.len();
        }

        let mut clear_regions = [0usize; 2];
        let mut clear_area = [0u64; 2];
        for region in &round.clear_layer_regions {
            if region.texture_index < 2 {
                clear_regions[region.texture_index] += 1;
                clear_area[region.texture_index] +=
                    u64::from(region.width) * u64::from(region.height);
            }
        }
        let clear_passes = clear_regions.iter().filter(|regions| **regions > 0).count();

        let mut blend_targets = [0usize; 2];
        for blend in &round.blends {
            if blend.parent.texture_index < 2 {
                blend_targets[blend.parent.texture_index] += 1;
            }
        }

        eprintln!(
            "vello_hybrid schedule round: idx={} passes={} root_passes={} layer0_passes={} layer1_passes={} opaque_strips={} alpha_strips={} external_texture_runs={} blends={} blend_to_layer0={} blend_to_layer1={} clear_passes={} clear_rects={} clear_rects_layer0={} clear_rects_layer1={} clear_area_layer0={} clear_area_layer1={}",
            round_idx,
            strip_passes.total,
            strip_passes.root,
            strip_passes.layer[0],
            strip_passes.layer[1],
            opaque_strips,
            alpha_strips,
            external_texture_runs,
            round.blends.len(),
            blend_targets[0],
            blend_targets[1],
            clear_passes,
            round.clear_layer_regions.len(),
            clear_regions[0],
            clear_regions[1],
            clear_area[0],
            clear_area[1],
        );
    }

    eprintln!("vello_hybrid schedule debug end");
}

#[derive(Debug, Default)]
struct StripPassCount {
    total: usize,
    root: usize,
    layer: [usize; 2],
}

fn strip_pass_count(round: &round::Round) -> StripPassCount {
    let mut count = StripPassCount::default();
    let mut has_batched_layer_pass = [false; 2];

    for pass in &round.passes {
        match pass.target {
            RenderTarget::Root(_) => {
                count.root += 1;
                count.total += 1;
            }
            RenderTarget::Layer(region)
                if region.texture_index < has_batched_layer_pass.len()
                    && pass.load_op == LoadOp::Load =>
            {
                has_batched_layer_pass[region.texture_index] = true;
            }
            RenderTarget::Layer(region) if region.texture_index < count.layer.len() => {
                count.layer[region.texture_index] += 1;
                count.total += 1;
            }
            RenderTarget::Layer(_) => {
                count.total += 1;
            }
        }
    }

    for (texture_index, has_pass) in has_batched_layer_pass.into_iter().enumerate() {
        if has_pass {
            count.layer[texture_index] += 1;
            count.total += 1;
        }
    }

    count
}

fn clear_pass_count(round: &round::Round) -> usize {
    let mut has_clear_regions = [false; 2];
    for region in &round.clear_layer_regions {
        if region.texture_index < has_clear_regions.len() && region.width > 0 && region.height > 0 {
            has_clear_regions[region.texture_index] = true;
        }
    }

    has_clear_regions
        .iter()
        .filter(|has_clear_regions| **has_clear_regions)
        .count()
}

fn execute_schedule<R: RendererBackend>(renderer: &mut R, schedule: &Schedule) {
    for round in &schedule.rounds {
        let mut layer_draws = [Draw::default(), Draw::default()];
        let mut layer_other_passes = [Vec::new(), Vec::new()];
        let mut root_passes = Vec::new();

        for pass in &round.passes {
            if let RenderTarget::Layer(region) = pass.target
                && region.texture_index < layer_draws.len()
                && pass.load_op == LoadOp::Load
            {
                layer_draws[region.texture_index].append(&pass.draw);
                continue;
            }

            match pass.target {
                RenderTarget::Root(_) => root_passes.push(pass),
                RenderTarget::Layer(region) if region.texture_index < layer_other_passes.len() => {
                    layer_other_passes[region.texture_index].push(pass);
                }
                RenderTarget::Layer(_) => {}
            }
        }

        for texture_index in [1, 0] {
            for pass in &layer_other_passes[texture_index] {
                execute_pass(renderer, pass);
            }

            let draw = &layer_draws[texture_index];
            renderer.render_strips(
                &draw.opaque,
                &draw.alpha,
                &draw.external_texture_runs,
                StripPassRenderTarget::LayerAtlas(texture_index),
                LoadOp::Load,
            );
        }

        for pass in root_passes {
            execute_pass(renderer, pass);
        }

        renderer.blend_layers(&round.blends);
        renderer.clear_layer_regions(&round.clear_layer_regions);
    }
}

fn execute_pass<R: RendererBackend>(renderer: &mut R, pass: &round::RoundPass) {
    renderer.render_strips(
        &pass.draw.opaque,
        &pass.draw.alpha,
        &pass.draw.external_texture_runs,
        pass.target.strip_pass_target(),
        pass.load_op,
    );
}

#[derive(Debug, Clone, Copy)]
pub(super) enum RenderTarget {
    Root(RootRenderTarget),
    Layer(LayerTextureRegion),
}

impl RenderTarget {
    fn strip_pass_target(self) -> StripPassRenderTarget {
        match self {
            Self::Root(root) => StripPassRenderTarget::Root(root),
            Self::Layer(layer) => StripPassRenderTarget::Layer(layer),
        }
    }

    fn allows_opaque_pass(self) -> bool {
        // TODO: Allow opaque strips for intermediate targets?
        matches!(self, Self::Root(RootRenderTarget::UserSurface))
    }

    fn texture_index(self) -> Option<usize> {
        match self {
            Self::Root(_) => None,
            Self::Layer(region) => Some(region.texture_index),
        }
    }

    fn layer_region(self) -> LayerTextureRegion {
        match self {
            Self::Layer(region) => region,
            Self::Root(_) => panic!("root targets do not have layer regions"),
        }
    }

    fn geometry_offset(self) -> (i32, i32) {
        match self {
            Self::Root(_) => (0, 0),
            Self::Layer(region) => (
                region.x as i32 - i32::from(region.scene_bbox.x0),
                region.y as i32 - i32::from(region.scene_bbox.y0),
            ),
        }
    }
}

#[derive(Debug, Default)]
struct Draw {
    opaque: Vec<GpuStrip>,
    alpha: Vec<GpuStrip>,
    external_texture_runs: Vec<ExternalTextureRun>,
}

impl Draw {
    fn append(&mut self, other: &Self) {
        self.opaque.extend_from_slice(&other.opaque);

        let alpha_offset = self.alpha.len();
        for run in &other.external_texture_runs {
            let strips_start = alpha_offset + run.strips_start;
            if strips_start == self.alpha.len()
                && self
                    .external_texture_runs
                    .last()
                    .is_some_and(|last| last.texture_id == run.texture_id)
            {
                continue;
            }
            self.external_texture_runs.push(ExternalTextureRun {
                texture_id: run.texture_id,
                strips_start,
            });
        }
        self.alpha.extend_from_slice(&other.alpha);
    }

    #[inline(always)]
    fn push_opaque(&mut self, gpu_strip: GpuStrip) {
        self.opaque.push(gpu_strip);
    }

    #[inline(always)]
    fn push_alpha(&mut self, gpu_strip: GpuStrip, external_texture_id: Option<TextureId>) {
        if let Some(texture_id) = external_texture_id {
            let needs_new_run = self
                .external_texture_runs
                .last()
                .is_none_or(|run| run.texture_id != texture_id);
            if needs_new_run {
                let strips_start = if self.external_texture_runs.is_empty() {
                    0
                } else {
                    self.alpha.len()
                };
                self.external_texture_runs.push(ExternalTextureRun {
                    strips_start,
                    texture_id,
                });
            }
        }

        self.alpha.push(gpu_strip);
    }

    fn is_empty(&self) -> bool {
        self.opaque.is_empty() && self.alpha.is_empty()
    }
}
