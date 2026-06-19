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
pub(crate) enum StripPassRenderTarget {
    /// Render to the root output target.
    Root(RootRenderTarget),
    /// Render to an allocated region in one of the layer atlas textures.
    Layer(LayerTextureRegion),
    /// Render to a whole layer atlas texture.
    LayerAtlas(usize),
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
    execute_schedule(renderer, &schedule);
    Ok(())
}

fn execute_schedule<R: RendererBackend>(renderer: &mut R, schedule: &Schedule) {
    let direct_root_opaque = collect_direct_root_opaque(schedule);
    if !direct_root_opaque.draw.is_empty() {
        renderer.render_strips(
            &direct_root_opaque.draw.opaque,
            &[],
            &[],
            StripPassRenderTarget::Root(RootRenderTarget::UserSurface),
            direct_root_opaque.load_op,
        );
    }

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

        execute_root_passes(renderer, root_passes);

        renderer.blend_layers(&round.blends);
        renderer.clear_layer_regions(&round.clear_layer_regions);
    }
}

#[derive(Debug)]
struct DirectRootOpaque {
    draw: Draw,
    load_op: LoadOp,
}

impl Default for DirectRootOpaque {
    fn default() -> Self {
        Self {
            draw: Draw::default(),
            load_op: LoadOp::Load,
        }
    }
}

fn collect_direct_root_opaque(schedule: &Schedule) -> DirectRootOpaque {
    let mut opaque = DirectRootOpaque::default();
    for pass in schedule.rounds.iter().flat_map(|round| &round.passes) {
        if !matches!(
            pass.target,
            RenderTarget::Root(RootRenderTarget::UserSurface)
        ) || pass.draw.opaque.is_empty()
        {
            continue;
        }

        if opaque.draw.is_empty() {
            opaque.load_op = pass.load_op;
        }
        opaque.draw.append_opaque(&pass.draw);
    }

    opaque
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

fn execute_root_passes<R: RendererBackend>(renderer: &mut R, root_passes: Vec<&round::RoundPass>) {
    let mut batch = RootBatch::default();
    for pass in root_passes {
        batch.push(pass, |target, draw| {
            renderer.render_strips(
                &draw.draw.opaque,
                &draw.draw.alpha,
                &draw.draw.external_texture_runs,
                StripPassRenderTarget::Root(target),
                draw.root_load_op,
            );
        });
    }
    batch.finish(|target, draw| {
        renderer.render_strips(
            &draw.draw.opaque,
            &draw.draw.alpha,
            &draw.draw.external_texture_runs,
            StripPassRenderTarget::Root(target),
            draw.root_load_op,
        );
    });
}

#[derive(Debug, Default)]
struct RootBatch {
    target: Option<RootRenderTarget>,
    draw: RootBatchDraw,
}

impl RootBatch {
    fn push(
        &mut self,
        pass: &round::RoundPass,
        mut flush: impl FnMut(RootRenderTarget, &RootBatchDraw),
    ) {
        let RenderTarget::Root(target) = pass.target else {
            return;
        };

        let mut draw = pass.draw.clone();
        if target == RootRenderTarget::UserSurface {
            draw.opaque.clear();
        }

        if draw.is_empty() {
            return;
        }

        if !is_batchable_root_draw(target, &draw) {
            self.finish(&mut flush);
            flush(
                target,
                &RootBatchDraw {
                    draw,
                    root_load_op: pass.load_op,
                },
            );
            return;
        }

        if !self.draw.is_empty() && (self.target != Some(target) || pass.load_op == LoadOp::Clear) {
            self.finish(&mut flush);
        }

        if self.draw.is_empty() {
            self.target = Some(target);
            self.draw.root_load_op = pass.load_op;
        }
        self.draw.draw.append(&draw);
    }

    fn finish(&mut self, mut flush: impl FnMut(RootRenderTarget, &RootBatchDraw)) {
        if let Some(target) = self.target.take()
            && !self.draw.is_empty()
        {
            flush(target, &self.draw);
        }
        self.draw = RootBatchDraw::default();
    }
}

#[derive(Debug)]
struct RootBatchDraw {
    draw: Draw,
    root_load_op: LoadOp,
}

impl Default for RootBatchDraw {
    fn default() -> Self {
        Self {
            draw: Draw::default(),
            root_load_op: LoadOp::Load,
        }
    }
}

impl RootBatchDraw {
    fn is_empty(&self) -> bool {
        self.draw.is_empty()
    }
}

fn is_batchable_root_draw(target: RootRenderTarget, draw: &Draw) -> bool {
    target == RootRenderTarget::UserSurface || draw.opaque.is_empty()
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

#[derive(Debug, Default, Clone)]
struct Draw {
    opaque: Vec<GpuStrip>,
    alpha: Vec<GpuStrip>,
    external_texture_runs: Vec<ExternalTextureRun>,
}

impl Draw {
    fn append_opaque(&mut self, other: &Self) {
        self.opaque.extend_from_slice(&other.opaque);
    }

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
