// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! New scheduler implementation for `vello_hybrid`.
//!
//! This first slice supports root-level draws and regular property-less layers.

mod alloc;
mod builder;
mod draw;
mod round;
mod timeline;

use self::builder::ScheduleBuilder;
use self::draw::Draw;
pub(crate) use self::round::BlendOp;
pub(crate) use self::round::FilterOp;
use self::round::Rounds;
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

/// Identifies one of the intermediate textures used by the hybrid renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextureTarget {
    /// First layer atlas texture.
    Layer0,
    /// Second layer atlas texture.
    Layer1,
    /// First scratch texture.
    Scratch0,
    /// Second scratch texture.
    Scratch1,
}

impl TextureTarget {
    pub(crate) fn layer(index: usize) -> Self {
        match index {
            0 => Self::Layer0,
            1 => Self::Layer1,
            _ => panic!("vello_hybrid only supports two layer textures"),
        }
    }

    pub(crate) fn scratch(index: usize) -> Self {
        match index {
            0 => Self::Scratch0,
            1 => Self::Scratch1,
            _ => panic!("vello_hybrid only supports two scratch textures"),
        }
    }

    pub(crate) fn index(self) -> usize {
        match self {
            Self::Layer0 | Self::Scratch0 => 0,
            Self::Layer1 | Self::Scratch1 => 1,
        }
    }
}

/// A rectangular region in one of the layer atlas textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerTextureRegion {
    /// Layer texture index, currently `0` or `1`.
    pub(crate) texture_index: usize,
    /// X coordinate in the layer texture.
    pub(crate) x: u16,
    /// Y coordinate in the layer texture.
    pub(crate) y: u16,
    /// Width of the region.
    pub(crate) width: u16,
    /// Height of the region.
    pub(crate) height: u16,
    /// Bounds of this layer in viewport coordinates.
    pub(crate) scene_bbox: RectU16,
}

/// A rectangular region in one of the scratch textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScratchRegion {
    /// Scratch texture index, currently `0` or `1`.
    pub(crate) texture_index: usize,
    /// Region in the scratch texture.
    pub(crate) rect: RectU16,
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
    /// Ensure intermediate layer/scratch textures required by this scene are allocated.
    fn prepare_intermediate_textures(&mut self, requirements: TextureRequirements);

    /// Return the dimensions of each layer atlas texture.
    fn layer_texture_size(&self) -> (u32, u32);

    /// Clear rectangular regions in a texture to transparent black.
    fn clear_rects(&mut self, target: TextureTarget, populate: impl FnOnce(&mut Vec<RectU16>));

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
    fn blend(&mut self, blends: &[BlendOp], texture_index: usize);

    /// Apply filter operations to already-rendered layer atlas regions.
    fn apply_filters(&mut self, filters: &[FilterOp], texture_index: usize);

    /// Reusable CPU-side storage for rounds execution.
    fn schedule_scratch(&mut self) -> &mut ScheduleScratch;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextureRequirements {
    pub(crate) layer_textures: [bool; 2],
    pub(crate) scratch_textures: [bool; 2],
}

impl TextureRequirements {
    fn for_scene(scene: &Scene) -> Self {
        let layer_textures = if scene.recorder.root_is_blend_target {
            // When the root is a blend target, it becomes an intermediate layer too, so we need
            // both atlas textures even if the recorded layer nesting itself is shallow.
            [true, true]
        } else {
            let mut layer_textures = [false; 2];
            // The root layer is depth 0. Direct child layers have depth 1 and need one atlas
            // texture; deeper nesting alternates between the two atlas textures.
            for depth in 1..=scene.recorder.max_layer_depth.min(2) {
                layer_textures[depth & 1] = true;
            }
            layer_textures
        };
        let scratch_textures = if scene.recorder.has_filter_layer {
            [true, true]
        } else if scene.recorder.has_non_default_blend {
            [true, false]
        } else {
            [false, false]
        };

        Self {
            layer_textures,
            scratch_textures,
        }
    }
}

/// Backend agnostic enum that specifies the operation to perform to the output attachment at the
/// start of a render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoadOp {
    Load,
    Clear,
}

#[derive(Debug, Default)]
pub(crate) struct ScheduleScratch {
    layer_draws: [Draw; 2],
    root_batch: RootBatch,
}

impl ScheduleScratch {
    fn clear_round(&mut self) {
        for draw in &mut self.layer_draws {
            draw.clear();
        }
    }
}

/// Render the supported subset of a scene through the new recorder-based scheduler.
pub(crate) fn render_scene<R: RendererBackend>(
    renderer: &mut R,
    scene: &Scene,
    root_output_target: RootRenderTarget,
    paint_idxs: &[u32],
    encoded_paints: &[EncodedPaint],
) -> Result<(), RenderError> {
    renderer.prepare_intermediate_textures(TextureRequirements::for_scene(scene));
    let strip_storage = scene.strip_storage.borrow();
    let mut builder = ScheduleBuilder::new(
        scene,
        &strip_storage,
        root_output_target,
        paint_idxs,
        encoded_paints,
        renderer.layer_texture_size(),
    );
    let rounds = builder.build()?;
    let mut scratch = ScheduleScratch::default();
    core::mem::swap(&mut scratch, renderer.schedule_scratch());
    execute_rounds(renderer, &rounds, &mut scratch);
    core::mem::swap(&mut scratch, renderer.schedule_scratch());
    Ok(())
}

fn execute_rounds<R: RendererBackend>(
    renderer: &mut R,
    rounds: &Rounds,
    scratch: &mut ScheduleScratch,
) {
    let direct_root_opaque = collect_direct_root_opaque(rounds);
    if !direct_root_opaque.draw.is_empty() {
        renderer.render_strips(
            &direct_root_opaque.draw.opaque,
            &[],
            &[],
            StripPassRenderTarget::Root(RootRenderTarget::UserSurface),
            direct_root_opaque.load_op,
        );
    }

    for round in &rounds.rounds {
        scratch.clear_round();

        for texture_index in [1, 0] {
            let layer_round = &round.layer_passes[texture_index];

            for pass in &layer_round.render_passes {
                if pass.load_op == LoadOp::Load {
                    scratch.layer_draws[texture_index].append(&pass.draw);
                }
            }

            for pass in &layer_round.render_passes {
                if pass.load_op != LoadOp::Load {
                    execute_pass(renderer, pass);
                }
            }

            let draw = &scratch.layer_draws[texture_index];
            renderer.render_strips(
                &draw.opaque,
                &draw.alpha,
                &draw.external_texture_runs,
                StripPassRenderTarget::LayerAtlas(texture_index),
                LoadOp::Load,
            );

            renderer.apply_filters(&layer_round.filters, texture_index);
        }

        execute_root_passes(renderer, &mut scratch.root_batch, &round.root_passes);

        for texture_index in 0..round.layer_passes.len() {
            renderer.blend(&round.layer_passes[texture_index].blends, texture_index);
        }
        clear_layer_regions(renderer, &round.layer_clears);
        clear_scratch_regions(renderer, &round.scratch_clears);
    }
}

fn clear_layer_regions<R: RendererBackend>(renderer: &mut R, regions: &[LayerTextureRegion]) {
    let Some(max_texture_index) = regions.iter().map(|region| region.texture_index).max() else {
        return;
    };

    for texture_index in 0..=max_texture_index {
        renderer.clear_rects(TextureTarget::layer(texture_index), |clear_rects| {
            clear_rects.extend(regions.iter().filter_map(|region| {
                if region.texture_index != texture_index {
                    return None;
                }

                let rect = region_clear_rect(region.x, region.y, region.width, region.height);
                (!rect.is_empty()).then_some(rect)
            }));
        });
    }
}

fn clear_scratch_regions<R: RendererBackend>(renderer: &mut R, regions: &[ScratchRegion]) {
    let Some(max_texture_index) = regions.iter().map(|region| region.texture_index).max() else {
        return;
    };

    for texture_index in 0..=max_texture_index {
        renderer.clear_rects(TextureTarget::scratch(texture_index), |clear_rects| {
            clear_rects.extend(regions.iter().filter_map(|region| {
                if region.texture_index != texture_index {
                    return None;
                }

                (!region.rect.is_empty()).then_some(region.rect)
            }));
        });
    }
}

fn region_clear_rect(x: u16, y: u16, width: u16, height: u16) -> RectU16 {
    RectU16::new(x, y, x + width, y + height)
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

fn collect_direct_root_opaque(rounds: &Rounds) -> DirectRootOpaque {
    let mut opaque = DirectRootOpaque::default();
    for pass in rounds.rounds.iter().flat_map(|round| &round.root_passes) {
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

fn execute_pass<R: RendererBackend>(renderer: &mut R, pass: &round::RenderPass) {
    renderer.render_strips(
        &pass.draw.opaque,
        &pass.draw.alpha,
        &pass.draw.external_texture_runs,
        pass.target.strip_pass_target(),
        pass.load_op,
    );
}

fn execute_root_passes<'a, R: RendererBackend>(
    renderer: &mut R,
    batch: &mut RootBatch,
    root_passes: impl IntoIterator<Item = &'a round::RenderPass>,
) {
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
        pass: &round::RenderPass,
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
        self.draw.clear();
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

    fn clear(&mut self) {
        self.draw.clear();
        self.root_load_op = LoadOp::Load;
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
