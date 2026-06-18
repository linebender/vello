// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! New scheduler implementation for `vello_hybrid`.
//!
//! This first slice only supports root-level draw commands. Layers are deliberately rejected until
//! the layer-atlas scheduler lands.

use crate::scene::{FastPathRect, RecordedDraw};
use crate::schedule::{GpuStripBuilder, Scheduler as ExistingScheduler, make_gpu_rect, split_rect};
use crate::{GpuStrip, RenderError, Scene};
use alloc::vec::Vec;
use vello_common::TextureId;
use vello_common::encode::EncodedPaint;
use vello_common::paint::Paint;
use vello_common::record::RecordedCmd;
use vello_common::render_graph::LayerId;
use vello_common::strip_generator::StripStorage;
use vello_common::tile::Tile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootRenderTarget {
    /// The root render target is the user-provided surface.
    UserSurface,
    /// The root render target is an atlas layer.
    AtlasLayer,
}

/// Specifies the target for a strip render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum StripPassRenderTarget {
    /// Render to the root output target.
    Root(RootRenderTarget),
    /// Render to a layer in the filter atlas.
    FilterLayer(LayerId),
    /// Render to one of the slot textures used for clipping/blending.
    SlotTexture(u8),
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

#[allow(dead_code)]
pub(crate) trait RendererBackend {
    /// Clear specific slots in a texture.
    fn clear_slots(&mut self, texture_index: usize, slots: &[u32]);

    /// Execute a render pass for strips, split into opaque and alpha passes.
    fn render_strips(
        &mut self,
        opaque_strips: &[GpuStrip],
        alpha_strips: &[GpuStrip],
        external_texture_runs: &[ExternalTextureRun],
        target: StripPassRenderTarget,
        load_op: LoadOp,
    );

    /// Apply filter effects for the given layer after its content has been rendered.
    fn apply_filter(&mut self, layer_id: LayerId);
}

/// Backend agnostic enum that specifies the operation to perform to the output attachment at the
/// start of a render pass.
#[derive(Debug, PartialEq)]
#[allow(dead_code)]
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
    let mut builder = DirectRootBuilder::default();
    let strip_storage = scene.strip_storage.borrow();

    for cmd in &scene.recorder.root_cmds {
        match cmd {
            RecordedCmd::Batch(range) => {
                for draw in &scene.recorder.draws[range.start as usize..range.end as usize] {
                    builder.push_draw(draw, &strip_storage, scene, encoded_paints, paint_idxs);
                }
            }
            RecordedCmd::Layer(_) => {
                return Err(RenderError::UnsupportedFeature(
                    "layers are not supported by schedule_new yet",
                ));
            }
        }
    }

    builder.flush(renderer, root_output_target);
    Ok(())
}

#[derive(Debug, Default)]
struct DirectRootBuilder {
    draw: Draw,
    depth: DepthCounter,
}

impl DirectRootBuilder {
    fn push_draw(
        &mut self,
        draw: &RecordedDraw,
        strip_storage: &StripStorage,
        scene: &Scene,
        encoded_paints: &[EncodedPaint],
        paint_idxs: &[u32],
    ) {
        match draw {
            RecordedDraw::Path(path) => {
                self.push_path(
                    path.strips.clone(),
                    path.paint.clone(),
                    strip_storage,
                    scene,
                    encoded_paints,
                    paint_idxs,
                );
            }
            RecordedDraw::Rect(rect) => {
                self.push_rect(&rect.rect, encoded_paints, paint_idxs);
            }
        }
    }

    fn push_path(
        &mut self,
        strips: core::ops::Range<usize>,
        paint: Paint,
        strip_storage: &StripStorage,
        scene: &Scene,
        encoded_paints: &[EncodedPaint],
        paint_idxs: &[u32],
    ) {
        let strips = &strip_storage.strips[strips];

        if strips.is_empty() {
            return;
        }

        let is_opaque = is_paint_opaque(&paint, encoded_paints);
        let depth_index = self.depth.next(is_opaque);

        for i in 0..strips.len() - 1 {
            let strip = &strips[i];

            if strip.x >= scene.width {
                continue;
            }

            let next_strip = &strips[i + 1];
            let col = strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let next_col = next_strip.alpha_idx() / u32::from(Tile::HEIGHT);
            let strip_width = next_col.saturating_sub(col) as u16;
            let x0 = strip.x;
            let y = strip.y;

            if strip_width > 0 {
                let processed =
                    ExistingScheduler::process_paint(&paint, encoded_paints, (x0, y), paint_idxs);
                self.draw.push_alpha(
                    GpuStripBuilder::at_surface(x0, y, strip_width)
                        .with_sparse(strip_width, col)
                        .paint(processed.payload, processed.paint, depth_index),
                    processed.external_texture_id,
                );
            }

            if next_strip.fill_gap() && strip.strip_y() == next_strip.strip_y() {
                let x1 = x0.saturating_add(strip_width);
                let x2 = next_strip.x.min(scene.width);
                if x2 > x1 {
                    let processed = ExistingScheduler::process_paint(
                        &paint,
                        encoded_paints,
                        (x1, y),
                        paint_idxs,
                    );
                    let strip = GpuStripBuilder::at_surface(x1, y, x2 - x1).paint(
                        processed.payload,
                        processed.paint,
                        depth_index,
                    );
                    if is_opaque {
                        self.draw.push_opaque(strip);
                    } else {
                        self.draw.push_alpha(strip, processed.external_texture_id);
                    }
                }
            }
        }
    }

    fn push_rect(
        &mut self,
        rect: &FastPathRect,
        encoded_paints: &[EncodedPaint],
        paint_idxs: &[u32],
    ) {
        let is_opaque = is_paint_opaque(&rect.paint, encoded_paints);
        let depth_index = self.depth.next(is_opaque);
        pack_rectangle_into_gpu(
            rect,
            encoded_paints,
            paint_idxs,
            depth_index,
            is_opaque,
            &mut self.draw,
        );
    }

    fn flush<R: RendererBackend>(
        &mut self,
        renderer: &mut R,
        root_output_target: RootRenderTarget,
    ) {
        if self.draw.is_empty() {
            return;
        }

        self.draw.opaque.reverse();
        renderer.render_strips(
            &self.draw.opaque,
            &self.draw.alpha,
            &self.draw.external_texture_runs,
            StripPassRenderTarget::Root(root_output_target),
            LoadOp::Load,
        );
    }
}

#[derive(Debug, Default)]
struct DepthCounter {
    count: u32,
}

impl DepthCounter {
    #[inline(always)]
    fn next(&mut self, opaque: bool) -> u32 {
        self.count += opaque as u32;
        self.count
    }
}

#[derive(Debug, Default)]
struct Draw {
    opaque: Vec<GpuStrip>,
    alpha: Vec<GpuStrip>,
    external_texture_runs: Vec<ExternalTextureRun>,
}

impl Draw {
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

fn is_paint_opaque(paint: &Paint, encoded_paints: &[EncodedPaint]) -> bool {
    match paint {
        Paint::Solid(color) => color.is_opaque(),
        Paint::Indexed(indexed_paint) => match encoded_paints.get(indexed_paint.index()) {
            Some(EncodedPaint::Image(image)) => {
                !image.may_have_transparency
                    && image.sampler.alpha == 1.0
                    && image.tint.is_none_or(|t| t.color.components[3] >= 1.0)
            }
            Some(EncodedPaint::ExternalTexture(_)) => false,
            Some(EncodedPaint::Gradient(gradient)) => !gradient.may_have_transparency,
            Some(EncodedPaint::BlurredRoundedRect(_)) => false,
            None => unreachable!("Paint must be in encoded paints"),
        },
    }
}

fn pack_rectangle_into_gpu(
    rect: &FastPathRect,
    encoded_paints: &[EncodedPaint],
    paint_idxs: &[u32],
    depth_index: u32,
    is_opaque: bool,
    draw: &mut Draw,
) {
    let split = split_rect(rect);

    let mut is_first = true;
    for part in [
        Some(split.main),
        split.top,
        split.bottom,
        split.left,
        split.right,
    ]
    .into_iter()
    .flatten()
    {
        let processed = ExistingScheduler::process_paint(
            &rect.paint,
            encoded_paints,
            (part.x, part.y),
            paint_idxs,
        );
        let strip = make_gpu_rect(part, processed.payload, processed.paint, depth_index);
        if is_first && is_opaque && part.frac == 0 {
            draw.push_opaque(strip);
        } else {
            draw.push_alpha(strip, processed.external_texture_id);
        }
        is_first = false;
    }
}
