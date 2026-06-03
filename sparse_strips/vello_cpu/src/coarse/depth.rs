// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Depth buffers for saving unnecessary per-pixel work.
//!
//! GPUs have depth buffers, so why not use them in CPU rendering as well! For many of the existing
//! CPU-based 2D renderers, this is not really possible because they use immediate-mode rendering.
//! But Vello CPU first converts all rendering commands into strips and then processes them, meaning
//! that before we even start with rasterization, we already have a rough idea of what we are about
//! to rasterize. Therefore, we can use this information to go about performing rasterization in a
//! smarter way.
//!
//! Unlike GPUs, it is not feasible to have a per-pixel depth buffer. While certain per-pixel work
//! can be expensive (images, gradients), what is even more expensive is splitting up work in a
//! too granular fashion, especially when only dealing with solid colors. It is much faster to just
//! fill a 256x1 buffer of pixels with a single colors than doing it in 32 chunks of 4x1, just to
//! save 50% pixel work.
//!
//! Therefore, the CPU-based depth buffer acts at a much coarser granularity. Vertically, it comes
//! very natural to simply decide that one depth buffer entry covers a range of [`Tile::HEIGHT`]
//! pixels, since all commands are executed at this height anyway. Choosing a width is much trickier:
//! Similarly, the width should be a multiple of [`Tile::WIDTH`], but using this as the granularity
//! is still to narrow. After some empirical measurements, it was decided that a width of
//! [`DEPTH_BUCKET_WIDTH`] overall represents a good compromise across different paint types.
//!
//! How this essentially works now is that, once we've collected all strips, we split them up
//! into opaque fills aligned to the depth bucket width, and all other fills which either have
//! transparency or stem from non-aligned parts of an opaque fill. During rasterization, we
//! first render the opaque strips front-to-back, each time checking the depth buffer whether
//! this hasn't already been filled by an opaque strip with a higher z-index. Otherwise, we
//! perform the fill and update the depth buffer.
//!
//! Then, we simply render the remaining strips back to front, again always comparing the depth
//! against what's written in the depth buffer and skipping any commands that would be fully
//! covered by existing content.

use super::cmd::Span;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::tile::Tile;

pub(crate) const DEPTH_BUCKET_WIDTH: u16 = 128;
const DEPTH_BUCKET_TILE_WIDTH: u16 = DEPTH_BUCKET_WIDTH / Tile::WIDTH;
const _: () = assert!(
    DEPTH_BUCKET_WIDTH.is_multiple_of(Tile::WIDTH),
    "depth bucket width must be a multiple of tile width"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DepthSegment {
    /// An opaque span that cannot be tracked in the depth buffer because it is not
    /// aligned to whole depth buckets.
    Regular,
    /// An opaque span that is aligned and can therefore be rendered front-to-back with
    /// depth-buffer write enabled.
    Opaque,
}

/// Splits a tile-aligned span into regular edge spans and a depth-trackable
/// opaque middle span.
pub(crate) fn split_opaque_span(span: Span, mut segment: impl FnMut(Span, DepthSegment)) {
    let x = span.tile_x();
    let end = span.tile_end();
    let aligned_x = x.next_multiple_of(DEPTH_BUCKET_TILE_WIDTH);
    let aligned_end = (end / DEPTH_BUCKET_TILE_WIDTH) * DEPTH_BUCKET_TILE_WIDTH;

    if aligned_x >= aligned_end {
        segment(span, DepthSegment::Regular);

        return;
    }

    if x < aligned_x {
        segment(Span::new_tile(x, aligned_x - x), DepthSegment::Regular);
    }

    if aligned_x < aligned_end {
        segment(
            Span::new_tile(aligned_x, aligned_end - aligned_x),
            DepthSegment::Opaque,
        );
    }

    if aligned_end < end {
        segment(
            Span::new_tile(aligned_end, end - aligned_end),
            DepthSegment::Regular,
        );
    }
}

/// Coarse state for the depth buffer.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DepthState {
    /// Coarse union of all depth-trackable opaque spans in this row.
    bounds: Option<Span>,
    /// Maximum draw ID of any depth-trackable opaque command in this row.
    ///
    /// Draw IDs start at 1, so 0 represents "no opaque command".
    max_draw_id: u32,
}

impl DepthState {
    pub(crate) fn reset(&mut self) {
        *self = DepthState::default();
    }

    pub(crate) fn include_span(&mut self, span: Span, draw_id: u32) {
        if let Some(bounds) = &mut self.bounds {
            bounds.extend(span);
        } else {
            self.bounds = Some(span);
        }

        self.max_draw_id = self.max_draw_id.max(draw_id);
    }

    /// Returns whether this draw can skip consulting the depth buffer.
    pub(crate) fn can_skip(self, span: Span, draw_id: u32) -> bool {
        if draw_id >= self.max_draw_id {
            return true;
        }

        let Some(opaque_bounds) = self.bounds else {
            return true;
        };

        let opaque_start = opaque_bounds.tile_x();
        let opaque_end = opaque_bounds.tile_end();
        let x = span.tile_x();
        let end = span.tile_end();

        x >= opaque_end || end <= opaque_start
    }
}

#[derive(Debug)]
pub(crate) struct DepthBuffer {
    data: Vec<u32>,
}

impl DepthBuffer {
    pub(crate) fn new(buffer_width: u16) -> Self {
        Self {
            data: vec![0; usize::from(buffer_width.div_ceil(DEPTH_BUCKET_WIDTH))],
        }
    }

    /// Calls `f` for every unset depth run touched by `span`.
    pub(crate) fn for_each_unset_run(&self, span: Span, mut f: impl FnMut(Span)) {
        let (mut idx, depth_end) = self.range(span);
        while let Some((span, _)) = self.next_unset_run(&mut idx, depth_end) {
            f(span);
        }
    }

    /// Calls `f` for every depth run visible to `draw_id` in `span`, and then marks it with
    /// `draw_id`.
    pub(crate) fn for_each_visible_run_with_write(
        &mut self,
        span: Span,
        draw_id: u32,
        mut f: impl FnMut(Span),
    ) {
        let (mut idx, depth_end) = self.range(span);

        while let Some((span, depth_range)) = self.next_unset_run(&mut idx, depth_end) {
            f(span);
            self.mark(depth_range, draw_id);
        }
    }

    /// Calls `f` for every depth run visible to `draw_id` in `span`.
    pub(crate) fn for_each_visible_run(&self, span: Span, draw_id: u32, mut f: impl FnMut(Span)) {
        let (mut idx, depth_end) = self.range(span);

        while let Some(span) = self.next_visible_run(&mut idx, depth_end, draw_id, span) {
            f(span);
        }
    }

    /// Returns the depth-bucket index range touched by `span`.
    fn range(&self, span: Span) -> (usize, usize) {
        (
            usize::from(span.pixel_x() / DEPTH_BUCKET_WIDTH),
            usize::from(span.pixel_end().div_ceil(DEPTH_BUCKET_WIDTH)).min(self.data.len()),
        )
    }

    pub(crate) fn clear_range(&mut self, span: Span) {
        let (start, end) = self.range(span);
        self.data[start..end].fill(0);
    }

    fn mark(&mut self, range: core::ops::Range<usize>, draw_id: u32) {
        self.data[range].fill(draw_id);
    }

    /// Finds the next consecutive run of unset depth buckets.
    fn next_unset_run(
        &self,
        idx: &mut usize,
        end: usize,
    ) -> Option<(Span, core::ops::Range<usize>)> {
        while *idx < end && self.data[*idx] != 0 {
            *idx += 1;
        }

        let run_start = *idx;
        while *idx < end && self.data[*idx] == 0 {
            *idx += 1;
        }

        if run_start == *idx {
            return None;
        }

        Some((bucket_span(run_start, *idx), run_start..*idx))
    }

    /// Finds the next consecutive run visible to `draw_id`.
    fn next_visible_run(
        &self,
        idx: &mut usize,
        end: usize,
        draw_id: u32,
        bounds: Span,
    ) -> Option<Span> {
        while *idx < end && self.data[*idx] > draw_id {
            *idx += 1;
        }

        let run_start = *idx;
        while *idx < end && self.data[*idx] <= draw_id {
            *idx += 1;
        }

        if run_start == *idx {
            return None;
        }

        bucket_span(run_start, *idx).intersect(bounds)
    }
}

fn bucket_span(start: usize, end: usize) -> Span {
    let x = start as u16 * DEPTH_BUCKET_WIDTH;
    Span::new(x, (end - start) as u16 * DEPTH_BUCKET_WIDTH)
}
