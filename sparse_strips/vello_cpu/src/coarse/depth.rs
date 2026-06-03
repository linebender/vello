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
//! fill a 256x1 buffer of pixels with a single colors than doing it in in 32 chunks of 4x1, just to
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
//! this are hasn't already been filled by an opaque strip with a higher z-index. Otherwise, we
//! perform the fill and update the depth buffer.
//!
//! Then, we simply render the remaining strips back to front, again always comparing the depth
//! against what's written in the depth buffer and skipping any commands that would be fully
//! covered by existing content.

use super::cmd::Span;
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
        segment(Span::new(x, aligned_x - x), DepthSegment::Regular);
    }

    if aligned_x < aligned_end {
        segment(
            Span::new(aligned_x, aligned_end - aligned_x),
            DepthSegment::Opaque,
        );
    }

    if aligned_end < end {
        segment(
            Span::new(aligned_end, end - aligned_end),
            DepthSegment::Regular,
        );
    }
}

/// Returns whether a draw might be affected by later opaque draws in this row.
///
/// If no later opaque draw exists (`draw_id >= max_opaque_draw_id`) or the draw
/// is outside the combined opaque bounds, fine rasterization can skip consulting
/// the depth buffer for this command.
pub(crate) fn affects_later_draw(
    span: Span,
    draw_id: u32,
    max_opaque_draw_id: u32,
    opaque_bounds: Option<Span>,
) -> bool {
    if draw_id >= max_opaque_draw_id {
        return false;
    }

    let Some(opaque_bounds) = opaque_bounds else {
        return false;
    };

    let opaque_start = opaque_bounds.pixel_x();
    let opaque_end = opaque_bounds.pixel_end();
    let x = span.pixel_x();
    let end = span.pixel_end();

    x < opaque_end && end > opaque_start
}

/// Returns the depth-bucket index range touched by the pixel range `x..end`.
pub(crate) fn depth_range(x: u16, end: u16) -> (usize, usize) {
    (
        usize::from(x / DEPTH_BUCKET_WIDTH),
        usize::from(end.div_ceil(DEPTH_BUCKET_WIDTH)),
    )
}

/// Finds the next consecutive run of unset depth buckets.
///
/// Used by opaque fills: unset buckets (`0`) have not yet been covered by a
/// later opaque draw, so the opaque fill can render those buckets and then
/// write its draw id into the returned depth range.
pub(crate) fn next_unset_run(
    depth: &[u32],
    idx: &mut usize,
    end: usize,
) -> Option<(u16, u16, core::ops::Range<usize>)> {
    while *idx < end && depth[*idx] != 0 {
        *idx += 1;
    }

    let run_start = *idx;
    while *idx < end && depth[*idx] == 0 {
        *idx += 1;
    }

    if run_start == *idx {
        return None;
    }

    let x = run_start as u16 * DEPTH_BUCKET_WIDTH;
    let run_end = *idx as u16 * DEPTH_BUCKET_WIDTH;
    Some((x, run_end, run_start..*idx))
}

/// Finds the next consecutive run visible to `draw_id`.
///
/// Depth entries larger than `draw_id` belong to later opaque draws and hide
/// this command. Entries less than or equal to `draw_id` are still visible for
/// this command and are returned as pixel ranges clipped to `x..x_end`.
pub(crate) fn next_visible_run(
    depth: &[u32],
    idx: &mut usize,
    end: usize,
    draw_id: u32,
    x: u16,
    x_end: u16,
) -> Option<(u16, u16)> {
    while *idx < end && depth[*idx] > draw_id {
        *idx += 1;
    }

    let run_start = *idx;
    while *idx < end && depth[*idx] <= draw_id {
        *idx += 1;
    }

    if run_start == *idx {
        return None;
    }

    let run_x = x.max(run_start as u16 * DEPTH_BUCKET_WIDTH);
    let run_end = x_end.min(*idx as u16 * DEPTH_BUCKET_WIDTH);
    Some((run_x, run_end))
}
