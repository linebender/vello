// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Port of Skia's `Tiler.h` (`Tiles::makeTilesMSAA` and helpers), at 16×16 tiles.
//!
//! OWNER: tiler worker. Frozen interface (do not change signatures without coordinating):
//!
//! ```ignore
//! pub struct Tiles { .. }
//! impl Tiles {
//!     pub fn new() -> Self;
//!     pub fn reset(&mut self);
//!     /// Returns true if a culling event occurred (lines left of `clip` wrote to `histogram`).
//!     pub fn make_tiles_msaa(&mut self, lines: &[Line], clip: IRect, histogram: &mut WindingHistogram) -> bool;
//!     pub fn sort_tiles(&mut self);
//!     pub fn tiles(&self) -> &[Tile];
//! }
//! ```
//!
//! Differences from Skia:
//! - The tile size is fixed at [`TILE_SIZE`] instead of being a template parameter.
//! - A path is a slice of [`Line`]s instead of a NaN-separated `Polyline`, and
//!   [`Tile::line_idx`] is the index of the line in that slice. (Skia stores the index of the
//!   line's first point in the polyline, which plays the same role.)
//! - Float to int conversions saturate (`as`), where an out of range `static_cast` is UB in C++.
//!   Vello's flatten does not cull, so lines can be arbitrarily far outside the clip.
//! - [`WindingHistogram::add_winding_range`] requires `start <= end`, so empty ranges are
//!   skipped explicitly.

#![allow(dead_code, reason = "Not called by `make_strips` yet")]

use super::TILE_SIZE;
use super::tile::{IRect, Line, Tile, WindingHistogram};

// Skia's `Tiles` inherits these from `IntersectionBits`.
const T: u32 = Tile::T;
const B: u32 = Tile::B;
const L: u32 = Tile::L;
const R: u32 = Tile::R;
const W: u32 = Tile::W;
const BOT_SHIFT: u32 = Tile::BOT_SHIFT;
const LEFT_SHIFT: u32 = Tile::LEFT_SHIFT;
const RIGHT_SHIFT: u32 = Tile::RIGHT_SHIFT;

// For now, only square tiles are supported.
const TILE_WIDTH: u16 = TILE_SIZE;
const TILE_HEIGHT: u16 = TILE_SIZE;

const INV_W: f32 = 1.0 / TILE_WIDTH as f32;
const INV_H: f32 = 1.0 / TILE_HEIGHT as f32;

/// Tiles touched by a path's lines.
///
/// Port of Skia's `Tiles<kTileWidth, kTileHeight>`: holds the tiles and manages their state
/// (sorting).
#[derive(Debug, Default)]
pub(super) struct Tiles {
    tiles: Vec<Tile>,
}

/// Port of `Tiles::LineContext`: per-line state shared by the helpers.
#[derive(Clone, Copy, Debug)]
struct LineContext {
    line_idx: u32,
    top_x: f32,
    top_y: f32,
    bottom_x: f32,
    bottom_y: f32,
    x_slope: f32,
    line_left_x: f32,
    line_right_x: f32,
    p0_tile_x: i32,
    p0_tile_y: i32,
    p1_tile_x: i32,
    p1_tile_y: i32,
    tile_columns: u16,
    winding_dir: i16,
    clip_left_tile: f32,
    clip_top_tile: f32,
    min_tile_x: u16,
}

impl Tiles {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn reset(&mut self) {
        self.tiles.clear();
    }

    /// Port of `Tiles::makeTilesMSAA(polyline, clip, histogram)`.
    ///
    /// Appends the tiles touched by `lines` to the tile buffer. Returns whether a culling event
    /// occurred, i.e. whether lines left of `clip` may have written their winding to `histogram`.
    /// When it returns false, `histogram` is untouched. `histogram` must have at least
    /// `clip.bottom.div_ceil(16)` rows.
    ///
    /// From Skia: this function has two purposes.
    ///
    /// 1. The viewport is divided into tiles. Tiles act as a "super coarse rasterization stage",
    ///    which elides carrying a scanline for each MSAA subsample point. So for each line
    ///    produced by flattening, we need to find which tiles this line intersects. Tile edge
    ///    touches are vertical exclusive, horizontal inclusive. This is because:
    ///    - In the vertical case, inclusivity would cause the coarse winding to be double counted
    ///      or negated by a single-point tile produced by a grazing touch.
    ///    - In the horizontal case, the inclusivity is necessary because the tile produced by the
    ///      succeeding line may not consider itself left-touching (depending on its direction),
    ///      so the single-point tile is necessary to carry over the left-edge winding.
    ///
    ///    Using the line's direction enforces mutually exclusive ownership of boundary
    ///    intersections between consecutive segments *of the same direction*, ensuring that the
    ///    left-edge winding is neither lost nor double-counted:
    ///
    ///    ```text
    ///                        Left edge behavior:
    ///    +----------------------+----------+------------+
    ///    | X Direction          | Endpoint | L Bit Set? |
    ///    +----------------------+----------+------------+
    ///    | Left-to-Right (L->R) | Start    | No         |
    ///    | Left-to-Right (L->R) | End      | Yes        |
    ///    | Right-to-Left (R->L) | Start    | Yes        |
    ///    | Right-to-Left (R->L) | End      | No         |
    ///    +----------------------+----------+------------+
    ///    ```
    ///
    /// 2. To enable parallel rasterization, we need to establish a source of truth for the line
    ///    intersection points on tiles, such that adjacent tiles agree where intersections occur.
    ///    Instead of calculating exact intersection coordinates here, we defer the heavy math to
    ///    the rasterizer and produce a lightweight intersection bitmask, which unambiguously
    ///    defines which edges of a tile a line segment touches (see [`Tile`]):
    ///    - W (Winding): Tracks whether the line touched the top edge of the tile.
    ///    - R/L/B/T: Right, Left, Bottom, and Top edge intersections.
    pub(super) fn make_tiles_msaa(
        &mut self,
        lines: &[Line],
        clip: IRect,
        histogram: &mut WindingHistogram,
    ) -> bool {
        debug_assert!(
            lines.len() <= Tile::MAX_LINES_PER_PATH as usize,
            "too many lines in one path"
        );
        debug_assert!(
            clip.right <= i32::from(u16::MAX) && clip.bottom <= i32::from(u16::MAX),
            "clip too large"
        );

        let mut culling_event_occurred = false;
        if clip.is_empty() {
            return culling_event_occurred;
        }

        // As in Skia, the clip's right and bottom are truncated to u16.
        let tile_columns = div_ceil(clip.right as u16, TILE_WIDTH);
        let tile_rows = div_ceil(clip.bottom as u16, TILE_HEIGHT);

        let min_tile_x = f32_to_u16_sat(clip.left as f32 * INV_W);
        let min_tile_y = f32_to_u16_sat(clip.top as f32 * INV_H);

        let clip_left_tile = f32::from(min_tile_x);
        let clip_top_tile = f32::from(min_tile_y);
        let clip_right_tile = clip.right as f32 * INV_W;

        for (line_idx, line) in lines.iter().enumerate() {
            // map line into tile units
            let p0_x = line.p0.x * INV_W;
            let p0_y = line.p0.y * INV_H;
            let p1_x = line.p1.x * INV_W;
            let p1_y = line.p1.y * INV_H;

            let (line_left_x, line_right_x) = if p0_x < p1_x {
                (p0_x, p1_x)
            } else {
                (p1_x, p0_x)
            };

            // If the leftmost point of this line is right of the clip bounds, cull it. Although we
            // cull path verbs right of the viewport in the flattening stage, a right edge crossing
            // path verb may still be flattened into lines, some of which may be completely outside
            // of the viewport/clip. (Vello's flatten doesn't cull at all, so this culls everything
            // right of the clip.)
            if line_left_x > clip_right_tile {
                continue;
            }

            let (line_top_y, line_top_x, line_bottom_y, line_bottom_x, winding_dir) = if p0_y < p1_y
            {
                (p0_y, p0_x, p1_y, p1_x, 1_i16)
            } else {
                (p1_y, p1_x, p0_y, p0_x, -1_i16)
            };

            let y_top_tiles = min_tile_y.max(f32_to_u16_sat(line_top_y).min(tile_rows));
            let line_bottom_y_ceil = line_bottom_y.ceil();
            let y_bottom_tiles = min_tile_y.max(f32_to_u16_sat(line_bottom_y_ceil).min(tile_rows));

            // If y_top_tiles == y_bottom_tiles, then the line is either completely above or below
            // the viewport/clip OR it is perfectly horizontal and aligned to the tile grid,
            // contributing no winding. In either case, it should be culled.
            if y_top_tiles >= y_bottom_tiles {
                // Technically, the `>` part of the `>=` is unnecessary due to clamping, but this
                // gives stronger signal.
                continue;
            }

            let p0_tile_x = line_top_x.floor() as i32;
            let p0_tile_y = line_top_y.floor() as i32;
            let p1_tile_x = line_bottom_x.floor() as i32;
            let p1_tile_y = if line_bottom_y == line_bottom_y_ceil {
                // `line_bottom_y > 0` here, so this can't overflow.
                line_bottom_y as i32 - 1
            } else {
                line_bottom_y.floor() as i32
            };

            // Each line processed falls into 1 of four categories:
            //  1) The line is completely to the left of the clip (culled into WindingHistogram).
            //  2) The line is perfectly vertical.
            //  3) The line produces a single tile.
            //  4) The line is general (sloped across tiles, potentially crossing clip edges).
            let not_same_tile = p0_tile_y != p1_tile_y || p0_tile_x != p1_tile_x;

            // Left-edge culling: Lines completely to the left of the clip bounds do not generate
            // any tiles in the visible grid. However, their vertical span contributes winding to
            // all scanlines to their right. Record this winding directly into the WindingHistogram
            // rather than allocating tiles into the tile buffer.
            if line_right_x < clip_left_tile {
                culling_event_occurred = true;
                let is_start_culled = line_top_y < clip_top_tile;
                if !is_start_culled && f32::from(y_top_tiles) >= line_top_y {
                    histogram.add_winding(usize::from(y_top_tiles), winding_dir);
                }

                let y_start = if is_start_culled {
                    y_top_tiles
                } else {
                    y_top_tiles + 1
                };
                let line_bottom_floor = line_bottom_y.floor();
                let y_end_idx = f32_to_u16_sat(line_bottom_floor).min(tile_rows);
                // Skia's `addWindingRange` is a no-op for empty (or reversed) ranges.
                if y_start < y_end_idx {
                    histogram.add_winding_range(
                        usize::from(y_start),
                        usize::from(y_end_idx),
                        winding_dir,
                    );
                }

                if p0_tile_y != p1_tile_y
                    && line_bottom_y != line_bottom_floor
                    && y_end_idx < tile_rows
                {
                    histogram.add_winding(usize::from(y_end_idx), winding_dir);
                }
                continue;
            }

            let line_idx = line_idx as u32;
            if not_same_tile {
                if line_left_x == line_right_x {
                    // Vertical line case
                    let x = f32_to_u16_sat(line_left_x).min(tile_columns.saturating_sub(1));
                    let x = min_tile_x.max(x);

                    // Process the Top Row (if visible on screen)
                    let is_start_culled = line_top_y < clip_top_tile;
                    if !is_start_culled {
                        let winding = if f32::from(y_top_tiles) >= line_top_y {
                            W
                        } else {
                            0
                        };
                        let intersection_mask = B | winding;
                        self.tiles
                            .push(Tile::new(x, y_top_tiles, line_idx, intersection_mask));
                    }

                    // Process all "fully crossed" tiles (W | T | B).
                    let y_start = if is_start_culled {
                        y_top_tiles
                    } else {
                        y_top_tiles + 1
                    };
                    let y_end_idx = p1_tile_y.min(i32::from(tile_rows));
                    for y_idx in i32::from(y_start)..y_end_idx {
                        let intersection_mask = W | T | B;
                        self.tiles
                            .push(Tile::new(x, y_idx as u16, line_idx, intersection_mask));
                    }

                    // Process the terminal tile (W | T), if it exists. We only emit this if the
                    // line actually terminates on the screen, and if we haven't already
                    // processed/culled it via `y_start`.
                    if (i32::from(y_start)..i32::from(tile_rows)).contains(&p1_tile_y) {
                        let intersection_mask = W | T;
                        self.tiles.push(Tile::new(
                            x,
                            p1_tile_y as u16,
                            line_idx,
                            intersection_mask,
                        ));
                    }
                } else {
                    // General case
                    let dx = p1_x - p0_x;
                    let dy = p1_y - p0_y;
                    let x_slope = dx / dy;

                    // Package for the helper functions, with inlining this should be zero cost.
                    // (In Skia, changing to a class member regresses perf by ~10% on
                    // TilerSortBench.)
                    let ctx = LineContext {
                        line_idx,
                        top_x: line_top_x,
                        top_y: line_top_y,
                        bottom_x: line_bottom_x,
                        bottom_y: line_bottom_y,
                        x_slope,
                        line_left_x,
                        line_right_x,
                        p0_tile_x,
                        p0_tile_y,
                        p1_tile_x,
                        p1_tile_y,
                        tile_columns,
                        winding_dir,
                        clip_left_tile,
                        clip_top_tile,
                        min_tile_x,
                    };

                    let left_crossing = line_left_x < clip_left_tile;
                    let right_crossing = line_right_x >= clip_right_tile;
                    let x_dir = line_bottom_x >= line_top_x;
                    let crosses_edge = left_crossing || right_crossing;

                    if crosses_edge {
                        if left_crossing {
                            culling_event_occurred = true;
                        }

                        if x_dir {
                            self.run_loops::<true, true>(
                                &ctx,
                                line_top_y,
                                line_bottom_y,
                                y_top_tiles,
                                tile_rows,
                                histogram,
                            );
                        } else {
                            self.run_loops::<false, true>(
                                &ctx,
                                line_top_y,
                                line_bottom_y,
                                y_top_tiles,
                                tile_rows,
                                histogram,
                            );
                        }
                    } else if x_dir {
                        self.run_loops::<true, false>(
                            &ctx,
                            line_top_y,
                            line_bottom_y,
                            y_top_tiles,
                            tile_rows,
                            histogram,
                        );
                    } else {
                        self.run_loops::<false, false>(
                            &ctx,
                            line_top_y,
                            line_bottom_y,
                            y_top_tiles,
                            tile_rows,
                            histogram,
                        );
                    }
                }
            } else {
                // Single tile case
                let x_clamped = f32_to_u16_sat(line_left_x).min(tile_columns.saturating_sub(1));
                let x_clamped = min_tile_x.max(x_clamped);
                let winding = if f32::from(y_top_tiles) >= line_top_y {
                    W
                } else {
                    0
                };
                self.tiles
                    .push(Tile::new(x_clamped, y_top_tiles, line_idx, winding));
            }
        }

        culling_event_occurred
    }

    pub(super) fn sort_tiles(&mut self) {
        self.tiles.sort_unstable_by_key(|t| t.to_bits());
    }

    pub(super) fn tiles(&self) -> &[Tile] {
        &self.tiles
    }

    /// Port of `Tiles::pushEdge<kXDir>`: push the leftmost or rightmost tile of a row.
    #[inline(always)]
    fn push_edge<const X_DIR: bool>(
        &mut self,
        ctx: &LineContext,
        x_idx: u16,
        y: u16,
        row_top_x: f32,
        row_bottom_x: f32,
        canonical_start: i32,
        canonical_end: u16,
        winding_input: u32,
        check_start: bool,
        check_end: bool,
    ) {
        // Determine whether this tile is the true start or end of the line within this horizontal
        // row. We need to account for clamping because the line may start or end off-screen (e.g.,
        // X = -5), but we clamp X to the viewport.
        let unc_row_start = (i32::from(x_idx) == canonical_start) as u32;
        let unc_row_end = (x_idx == canonical_end) as u32;

        // Relativize the start/end based on line direction
        let canonical_row_start = if X_DIR { unc_row_start } else { unc_row_end };
        let canonical_row_end = if X_DIR { unc_row_end } else { unc_row_start };

        // Mask out the Top/Bottom bits if this tile contains the line endpoints.
        let mut not_start_tile = 1_u32;
        if check_start {
            not_start_tile ^=
                (i32::from(x_idx) == ctx.p0_tile_x && i32::from(y) == ctx.p0_tile_y) as u32;
        }

        let mut not_end_tile = 1_u32;
        if check_end {
            not_end_tile ^=
                (i32::from(x_idx) == ctx.p1_tile_x && i32::from(y) == ctx.p1_tile_y) as u32;
        }

        let mut mask = winding_input;
        // If this tile is the start of the row, the line must have entered through the Top edge.
        // (Unless it's the line start).
        mask |= canonical_row_start & not_start_tile;
        // If this tile is the end of the row, the line must have exited through the Bottom edge.
        // (Unless it's the line end).
        mask |= (canonical_row_end & not_end_tile) << BOT_SHIFT;

        // If a tile is NOT the start of the row, it must have been entered horizontally. If it is
        // NOT the end, it must have exited horizontally. Base L/R on the direction of the line.
        if X_DIR {
            mask |= (1 ^ canonical_row_start) << LEFT_SHIFT;
            mask |= (1 ^ canonical_row_end) << RIGHT_SHIFT;
        } else {
            mask |= (1 ^ canonical_row_start) << RIGHT_SHIFT;
            mask |= (1 ^ canonical_row_end) << LEFT_SHIFT;
        }

        // Corner handling
        let x_left_f = f32::from(x_idx);
        let x_right_f = (i32::from(x_idx) + 1) as f32;
        let trc = (row_top_x == x_right_f) as u32 & not_start_tile;
        let tlc = (row_top_x == x_left_f) as u32 & not_start_tile;
        let brc = (row_bottom_x == x_right_f) as u32 & not_end_tile;
        let blc = (row_bottom_x == x_left_f) as u32 & not_end_tile;

        // If the line hits the exact Top-Left corner, but it is NOT the canonical start of the row,
        // we must treat it as a Left intersection to properly bridge the mask to the adjacent tile.
        let tie_break = tlc & (canonical_row_start ^ 1);

        // Force corners into into purely horizontal intersections. This makes the downstream
        // intersection calculation logic simpler.
        mask |= (tie_break | blc) << LEFT_SHIFT;
        mask |= (trc | brc) << RIGHT_SHIFT;
        mask &= !(tie_break | trc);
        mask &= !((blc | brc) << BOT_SHIFT);

        self.tiles.push(Tile::new(x_idx, y, ctx.line_idx, mask));
    }

    /// Port of `Tiles::processRow<kXDir, kCrossesEdge>`: push the tiles of one tile row.
    #[inline(always)]
    fn process_row<const X_DIR: bool, const CROSSES_EDGE: bool>(
        &mut self,
        ctx: &LineContext,
        y_idx: u16,
        row_top_x: f32,
        row_bottom_x: f32,
        mut w_mask: u32,
        check_start: bool,
        check_end: bool,
        histogram: &mut WindingHistogram,
    ) {
        let (lx, rx) = if CROSSES_EDGE {
            // Clamp the row's crossings to the extent of the line, since the computed crossings
            // may drift slightly past it.
            let lx = clamp(
                row_top_x.min(row_bottom_x),
                ctx.line_left_x,
                ctx.line_right_x,
            );
            let rx = clamp(
                row_top_x.max(row_bottom_x),
                ctx.line_left_x,
                ctx.line_right_x,
            );

            // When the line crosses the clip's left boundary, the portion of the segment left of
            // the boundary contributes winding but no visible tiles. If the entire segment is left
            // of clip, early-out. Otherwise, clear the winding flag from interior tiles so winding
            // is not double-counted.
            //
            // Note: This tests the clamped crossing at the top of the row rather than `row_top_x`.
            // Otherwise, a line ending exactly on the clip's left edge could drift left of it, and
            // move its winding into the histogram even though:
            //  - It does not cross the left edge, so no culling event is reported for it, and the
            //    histogram is ignored, dropping its winding from the rest of the row.
            //  - Its leftmost tile, classified from `lx` below, still enters through the top (T)
            //    rather than the left (L), so honoring the histogram would count it twice.
            let row_top_cross_x = if X_DIR { lx } else { rx };
            if row_top_cross_x < ctx.clip_left_tile {
                if w_mask & W != 0 {
                    histogram.add_winding(usize::from(y_idx), ctx.winding_dir);
                }

                if rx < ctx.clip_left_tile {
                    return;
                } else {
                    w_mask &= !W;
                }
            }
            (lx, rx)
        } else {
            (row_top_x.min(row_bottom_x), row_top_x.max(row_bottom_x))
        };
        // (Skia computes this in both branches above.)
        let x_end_val = f32_to_u16_sat(rx).min(ctx.tile_columns.saturating_sub(1));

        // Convert floating-point boundaries into discrete integer tile indices. Note:
        // `canonical_x_start` preserves the true start (even if negative) before clamping, which
        // is required by `push_edge` to tie-break in some cases.
        let canonical_x_start = lx.floor() as i32;
        let canonical_x_end = f32_to_u16_sat(rx);
        let x_start = ctx.min_tile_x.max(f32_to_u16_sat(lx));

        if x_start <= x_end_val {
            // Process the Leftmost Tile of the row. If this is the *only* tile in the row,
            // `is_single`, it is both the row start and row end so the Winding (W) bit is passed
            // regardless of direction.
            let is_single = x_start == x_end_val;
            let w_left = (if X_DIR || is_single { W } else { 0 }) & w_mask;
            self.push_edge::<X_DIR>(
                ctx,
                x_start,
                y_idx,
                row_top_x,
                row_bottom_x,
                canonical_x_start,
                canonical_x_end,
                w_left,
                check_start,
                check_end,
            );

            // Process all captive "Middle" tiles in this row. These tiles never have vertical
            // crossings, and for the purpose of the intersection mask, are identical as they
            // always receive [R | L].
            let inner_count = i32::from(x_end_val) - i32::from(x_start) - 1;
            if inner_count > 0 {
                let inner_mask = R | L;
                self.tiles.extend((0..inner_count).map(|i| {
                    Tile::new(
                        (i32::from(x_start) + 1 + i) as u16,
                        y_idx,
                        ctx.line_idx,
                        inner_mask,
                    )
                }));
            }

            // Process the Rightmost Tile of the row. Emitted only if the row spans more than one
            // tile (i.e., we haven't already processed this exact tile as the Leftmost Tile).
            if x_start < x_end_val {
                let w_right = (if X_DIR { 0 } else { W }) & w_mask;
                self.push_edge::<X_DIR>(
                    ctx,
                    x_end_val,
                    y_idx,
                    row_top_x,
                    row_bottom_x,
                    canonical_x_start,
                    canonical_x_end,
                    w_right,
                    check_start,
                    check_end,
                );
            }
        }
    }

    /// Port of `Tiles::runLoops<kXDir, kCrossesEdge>`: push the tiles of a general line, row by
    /// row.
    #[inline(always)]
    fn run_loops<const X_DIR: bool, const CROSSES_EDGE: bool>(
        &mut self,
        ctx: &LineContext,
        line_top_y: f32,
        line_bottom_y: f32,
        y_top_tiles: u16,
        tile_rows: u16,
        histogram: &mut WindingHistogram,
    ) {
        // Process the Top Row (if visible on screen)
        let mut y_start = y_top_tiles;
        let is_start_culled = line_top_y < ctx.clip_top_tile;
        if !is_start_culled {
            let y = f32::from(y_start);
            let row_bottom_y = (y + 1.0).min(line_bottom_y);
            // Catch perfectly horizontal lines and/or prevent floating point drift
            let row_bottom_x = if row_bottom_y == ctx.bottom_y {
                ctx.bottom_x
            } else {
                ctx.top_x + (row_bottom_y - ctx.top_y) * ctx.x_slope
            };
            let mask = if y >= line_top_y { W } else { 0 };
            // The top row might ALSO be the bottom row, so check_end = true
            self.process_row::<X_DIR, CROSSES_EDGE>(
                ctx,
                y_start,
                ctx.top_x,
                row_bottom_x,
                mask,
                /* check_start */ true,
                /* check_end */ true,
                histogram,
            );
            y_start += 1;
        }

        // Process all "Middle" fully crossed rows; the tiles cannot be the start or the end
        let y_end_idx = ctx.p1_tile_y.min(i32::from(tile_rows));
        for y_idx in i32::from(y_start)..y_end_idx {
            let y = y_idx as f32;
            // Although this seems like duplicate calculation, finding the intersections
            // independently allows the entire loop to auto-vectorize (in Skia's C++).
            let row_top_x = ctx.top_x + (y - ctx.top_y) * ctx.x_slope;
            let row_bottom_x = ctx.top_x + (y + 1.0 - ctx.top_y) * ctx.x_slope;
            self.process_row::<X_DIR, CROSSES_EDGE>(
                ctx,
                y_idx as u16,
                row_top_x,
                row_bottom_x,
                0xffff_ffff,
                /* check_start */ false,
                /* check_end */ false,
                histogram,
            );
        }

        // Process the Terminal Row, if it exists. I.e. if it's on-screen AND wasn't already
        // processed as the Top Row.
        if (i32::from(y_start)..i32::from(tile_rows)).contains(&ctx.p1_tile_y) {
            let y = ctx.p1_tile_y as f32;
            // No guard is necessary here against horizontal lines, as a horizontal line would
            // have been processed as a starting row.
            let row_top_x = ctx.top_x + (y - ctx.top_y) * ctx.x_slope;
            // No need to check start (we are past it), but must check end.
            self.process_row::<X_DIR, CROSSES_EDGE>(
                ctx,
                ctx.p1_tile_y as u16,
                row_top_x,
                ctx.bottom_x,
                0xffff_ffff,
                /* check_start */ false,
                /* check_end */ true,
                histogram,
            );
        }
    }
}

/// Port of `Tiles::DivCeil`.
#[inline(always)]
fn div_ceil(a: u16, b: u16) -> u16 {
    a.div_ceil(b)
}

/// Port of `Tiles::f32ToU16Sat`.
#[inline(always)]
fn f32_to_u16_sat(v: f32) -> u16 {
    // `clamp` will catch +/- inf here, but not NaN. However we should never get NaN here.
    debug_assert!(!v.is_nan(), "NaN in the tiler");
    clamp(v, 0.0, 65535.0) as u16
}

/// `std::clamp` for floats. Unlike [`f32::clamp`], this never panics.
#[inline(always)]
fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if hi < v {
        hi
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::super::tests::{XorShift, polygon, rect};
    use super::super::tile::Point;
    use super::*;

    /// The tile grid of a clip, as computed by `make_tiles_msaa`.
    #[derive(Clone, Copy, Debug)]
    struct Grid {
        clip: IRect,
        min_x: u16,
        min_y: u16,
        columns: u16,
        rows: u16,
    }

    impl Grid {
        fn new(clip: IRect) -> Self {
            let ts = u32::from(TILE_SIZE);
            Self {
                clip,
                min_x: (clip.left as u32 / ts) as u16,
                min_y: (clip.top as u32 / ts) as u16,
                columns: (clip.right as u32).div_ceil(ts) as u16,
                rows: (clip.bottom as u32).div_ceil(ts) as u16,
            }
        }

        /// Pixel bounds of the tile grid.
        fn left(&self) -> f32 {
            f32::from(self.min_x * TILE_SIZE)
        }

        fn top(&self) -> f32 {
            f32::from(self.min_y * TILE_SIZE)
        }

        fn bottom(&self) -> f32 {
            f32::from(self.rows * TILE_SIZE)
        }
    }

    const CLIPS: [IRect; 4] = [
        IRect::from_wh(64, 64),
        // Not a multiple of the tile size.
        IRect::from_wh(37, 21),
        IRect::from_wh(100, 75),
        // Tile aligned, but not at the origin.
        IRect {
            left: 32,
            top: 16,
            right: 130,
            bottom: 90,
        },
    ];

    fn range(rng: &mut XorShift, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * rng.next_f32()
    }

    /// Random coordinate, often snapped to the tile or pixel grid to hit the edge cases.
    fn coord(rng: &mut XorShift, lo: f32, hi: f32) -> f32 {
        let v = range(rng, lo, hi);
        match (rng.next_f32() * 4.0) as u32 {
            0 => (v / 16.0).round() * 16.0,
            1 => (v / 4.0).round() * 4.0,
            _ => v,
        }
    }

    fn line(x0: f32, y0: f32, x1: f32, y1: f32) -> Line {
        Line::new(Point::new(x0, y0), Point::new(x1, y1))
    }

    /// A line's endpoints in tile units (exact).
    fn to_tiles(line: &Line) -> (f64, f64, f64, f64) {
        let ts = f64::from(TILE_SIZE);
        (
            f64::from(line.p0.x) / ts,
            f64::from(line.p0.y) / ts,
            f64::from(line.p1.x) / ts,
            f64::from(line.p1.y) / ts,
        )
    }

    /// Generous bound (in tiles) on the tiler's f32 error when it computes where `line` crosses
    /// a tile row edge.
    fn tolerance(line: &Line) -> f64 {
        let (x0, y0, x1, y1) = to_tiles(line);
        let slope = if y0 == y1 {
            0.0
        } else {
            ((x1 - x0) / (y1 - y0)).abs()
        };
        let mx = x0.abs().max(x1.abs());
        let my = y0.abs().max(y1.abs()).max(16.0);
        1e-4 + 2e-6 * (mx + slope * my)
    }

    /// Whether `line` intersects the closed box `[x0, x1] × [y0, y1]` (in tiles). Liang-Barsky.
    fn hits_box(line: &Line, x0: f64, x1: f64, y0: f64, y1: f64) -> bool {
        let (ax, ay, bx, by) = to_tiles(line);
        let (dx, dy) = (bx - ax, by - ay);
        let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
        for (p, q) in [(-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)] {
            if p == 0.0 {
                if q < 0.0 {
                    return false;
                }
            } else if p < 0.0 {
                t0 = t0.max(q / p);
            } else {
                t1 = t1.min(q / p);
            }
        }
        t0 <= t1
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Kind {
        /// Inside or crossing the clip.
        Near,
        /// Long lines, mostly crossing the clip.
        Far,
        /// Coordinates up to 1e30.
        Huge,
        /// Vertical or horizontal.
        Axis,
        /// Entirely above the tile grid.
        Above,
        /// Entirely below the tile grid.
        Below,
        /// Entirely right of the clip.
        Right,
        /// Entirely left of the tile grid.
        Left,
    }

    const KINDS: [Kind; 8] = [
        Kind::Near,
        Kind::Far,
        Kind::Huge,
        Kind::Axis,
        Kind::Above,
        Kind::Below,
        Kind::Right,
        Kind::Left,
    ];

    fn random_segment(rng: &mut XorShift, g: &Grid, kind: Kind) -> Line {
        let (l, t) = (g.clip.left as f32, g.clip.top as f32);
        let (r, b) = (g.clip.right as f32, g.clip.bottom as f32);
        loop {
            let x0 = coord(rng, l - 40.0, r + 40.0);
            let x1 = coord(rng, l - 40.0, r + 40.0);
            let y0 = coord(rng, t - 40.0, b + 40.0);
            let y1 = coord(rng, t - 40.0, b + 40.0);
            let seg = match kind {
                Kind::Near => line(x0, y0, x1, y1),
                Kind::Far => line(
                    range(rng, -1e4, 1e4),
                    range(rng, -1e4, 1e4),
                    x0,
                    range(rng, -1e4, 1e4),
                ),
                Kind::Huge => {
                    const HUGE: [f32; 6] = [-1e30, -1e15, -1e7, 1e7, 1e15, 1e30];
                    let mut pick = |v: f32| {
                        if rng.next_f32() < 0.4 {
                            HUGE[(rng.next_f32() * 6.0) as usize]
                        } else {
                            v
                        }
                    };
                    let (x0, y0) = (pick(x0), pick(y0));
                    line(x0, y0, pick(x1), pick(y1))
                }
                Kind::Axis => {
                    if rng.next_f32() < 0.5 {
                        line(x0, y0, x0, y1)
                    } else {
                        line(x0, y0, x1, y0)
                    }
                }
                Kind::Above => {
                    let y0 = g.top() - range(rng, 0.0, 300.0);
                    let y1 = g.top() - range(rng, 0.0, 300.0);
                    line(x0, y0, x1, y1)
                }
                Kind::Below => {
                    let y0 = g.bottom() + range(rng, 0.0, 300.0);
                    let y1 = g.bottom() + range(rng, 0.0, 300.0);
                    line(x0, y0, x1, y1)
                }
                Kind::Right => {
                    let x0 = r + range(rng, 0.01, 300.0);
                    let x1 = r + range(rng, 0.01, 300.0);
                    line(x0, y0, x1, y1)
                }
                Kind::Left => {
                    let x0 = g.left() - range(rng, 0.01, 300.0);
                    let x1 = g.left() - range(rng, 0.01, 300.0);
                    line(x0, y0, x1, y1)
                }
            };
            if seg.p0 != seg.p1 {
                return seg;
            }
        }
    }

    /// Tiles are inside the clip's tile grid, every cell a segment passes through (within the
    /// clip) gets a tile for it, and segments outside the clip produce none.
    #[test]
    fn tiles_cover_segments() {
        let mut rng = XorShift(0x2545_f491_4f6c_dd1d);
        let mut tiles = Tiles::new();
        let mut covered_cells = 0;
        for clip in CLIPS {
            let g = Grid::new(clip);
            let mut histogram = WindingHistogram::new(usize::from(g.rows));
            for _ in 0..400 {
                let n = 1 + (rng.next_f32() * 8.0) as usize;
                let segs: Vec<(Kind, Line)> = (0..n)
                    .map(|_| {
                        let kind = KINDS[(rng.next_f32() * KINDS.len() as f32) as usize];
                        (kind, random_segment(&mut rng, &g, kind))
                    })
                    .collect();
                let lines: Vec<Line> = segs.iter().map(|s| s.1).collect();
                tiles.reset();
                histogram.clear();
                let culled = tiles.make_tiles_msaa(&lines, clip, &mut histogram);

                let mut seen = HashSet::new();
                for t in tiles.tiles() {
                    assert!(
                        (g.min_x..g.columns).contains(&t.x) && (g.min_y..g.rows).contains(&t.y),
                        "tile outside the grid: {t:?}, {clip:?}, {lines:?}"
                    );
                    let k = t.line_idx() as usize;
                    assert!(k < lines.len(), "bad line index: {t:?}");
                    assert!(
                        seen.insert((t.x, t.y, k)),
                        "duplicate tile: {t:?}, {:?}",
                        lines[k]
                    );
                    let (kind, seg) = segs[k];
                    assert!(
                        !matches!(kind, Kind::Above | Kind::Below | Kind::Right | Kind::Left),
                        "{kind:?} segment produced a tile: {t:?}, {seg:?}, {clip:?}"
                    );
                    // Corners are forced into horizontal intersections, so a line touches at most
                    // two edges of a tile. (This relies on the computed crossings being accurate to
                    // well within a tile, which doesn't hold for huge coordinates.)
                    if matches!(kind, Kind::Near | Kind::Axis) {
                        assert!(
                            (t.intersection_mask() & (T | B | L | R)).count_ones() <= 2,
                            "more than two edge intersections: {t:?}, {seg:?}"
                        );
                    }
                    // The segment touches the tile (edge and corner touches count).
                    let (c, r) = (f64::from(t.x), f64::from(t.y));
                    let eps = tolerance(&seg);
                    assert!(
                        hits_box(&seg, c - eps, c + 1.0 + eps, r - eps, r + 1.0 + eps),
                        "tile not touched by its segment: {t:?}, {seg:?}"
                    );
                }

                let ts = f64::from(TILE_SIZE);
                for (k, (kind, seg)) in segs.iter().enumerate() {
                    let eps = tolerance(seg);
                    for r in g.min_y..g.rows {
                        for c in g.min_x..g.columns {
                            // The part of the cell inside the clip, shrunk by the tolerance.
                            let (cf, rf) = (f64::from(c), f64::from(r));
                            let x0 = cf.max(f64::from(clip.left) / ts) + eps;
                            let x1 = (cf + 1.0).min(f64::from(clip.right) / ts) - eps;
                            let y0 = rf.max(f64::from(clip.top) / ts) + eps;
                            let y1 = (rf + 1.0).min(f64::from(clip.bottom) / ts) - eps;
                            if x0 <= x1 && y0 <= y1 && hits_box(seg, x0, x1, y0, y1) {
                                covered_cells += 1;
                                assert!(
                                    seen.contains(&(c, r, k)),
                                    "missing tile ({c}, {r}) for {kind:?} {seg:?}, {clip:?}"
                                );
                            }
                        }
                    }
                    // A line entirely left of the grid that spans a tile row is a culling event.
                    let (y_min, y_max) = (seg.p0.y.min(seg.p1.y), seg.p0.y.max(seg.p1.y));
                    if *kind == Kind::Left
                        && y_min != y_max
                        && y_max > g.top()
                        && y_min < g.bottom()
                    {
                        assert!(culled, "no culling event for {seg:?}, {clip:?}");
                    }
                }
                if segs
                    .iter()
                    .all(|s| matches!(s.0, Kind::Above | Kind::Below | Kind::Right))
                {
                    assert!(!culled, "culling event without lines left of the clip");
                }
                if !culled {
                    assert!(
                        histogram.data().iter().all(|&w| w == 0),
                        "histogram written without a culling event"
                    );
                }
            }
        }
        assert!(covered_cells > 1000, "test is vacuous: {covered_cells}");
    }

    /// Winding of `lines` along the top edge of tile row `r`, from crossings strictly left of
    /// `x = c` (in tiles). Lines are half-open in y (`top <= r < bottom`). This is what the
    /// coarse winding at the top left corner of tile `(c, r)` must be. Returns `None` if a
    /// computed crossing is too close to `c` to call.
    fn expected_winding(lines: &[Line], r: u16, c: f64) -> Option<i32> {
        let r = f64::from(r);
        let mut winding = 0;
        for line in lines {
            let (x0, y0, x1, y1) = to_tiles(line);
            let (tx, ty, bx, by, dir) = if y0 < y1 {
                (x0, y0, x1, y1, 1)
            } else {
                (x1, y1, x0, y0, -1)
            };
            if !(ty..by).contains(&r) {
                continue;
            }
            // Crossings at endpoints and on vertical lines are exact.
            let x = if ty == r || tx == bx {
                tx
            } else {
                let x = tx + (r - ty) * (bx - tx) / (by - ty);
                if (x - c).abs() <= tolerance(line) {
                    return None;
                }
                x
            };
            if x < c {
                winding += dir;
            }
        }
        Some(winding)
    }

    /// The coarse winding at the top left corner of tile `(c, r)`, as accumulated by the strip
    /// processor: the histogram plus the W tiles left of `c`.
    fn coarse_winding(
        tiles: &[Tile],
        lines: &[Line],
        histogram: &WindingHistogram,
        c: u16,
        r: u16,
    ) -> i32 {
        let mut winding = i32::from(histogram.get(usize::from(r)));
        for t in tiles {
            if t.y == r && t.x < c && t.coarse_winding() {
                let line = lines[t.line_idx() as usize];
                winding += if line.p0.y < line.p1.y { 1 } else { -1 };
            }
        }
        winding
    }

    /// For closed polygons partly left of the clip, the histogram holds the winding of the
    /// geometry left of the clip at the top edge of each tile row, and together with the W bits
    /// gives the right coarse winding for every tile.
    #[test]
    fn histogram_and_coarse_winding() {
        let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
        let mut tiles = Tiles::new();
        let (mut checked_rows, mut nonzero_rows) = (0, 0);
        for clip in CLIPS {
            let g = Grid::new(clip);
            let (l, r) = (clip.left as f32, clip.right as f32);
            let mut histogram = WindingHistogram::new(usize::from(g.rows));
            for i in 0..300 {
                // One to three closed polygons, reaching up to 120 px left of the clip.
                let mut lines = Vec::new();
                for _ in 0..1 + i % 3 {
                    let n = 3 + (rng.next_f32() * 10.0) as usize;
                    let pts: Vec<(f32, f32)> = (0..n)
                        .map(|_| {
                            let x = if rng.next_f32() < 0.05 {
                                -1e4
                            } else {
                                coord(&mut rng, l - 120.0, r + 40.0)
                            };
                            (x, coord(&mut rng, g.top() - 30.0, g.bottom() + 30.0))
                        })
                        .collect();
                    lines.extend(polygon(&pts));
                }
                tiles.reset();
                histogram.clear();
                let culled = tiles.make_tiles_msaa(&lines, clip, &mut histogram);
                tiles.sort_tiles();
                assert!(
                    tiles.tiles().is_sorted_by_key(|t| t.to_bits()),
                    "tiles not sorted"
                );
                if !culled {
                    assert!(
                        histogram.data().iter().all(|&w| w == 0),
                        "histogram written without a culling event"
                    );
                }

                for row in 0..g.rows {
                    if row < g.min_y {
                        assert_eq!(histogram.get(usize::from(row)), 0, "row above the clip");
                        continue;
                    }
                    if let Some(w) = expected_winding(&lines, row, f64::from(g.min_x)) {
                        assert_eq!(
                            i32::from(histogram.get(usize::from(row))),
                            w,
                            "histogram row {row}, {clip:?}, {lines:?}"
                        );
                        checked_rows += 1;
                        nonzero_rows += usize::from(w != 0);
                    }
                    for c in g.min_x..g.columns {
                        if let Some(w) = expected_winding(&lines, row, f64::from(c)) {
                            assert_eq!(
                                coarse_winding(tiles.tiles(), &lines, &histogram, c, row),
                                w,
                                "coarse winding at ({c}, {row}), {clip:?}, {lines:?}"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            checked_rows > 2000 && nonzero_rows > 500,
            "test is vacuous: {checked_rows} rows, {nonzero_rows} nonzero"
        );
    }

    /// Exact output for a rectangle, pinning Skia's edge conventions.
    #[test]
    fn rect_tiles() {
        // Clockwise in y-down coordinates: top, right, bottom, left.
        let lines = rect(16.0, 16.0, 48.0, 40.0);
        let mut tiles = Tiles::new();
        let mut histogram = WindingHistogram::new(4);
        let culled = tiles.make_tiles_msaa(&lines, IRect::from_wh(64, 64), &mut histogram);
        assert!(!culled, "nothing is left of the viewport");
        tiles.sort_tiles();
        let got: Vec<_> = tiles
            .tiles()
            .iter()
            .map(|t| (t.x, t.y, t.line_idx(), t.intersection_mask()))
            .collect();
        // The top edge lies on a tile row boundary, so it has no tiles. The vertical edges cross
        // the top of row 1 (W) and end in row 2. The bottom edge is horizontal inside row 2.
        let expected = [
            (1, 1, 3, B | W),
            (3, 1, 1, B | W),
            (1, 2, 2, R),
            (1, 2, 3, T | W),
            (2, 2, 2, L | R),
            (3, 2, 1, T | W),
            (3, 2, 2, L),
        ];
        assert_eq!(got, expected, "rect tiles");
    }

    /// A line left of the viewport goes into the histogram, with winding at each row it crosses
    /// the top edge of.
    #[test]
    fn left_of_viewport_histogram() {
        // Clockwise: the left edge (line 3), at x = -100, goes up from y = 44 to y = 4.
        let lines = rect(-100.0, 4.0, 20.0, 44.0);
        let mut tiles = Tiles::new();
        let mut histogram = WindingHistogram::new(3);
        let culled = tiles.make_tiles_msaa(&lines, IRect::from_wh(64, 48), &mut histogram);
        assert!(culled, "a line is left of the viewport");
        assert_eq!(histogram.data(), [0, -1, -1], "histogram");
        assert!(
            tiles.tiles().iter().all(|t| t.line_idx() != 3),
            "culled line produced tiles"
        );
        // Each row's coarse winding is -1 (inside) between the left edge and the right edge.
        for row in 1..3 {
            assert_eq!(
                coarse_winding(tiles.tiles(), &lines, &histogram, 1, row),
                -1,
                "inside"
            );
            assert_eq!(
                coarse_winding(tiles.tiles(), &lines, &histogram, 2, row),
                0,
                "outside"
            );
        }
    }
}
