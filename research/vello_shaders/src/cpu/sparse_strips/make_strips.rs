// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Port of Skia's `MakeStrips::TraverseCPU` plus the per-path driver from `StripGenerator.cpp`.
//!
//! At this point in the sparse strips pipeline, the path has been flattened into lines, tiled,
//! and sorted. Now, the tiles are consumed by [`traverse_cpu`] to produce:
//!
//! 1. Mask tiles: rasterized boundary tiles with per-pixel sample masks (Skia's `EndCaps`, whose
//!    coverage is stored in the alpha atlas), pushed with [`StripSink::push_mask_tile`].
//! 2. Solid spans: contiguous interior regions with full coverage, which are rendered without
//!    masks (Skia's `WideTiles`), pushed with [`StripSink::add_wide`].
//!
//! Two things are required here:
//!
//! 1. Coverage resolution: multiple line segments often intersect the exact same spatial tile.
//!    Because the input tiles are generated per line segment, the individual winding
//!    contributions must be combined to produce the final coverage mask for that location:
//!
//!    ```text
//!    Line 1 (\)       Line 2 (/)        Combined Mask (V)
//!    +----------+     +----------+         +----------+
//!    | \        |     |        / |         | \      / |
//!    |  \       |  +  |       /  |    =    |  \    /  |
//!    |███\      |     |      /███|         |███\  /███|
//!    |████\     |     |     /████|         |████\/████|
//!    +----------+     +----------+         +----------+
//!    ```
//!
//! 2. Geometry generation: because the incoming tiles are sorted by y, then x, runs of contiguous
//!    tiles identify boundary tiles, while gaps between runs are checked against the winding fill
//!    rule to identify solid interior spans.
//!
//!    ```text
//!    0           1           2           3           4           5
//!    +-----------+-----------+-----------+-----------+-----------+-----------+
//!    |           |     /     |███████████|███████████|     \     |           |
//!    |  Outside  |   /       |██ Solid ██|██ Solid ██|       \   |  Outside  |
//!    | (No Fill) | /  Mask   |██ Span  ██|██ Span  ██|   Mask  \ | (No Fill) |
//!    |           |   Tile    |██ (100%)██|██ (100%)██|   Tile    |           |
//!    +-----------+-----------+-----------+-----------+-----------+-----------+
//!                ^           ^                       ^           ^
//!                |           |                       |           |
//!                +-- Mask ---+<----- Solid span ---->+-- Mask ---+
//!                   [x: 1]          [x: 2, w: 2]        [x: 4]
//!    ```
//!
//! Unlike Skia, boundary tiles are not batched into contiguous `EndCap` runs backed by an alpha
//! atlas (`FinalizeRun`, `AlphaAtlasManager`): the masks of each tile are resolved into
//! [`StripSink::scratch_mut`] and pushed on their own. Inverse fill types are not supported, so
//! Skia's `isInverse` is always false.

use super::TILE_SIZE;
use super::lut::msaa_lut;
use super::processor::StripProcessor;
use super::sink::StripSink;
use super::tile::{IRect, Line, Point, Tile, WindingHistogram};
use super::tiler::Tiles;

const TILE_WIDTH: u32 = TILE_SIZE as u32;
const TILE_HEIGHT: u32 = TILE_SIZE as u32;

/// Per-scene rasterizer state, reused across paths (histogram, tile buffer).
///
/// Port of the per-path parts of Skia's `StripGenerator`.
#[derive(Debug)]
pub(super) struct Rasterizer {
    width: u32,
    height: u32,
    /// Winding contributed by lines culled to the left of the clip, per tile row.
    winding_histogram: WindingHistogram,
    /// Whether the tiler reported a culling event for the previous path, in which case
    /// `winding_histogram` is dirty and must be cleared before it is reused. Skia's `fCulled`.
    culled: bool,
    /// The tiles of the current path.
    tiles: Tiles,
}

impl Rasterizer {
    pub(super) fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            winding_histogram: WindingHistogram::new(height.div_ceil(TILE_HEIGHT) as usize),
            culled: true,
            tiles: Tiles::new(),
        }
    }

    /// Rasterize one path's lines into `sink`, clipped to the viewport.
    ///
    /// Port of `StripGenerator::processGeometry` from tiling onward (the front end has already
    /// flattened the path), with the viewport as the clip.
    pub(super) fn rasterize_path(&mut self, lines: &[Line], even_odd: bool, sink: &mut StripSink) {
        // Skia intersects the clip with the viewport, then rounds its left and top down to whole
        // tiles. Here the clip is the viewport, so both are already 0.
        let tile_clip = IRect::from_wh(self.width as i32, self.height as i32);
        if tile_clip.is_empty() || lines.is_empty() {
            return;
        }

        self.winding_histogram
            .resize(self.height.div_ceil(TILE_HEIGHT) as usize);
        if self.culled {
            self.winding_histogram.clear();
        }
        // The tiler only writes to the histogram when it reports a culling event, so if the
        // previous path did not cull, the histogram must still be zero from the last time it was
        // cleared.
        debug_assert!(
            self.winding_histogram.data().iter().all(|&w| w == 0),
            "winding histogram must be zero before tiling"
        );

        self.tiles.reset();
        self.culled = self
            .tiles
            .make_tiles_msaa(lines, tile_clip, &mut self.winding_histogram);
        self.tiles.sort_tiles();

        msaa_simd(
            self.tiles.tiles(),
            sink,
            even_odd,
            lines,
            self.culled,
            &self.winding_histogram,
            tile_clip,
        );
    }
}

/// Port of `MakeStrips::MsaaSimd`, including `Dispatch` on the fill rule (without the inverse
/// fill types).
fn msaa_simd(
    tiles: &[Tile],
    sink: &mut StripSink,
    even_odd: bool,
    lines: &[Line],
    is_culled: bool,
    winding_histogram: &WindingHistogram,
    clip_rect: IRect,
) {
    let mask_lut = msaa_lut();
    if even_odd {
        let mut processor = StripProcessor::<'_, false>::new(lines, mask_lut);
        traverse_cpu(
            tiles,
            sink,
            is_culled,
            winding_histogram,
            clip_rect,
            &mut processor,
        );
    } else {
        let mut processor = StripProcessor::<'_, true>::new(lines, mask_lut);
        traverse_cpu(
            tiles,
            sink,
            is_culled,
            winding_histogram,
            clip_rect,
            &mut processor,
        );
    }
}

/// The top left and bottom right corners of `tile` in device space.
#[inline(always)]
fn tile_bounds(tile: Tile) -> [Point; 2] {
    let size = f32::from(TILE_SIZE);
    let x = f32::from(tile.x) * size;
    let y = f32::from(tile.y) * size;
    [Point::new(x, y), Point::new(x + size, y + size)]
}

/// The winding at the left edge of the clip in tile row `y`, from lines culled to the left of
/// the clip.
#[inline(always)]
fn row_winding(is_culled: bool, winding_histogram: &WindingHistogram, y: u16) -> i32 {
    if is_culled {
        i32::from(winding_histogram.get(usize::from(y)))
    } else {
        0
    }
}

/// Port of `MakeStrips::ProcessEmptyTileRows`: tile rows `start_tile_y..end_tile_y` contain no
/// tiles, so they are either entirely inside (per the winding of the lines culled to the left of
/// the clip) or entirely outside.
fn process_empty_tile_rows<const IS_WINDING: bool>(
    start_tile_y: u32,
    end_tile_y: u32,
    clip_rect: IRect,
    is_culled: bool,
    winding_histogram: &WindingHistogram,
    processor: &StripProcessor<'_, IS_WINDING>,
    sink: &mut StripSink,
) {
    if start_tile_y >= end_tile_y {
        return;
    }

    let start_x = clip_rect.left as u32;
    let width = clip_rect.width() as u32;

    for y in start_tile_y..end_tile_y {
        let w = if is_culled && (y as usize) < winding_histogram.len() {
            winding_histogram.get(y as usize)
        } else {
            0
        };
        if processor.should_fill(i32::from(w)) {
            sink.add_wide(start_x, y * TILE_HEIGHT, width);
        }
    }
}

/// Port of `MakeStrips::TraverseCPU`.
///
/// While the underlying implementation may be scalar or SIMD, the core traversal across the
/// tiles is identical. To reiterate, the goal is twofold:
/// 1. Combine line segments at the same spatial tile to produce the final coverage.
/// 2. Generate mask tiles for antialiased boundary tiles and solid spans for interior fills.
///
/// To do this in a single pass, the traversal treats the sorted tile stream as a state machine
/// governed by three transition events:
///
/// 1. Tile start (`tile_start`): triggered when the current tile's x or y differs from the
///    previous tile. All overlapping segments at the previous spatial coordinate have been
///    processed, so the accumulated winding is resolved into sample masks and pushed to the
///    sink. If the new tile is on the same row, it is seeded with the carried coarse winding.
/// 2. Segment start (`seg_start`): triggered by a `row_start`, or when the current tile's x
///    coordinate skips forward by more than 1 (a non-contiguous gap in the same row). Then:
///    1. Skia finalizes the preceding contiguous boundary run here, emitting an `EndCap`. Here,
///       every tile was already pushed on its own.
///    2. If the coarse winding indicates an interior fill, a solid span covering the gap up to
///       the current tile is emitted.
///    3. If `row_start`, the previous row is closed out (emitting a trailing fill up to the right
///       of the clip if needed), any intervening empty tile rows are processed via the winding
///       histogram, the coarse winding of the new row is seeded from the winding histogram, any
///       leading fill from the left of the clip is emitted, and the new row begins.
fn traverse_cpu<const IS_WINDING: bool>(
    tiles: &[Tile],
    sink: &mut StripSink,
    is_culled: bool,
    winding_histogram: &WindingHistogram,
    clip_rect: IRect,
    processor: &mut StripProcessor<'_, IS_WINDING>,
) {
    if clip_rect.is_empty() || clip_rect.right <= 0 || clip_rect.bottom <= 0 {
        return;
    }

    let min_tile_y = clip_rect.top.max(0) as u32 / TILE_HEIGHT;
    let mut total_rows = (clip_rect.bottom as u32).div_ceil(TILE_HEIGHT);
    if is_culled {
        total_rows = total_rows.min(winding_histogram.len() as u32);
    }

    let Some(&first_tile) = tiles.first() else {
        process_empty_tile_rows(
            min_tile_y,
            total_rows,
            clip_rect,
            is_culled,
            winding_histogram,
            processor,
            sink,
        );
        return;
    };

    let mut prev_tile = first_tile;

    process_empty_tile_rows(
        min_tile_y,
        u32::from(prev_tile.y),
        clip_rect,
        is_culled,
        winding_histogram,
        processor,
        sink,
    );

    let winding_delta = row_winding(is_culled, winding_histogram, prev_tile.y);
    processor.set_coarse_winding(winding_delta);
    processor.clear_with_coarse_winding();

    let fill_left_start = processor.should_fill(winding_delta);
    if fill_left_start {
        let wide_start_x = clip_rect.left as u32;
        let wide_end_x = u32::from(prev_tile.x) * TILE_WIDTH;
        if wide_end_x > wide_start_x {
            sink.add_wide(
                wide_start_x,
                u32::from(prev_tile.y) * TILE_HEIGHT,
                wide_end_x - wide_start_x,
            );
        }
    }

    let mut bounds = tile_bounds(prev_tile);

    for &tile in tiles {
        // Determine tile traversal events.
        let row_start = tile.y != prev_tile.y;
        let tile_start = tile.x != prev_tile.x || row_start;
        let seg_start =
            tile_start && (row_start || u32::from(tile.x) != u32::from(prev_tile.x) + 1);

        if tile_start {
            // Moving to a new tile implies that all of the previous tile's coverage has been
            // combined: resolve the winding to sample masks, then clear it.
            processor.resolve_masks(sink.scratch_mut());
            sink.push_mask_tile(u32::from(prev_tile.x), u32::from(prev_tile.y));
            if !row_start {
                // If we're not a row start, carry the scanline winding by seeding the coverage
                // mask with the coarse winding.
                processor.clear_with_coarse_winding();
            }
        }

        if seg_start {
            // 1. Skia finalizes the contiguous `EndCap` run here.
            let run_end_x = (u32::from(prev_tile.x) + 1) * TILE_WIDTH;

            // 2. If the winding is inside, emit the solid interior span.
            let should_fill = processor.should_fill(processor.coarse_winding());
            if should_fill && !row_start {
                let wide_end_x = u32::from(tile.x) * TILE_WIDTH;
                if wide_end_x > run_end_x {
                    sink.add_wide(
                        run_end_x,
                        u32::from(prev_tile.y) * TILE_HEIGHT,
                        wide_end_x - run_end_x,
                    );
                }
            }

            // 3. Handle row breaks.
            if row_start {
                if should_fill {
                    let clip_right = clip_rect.right as u32;
                    if clip_right > run_end_x {
                        sink.add_wide(
                            run_end_x,
                            u32::from(prev_tile.y) * TILE_HEIGHT,
                            clip_right - run_end_x,
                        );
                    }
                }

                // Process intervening empty rows.
                process_empty_tile_rows(
                    u32::from(prev_tile.y) + 1,
                    u32::from(tile.y),
                    clip_rect,
                    is_culled,
                    winding_histogram,
                    processor,
                    sink,
                );

                // Reset and seed the coarse winding for the new row.
                let winding_delta = row_winding(is_culled, winding_histogram, tile.y);
                processor.set_coarse_winding(winding_delta);
                processor.clear_with_coarse_winding();

                let fill_left_new_row = processor.should_fill(winding_delta);
                if fill_left_new_row {
                    let wide_start_x = clip_rect.left as u32;
                    let wide_end_x = u32::from(tile.x) * TILE_WIDTH;
                    if wide_end_x > wide_start_x {
                        sink.add_wide(
                            wide_start_x,
                            u32::from(tile.y) * TILE_HEIGHT,
                            wide_end_x - wide_start_x,
                        );
                    }
                }
            }

            // 4. Skia starts a new contiguous alpha run here.
        }

        prev_tile = tile;

        // Lazily recalculate the tile bounds only if we have moved to a new tile.
        if tile_start {
            bounds = tile_bounds(tile);
        }

        processor.rasterize_line_to_tile(tile, bounds);
    }

    // Process the last tile and finalize.
    processor.resolve_masks(sink.scratch_mut());
    sink.push_mask_tile(u32::from(prev_tile.x), u32::from(prev_tile.y));

    let should_fill = processor.should_fill(processor.coarse_winding());
    if should_fill {
        let run_end_x = (u32::from(prev_tile.x) + 1) * TILE_WIDTH;
        let clip_right = clip_rect.right as u32;
        if clip_right > run_end_x {
            sink.add_wide(
                run_end_x,
                u32::from(prev_tile.y) * TILE_HEIGHT,
                clip_right - run_end_x,
            );
        }
    }

    process_empty_tile_rows(
        u32::from(prev_tile.y) + 1,
        total_rows,
        clip_rect,
        is_culled,
        winding_histogram,
        processor,
        sink,
    );
}

#[cfg(test)]
mod tests {
    use super::super::reference::rasterize_path_reference;
    use super::super::tests::{decode, polygon, rect};
    use super::*;

    /// Counter-clockwise (in y-down) rectangle: winding +1 inside.
    fn ccw_rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Line> {
        polygon(&[(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
    }

    /// Traverse hand-built tiles, standing in for the tiler: `culled` lines contribute only
    /// through `histogram` (as if the tiler had culled them to the left of the clip), and each
    /// shape lies strictly inside its tile, so it is tiled as one tile per line without
    /// intersections.
    ///
    /// Returns the decoded output, and the decoded output of the reference rasterizer for all
    /// lines.
    fn traverse_hand_tiled(
        culled: &[Line],
        histogram: &[i16],
        shapes: &[(u16, u16, Vec<Line>)],
        even_odd: bool,
        width: u32,
        height: u32,
    ) -> (Vec<u16>, Vec<u16>) {
        let mut lines = culled.to_vec();
        let mut tiles = Vec::new();
        for (tx, ty, shape) in shapes {
            for line in shape {
                tiles.push(Tile::new(*tx, *ty, lines.len() as u32, 0));
                lines.push(*line);
            }
        }
        tiles.sort_unstable_by_key(|t| t.to_bits());
        let mut winding_histogram = WindingHistogram::new(height.div_ceil(TILE_HEIGHT) as usize);
        for (row, &w) in histogram.iter().enumerate() {
            winding_histogram.add_winding(row, w);
        }
        let clip = IRect::from_wh(width as i32, height as i32);

        let mut sink = StripSink::new(width.div_ceil(TILE_WIDTH));
        sink.begin_path(0);
        let is_culled = !culled.is_empty();
        msaa_simd(
            &tiles,
            &mut sink,
            even_odd,
            &lines,
            is_culled,
            &winding_histogram,
            clip,
        );
        let got = decode(&sink.records, &sink.masks, 0, width, height);

        let mut sink = StripSink::new(width.div_ceil(TILE_WIDTH));
        sink.begin_path(0);
        rasterize_path_reference(&lines, even_odd, width, height, &mut sink);
        let expected = decode(&sink.records, &sink.masks, 0, width, height);
        (got, expected)
    }

    fn assert_same_in_viewport(got: &[u16], expected: &[u16], width: u32, height: u32) {
        let stride = (width.div_ceil(TILE_WIDTH) * TILE_WIDTH) as usize;
        for y in 0..height as usize {
            for x in 0..width as usize {
                let (g, e) = (got[y * stride + x], expected[y * stride + x]);
                assert_eq!(g, e, "pixel ({x}, {y}): {g:#06x} vs reference {e:#06x}");
            }
        }
    }

    #[test]
    fn traversal_matches_reference() {
        // A ragged viewport: 5 x 6 tiles, the last column and row partial.
        let (width, height) = (72, 92);
        let shapes = [
            // Row 0: outside the culled region, a clockwise (winding -1) rect.
            (2, 0, rect(35.0, 3.0, 45.5, 13.25)),
            // Rows 1 and 2: holes in the culled region, with a gap between tiles 1 and 3, and a
            // hole in the partial last column.
            (1, 1, ccw_rect(18.5, 19.0, 28.0, 30.5)),
            (3, 1, ccw_rect(50.25, 17.0, 60.0, 29.75)),
            (4, 2, ccw_rect(65.0, 36.0, 70.5, 44.0)),
            // Row 3: no tiles, entirely inside the culled region.
            // Row 4: outside the culled region again, two adjacent tiles.
            (0, 4, ccw_rect(2.0, 66.0, 12.5, 74.0)),
            (1, 4, ccw_rect(16.5, 64.5, 31.75, 79.0)),
            // Row 5: no tiles, outside.
        ];
        // A rect spanning tile rows 1 to 3, whose left edge is left of the viewport and right
        // edge right of it: it contributes winding -1 to each pixel of those rows.
        let culled = rect(-8.0, 16.0, width as f32 + 8.0, 64.0);
        let histogram = [0, -1, -1, -1, 0, 0];
        for even_odd in [false, true] {
            let (got, expected) =
                traverse_hand_tiled(&culled, &histogram, &shapes, even_odd, width, height);
            assert_same_in_viewport(&got, &expected, width, height);
            // Not culled: only the shapes themselves.
            let (got, expected) = traverse_hand_tiled(&[], &[], &shapes, even_odd, width, height);
            assert_same_in_viewport(&got, &expected, width, height);
        }
    }

    #[test]
    fn traversal_of_nested_culled_windings() {
        // Two culled rects overlapping in tile row 1: winding -2 there, which nonzero fills and
        // even-odd leaves empty.
        let (width, height) = (48, 48);
        let mut culled = rect(-8.0, 0.0, 56.0, 32.0);
        culled.extend(rect(-4.0, 16.0, 56.0, 48.0));
        let histogram = [-1, -2, -1];
        let shapes = [(1, 1, ccw_rect(20.0, 20.5, 27.5, 28.0))];
        for even_odd in [false, true] {
            let (got, expected) =
                traverse_hand_tiled(&culled, &histogram, &shapes, even_odd, width, height);
            assert_same_in_viewport(&got, &expected, width, height);
        }
    }
}
