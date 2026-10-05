// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Port of Skia's `Tile::ClipToTile` (`Tiler.h`), at 16×16 tiles.

use super::TILE_SIZE;
use super::tile::{Line, Point, Tile};

impl Tile {
    /// Port of `Tile::ClipToTile`.
    ///
    /// Responsible for consuming the tile coordinates and intersection mask produced by the tiler
    /// and producing the exact points at which the parent line intersects the edges of the tile.
    /// It also safely handles when there is only one edge intersection, either from the line
    /// ending inside the tile body, or a horizontal edge touch.
    ///
    /// `tile_bounds` are the top left and bottom right corners of the tile in device space, and
    /// `derivatives` are `[dx, dy, 1 / dx, 1 / dy]` of `line`. The returned line is in tile-local
    /// coordinates (`[0, 16]²`), followed by whether its top and bottom points are on the left edge
    /// of the tile.
    ///
    /// This function provides the following guarantees:
    ///  1. Y-sorting: the resulting points are always sorted top-to-bottom.
    ///  2. Interior endpoints: if a line ends inside the tile, its exact endpoint is preserved and
    ///     returned.
    ///  3. Zero-width degenerates: a single-point intersection (e.g., a horizontal graze or corner
    ///     touch) produces a second point along that exact same edge, yielding safe, zero-width
    ///     geometry.
    ///  4. Boundary snapping: edge intersections are explicitly snapped to the exact bounding
    ///     coordinate of the tile.
    ///
    /// (Note: guarantees 2 and 4 ensure downstream logic can safely rely on floating-point
    /// equality checks.)
    ///
    /// Note, this function is tightly coupled to the tiler and `MakeStrips`. If the incoming tile
    /// coordinates are incorrect, the results from this function are meaningless. It assumes that:
    ///  1. Derivatives resulting from division by zero (e.g. perfectly vertical or horizontal
    ///     lines) will be set to zero.
    ///  2. Only two bits are ever set in the mask. (Expects that perfect corner touches are tie
    ///     broken by the tiler.)
    #[inline(always)]
    pub(super) fn clip_to_tile(
        line: Line,
        tile_bounds: [Point; 2],
        derivatives: [f32; 4],
        intersection_mask: u32,
        canonical_x_dir: bool,
        canonical_y_dir: bool,
    ) -> (Line, bool, bool) {
        const WIDTH: f32 = TILE_SIZE as f32;
        const HEIGHT: f32 = TILE_SIZE as f32;

        let top_left = tile_bounds[0];
        let tile_min_x = tile_bounds[0].x;
        let tile_min_y = tile_bounds[0].y;
        let tile_max_x = tile_bounds[1].x;
        let tile_max_y = tile_bounds[1].y;

        let dx = derivatives[0];
        let dy = derivatives[1];

        // A line's direction dictates which edges it can cross from the outside-in (entry) versus
        // inside-out (exit). For example, if dx > 0 (canonical_x_dir = true), the vector is
        // monotonic in the positive X direction. It is impossible to intersect the right edge as
        // an entry point, or the left edge as an exit point. This reduces the number of edges
        // which must be checked from four to two candidate entry and two candidate exit edges.
        let (mask_v_in, bound_v_in, mask_v_out, bound_v_out) = if canonical_x_dir {
            (Self::L, tile_min_x, Self::R, tile_max_x)
        } else {
            (Self::R, tile_max_x, Self::L, tile_min_x)
        };

        let (mask_h_in, bound_h_in, mask_h_out, bound_h_out) = if canonical_y_dir {
            (Self::T, tile_min_y, Self::B, tile_max_y)
        } else {
            (Self::B, tile_max_y, Self::T, tile_min_y)
        };

        let inv_dx = derivatives[2];
        let inv_dy = derivatives[3];

        // Check the candidate edges against the intersection mask.
        let entry_hits = intersection_mask & (mask_v_in | mask_h_in);
        let exit_hits = intersection_mask & (mask_v_out | mask_h_out);

        let clip_pt = |mut p: Point, hits: u32, mask_h: u32, bound_h: f32, bound_v: f32| {
            let mut is_left = false;
            if hits != 0 {
                let use_h = (intersection_mask & mask_h) != 0;
                let bound = if use_h { bound_h } else { bound_v };
                let start = if use_h { line.p0.y } else { line.p0.x };
                let inv_d = if use_h { inv_dy } else { inv_dx };

                let t = (bound - start) * inv_d;

                p.x = line.p0.x + t * dx;
                p.y = line.p0.y + t * dy;

                if use_h {
                    p.y = bound;
                    p.x -= top_left.x;
                    p.y -= top_left.y;
                    p.x = p.x.clamp(0.0, WIDTH);
                } else {
                    p.x = bound;
                    p.x -= top_left.x;
                    p.y -= top_left.y;
                    p.y = p.y.clamp(0.0, HEIGHT);
                    is_left = bound == tile_min_x;
                }
            } else {
                p.x -= top_left.x;
                p.y -= top_left.y;
                p.x = p.x.clamp(0.0, WIDTH);
                p.y = p.y.clamp(0.0, HEIGHT);
            }
            (p, is_left)
        };

        let (p_entry, entry_left) = clip_pt(line.p0, entry_hits, mask_h_in, bound_h_in, bound_v_in);
        let (p_exit, exit_left) = clip_pt(line.p1, exit_hits, mask_h_out, bound_h_out, bound_v_out);

        // The entry point is the top point iff the line goes down, so the left-edge flags of the
        // entry and exit points map to the top and bottom points accordingly.
        let top_is_on_left_edge = if canonical_y_dir {
            entry_left
        } else {
            exit_left
        };
        let bot_is_on_left_edge = if canonical_y_dir {
            exit_left
        } else {
            entry_left
        };

        // Guarantee predictable winding order for downstream rasterization stages.
        if canonical_y_dir {
            (
                Line::new(p_entry, p_exit),
                top_is_on_left_edge,
                bot_is_on_left_edge,
            )
        } else {
            (
                Line::new(p_exit, p_entry),
                top_is_on_left_edge,
                bot_is_on_left_edge,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(tx: f32, ty: f32) -> [Point; 2] {
        [
            Point::new(tx * 16.0, ty * 16.0),
            Point::new(tx * 16.0 + 16.0, ty * 16.0 + 16.0),
        ]
    }

    fn clip(line: Line, tx: f32, ty: f32, mask: u32) -> (Line, bool, bool) {
        let dx = line.p1.x - line.p0.x;
        let dy = line.p1.y - line.p0.y;
        let inv = |d: f32| if d.abs() <= 1e-5 { 0.0 } else { 1.0 / d };
        Tile::clip_to_tile(
            line,
            bounds(tx, ty),
            [dx, dy, inv(dx), inv(dy)],
            mask,
            line.p1.x >= line.p0.x,
            line.p1.y >= line.p0.y,
        )
    }

    fn line(x0: f32, y0: f32, x1: f32, y1: f32) -> Line {
        Line::new(Point::new(x0, y0), Point::new(x1, y1))
    }

    #[test]
    fn interior_line_is_preserved_and_sorted() {
        let (l, top_left, bot_left) = clip(line(37.25, 30.5, 35.0, 20.0), 2.0, 1.0, 0);
        assert_eq!(l, line(3.0, 4.0, 5.25, 14.5));
        assert!(!top_left && !bot_left);
    }

    #[test]
    fn crossing_line_is_snapped() {
        // Enters through the top, leaves through the right edge, going right and down.
        let (l, top_left, bot_left) =
            clip(line(4.0, -8.0, 36.0, 24.0), 0.0, 0.0, Tile::T | Tile::R);
        assert_eq!(l, line(12.0, 0.0, 16.0, 4.0));
        assert!(!top_left && !bot_left);

        // Going up and left: enters through the bottom and leaves through the left edge.
        let (l, top_left, bot_left) =
            clip(line(20.0, 40.0, -12.0, 8.0), 0.0, 1.0, Tile::B | Tile::L);
        assert_eq!(l, line(0.0, 4.0, 12.0, 16.0));
        assert!(top_left && !bot_left);
    }

    #[test]
    fn single_edge_touch() {
        // Ends exactly on the left edge of the tile, coming from the right: one hit on L.
        let (l, top_left, bot_left) = clip(line(24.0, 4.0, 16.0, 8.0), 1.0, 0.0, Tile::L);
        assert_eq!(l, line(8.0, 4.0, 0.0, 8.0));
        assert!(!top_left && bot_left);
    }
}
