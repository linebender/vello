// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Primitives for creating tiles.

use crate::flatten::Line;
use alloc::vec;
use alloc::vec::Vec;
use fearless_simd::Level;
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

/// The max number of lines per path.
///
/// Trying to render a path with more lines than this may result in visual artifacts.
pub const MAX_LINES_PER_PATH: u32 = 1 << 26;

/// A tile represents an aligned area on the pixmap, used to subdivide the viewport into sub-areas
/// (currently 4x4) and analyze line intersections inside each such area.
///
/// Keep in mind that it is possible to have multiple tiles with the same index,
/// namely if we have multiple lines crossing the same 4x4 area!
///
/// # Note
///
/// This struct is `#[repr(C)]`, but the byte order of its fields is dependent on the endianness of
/// the compilation target.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Tile {
    // The field ordering is important.
    //
    // The given ordering (variant over little and big endian compilation targets), ensures that
    // `Tile::to_bits` doesn't do any actual work, as the ordering of the fields is such that the
    // numeric value of a `Tile` in memory is identical as returned by that method. This allows
    // for, e.g., comparison and sorting.
    #[cfg(target_endian = "big")]
    /// The index of the tile in the y direction.
    pub y: u16,

    #[cfg(target_endian = "big")]
    /// The index of the tile in the x direction.
    pub x: u16,

    /// The index of the line this tile belongs to into the line buffer, intersection data,
    /// and whether the line crosses the top edge of the tile, packed together.
    ///
    /// The layout is:
    /// - **Bits 0-25 (26 bits):** The line index (`line_idx`).
    /// - **Bits 26-30 (5 bits):** Intersection data.
    ///   - Bit 26 (mask `0b00001`): Intersects top edge
    ///   - Bit 27 (mask `0b00010`): Intersects bottom edge
    ///   - Bit 28 (mask `0b00100`): Intersects left edge
    ///   - Bit 29 (mask `0b01000`): Intersects right edge
    ///   - Bit 30 (mask `0b10000`): Does this tile have one unique intersection?
    /// - **Bit 31 (1 bit):** Winding (1 if crosses top edge).
    pub packed_winding_line_idx: u32,

    #[cfg(target_endian = "little")]
    /// The index of the tile in the x direction.
    pub x: u16,

    #[cfg(target_endian = "little")]
    /// The index of the tile in the y direction.
    pub y: u16,
}

impl Tile {
    /// The width of a tile in pixels.
    pub const WIDTH: u16 = 4;

    /// The height of a tile in pixels.
    pub const HEIGHT: u16 = 4;

    /// Create a new tile.
    /// `x` and `y` will be clamped to the largest possible coordinate if they are too large.
    ///
    /// `line_idx` must be smaller than [`MAX_LINES_PER_PATH`].
    #[inline]
    pub fn new_clamped(
        x: u16,
        y: u16,
        line_idx: u32,
        winding: bool,
        intersection_data: u32,
    ) -> Self {
        Self::new(
            // Make sure that x and y stay in range when multiplying
            // with the tile width and height during strips generation.
            x.min(u16::MAX / Self::WIDTH),
            y.min(u16::MAX / Self::HEIGHT),
            line_idx,
            winding,
            intersection_data,
        )
    }

    /// The base tile constructor
    #[inline]
    pub const fn new(x: u16, y: u16, line_idx: u32, winding: bool, intersection_data: u32) -> Self {
        #[cfg(debug_assertions)]
        if line_idx >= MAX_LINES_PER_PATH {
            panic!("Max. number of lines per path exceeded.");
        }
        Self {
            x,
            y,
            packed_winding_line_idx: ((winding as u32) << 31)
                | (intersection_data << 26)
                | line_idx,
        }
    }

    /// Check whether two tiles are at the same location.
    #[inline]
    pub const fn same_loc(&self, other: &Self) -> bool {
        self.same_row(other) && self.x == other.x
    }

    /// Check whether `self` is adjacent to the left of `other`.
    #[inline]
    pub const fn prev_loc(&self, other: &Self) -> bool {
        self.same_row(other) && self.x + 1 == other.x
    }

    /// Check whether two tiles are on the same row.
    #[inline]
    pub const fn same_row(&self, other: &Self) -> bool {
        self.y == other.y
    }

    /// The index of the line this tile belongs to into the line buffer.
    #[inline]
    pub const fn line_idx(&self) -> u32 {
        self.packed_winding_line_idx & (MAX_LINES_PER_PATH - 1)
    }

    /// Whether the line crosses the top edge of the tile.
    ///
    /// Lines making this crossing increment or decrement the coarse tile winding, depending on the
    /// line direction.
    #[inline]
    pub const fn winding(&self) -> bool {
        (self.packed_winding_line_idx & (1 << 31)) != 0
    }

    /// The 5 bits of intersection data.
    ///
    /// - **Bits 0-3 (mask `0b1111`):** Edge intersection mask.
    ///   - Bit 0 (mask `0b00001`): Intersects top edge
    ///   - Bit 1 (mask `0b00010`): Intersects bottom edge
    ///   - Bit 2 (mask `0b00100`): Intersects left edge
    ///   - Bit 3 (mask `0b01000`): Intersects right edge
    /// - **Bit 4 (mask `0b10000`):** Does this tile have one unique intersection?
    #[inline]
    pub const fn intersection_data(&self) -> u32 {
        (self.packed_winding_line_idx >> 26) & 0b11111
    }

    /// Whether the line intersects the top edge of the tile.
    #[inline]
    pub const fn intersects_top(&self) -> bool {
        (self.intersection_data() & 0b0001) != 0
    }

    /// Whether the line intersects the bottom edge of the tile.
    #[inline]
    pub const fn intersects_bottom(&self) -> bool {
        (self.intersection_data() & 0b0010) != 0
    }

    /// Whether the line intersects the left edge of the tile.
    #[inline]
    pub const fn intersects_left(&self) -> bool {
        (self.intersection_data() & 0b0100) != 0
    }

    /// Whether the line intersects the right edge of the tile.
    #[inline]
    pub const fn intersects_right(&self) -> bool {
        (self.intersection_data() & 0b1000) != 0
    }

    /// Whether the line's start point (p0) is inside the tile.
    #[inline]
    pub const fn perfect_intersection(&self) -> bool {
        (self.intersection_data() & 0b10000) != 0
    }

    /// Return the `u64` representation of this tile.
    ///
    /// This is the u64 interpretation of `(y, x, packed_winding_line_idx)` where `y` is the
    /// most-significant part of the number and `packed_winding_line_idx` the least significant.
    #[inline(always)]
    const fn to_bits(self) -> u64 {
        ((self.y as u64) << 48) | ((self.x as u64) << 32) | self.packed_winding_line_idx as u64
    }
}

impl PartialEq for Tile {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl Ord for Tile {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.to_bits().cmp(&other.to_bits())
    }
}

impl PartialOrd for Tile {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Tile {}

/// Handles the tiling of paths.
#[derive(Clone, Debug)]
pub struct Tiles {
    tile_buf: Vec<Tile>,
    level: Level,
    sorted: bool,
}

impl Tiles {
    /// Create a new tiles container.
    pub fn new(level: Level) -> Self {
        Self {
            tile_buf: vec![],
            level,
            sorted: false,
        }
    }

    /// Get the number of tiles in the container.
    pub fn len(&self) -> u32 {
        self.tile_buf.len() as u32
    }

    /// Returns `true` if the container has no tiles.
    pub fn is_empty(&self) -> bool {
        self.tile_buf.is_empty()
    }

    /// Reset the tiles' container.
    pub fn reset(&mut self) {
        self.tile_buf.clear();
        self.sorted = false;
    }

    /// Sort the tiles in the container.
    pub fn sort_tiles(&mut self) {
        self.sorted = true;
        // To enable auto-vectorization.
        self.level.dispatch(|_| self.tile_buf.sort_unstable());
    }

    /// Get the tile at a certain index.
    ///
    /// Panics if the container hasn't been sorted before.
    #[inline]
    pub fn get(&self, index: u32) -> &Tile {
        assert!(
            self.sorted,
            "attempted to call `get` before sorting the tile container."
        );

        &self.tile_buf[index as usize]
    }

    /// Iterate over the tiles in sorted order.
    ///
    /// Panics if the container hasn't been sorted before.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Tile> {
        assert!(
            self.sorted,
            "attempted to call `iter` before sorting the tile container."
        );

        self.tile_buf.iter()
    }

    /// Tiles exceeding the top, right or bottom of the viewport (given by `width` and `height` in
    /// pixels) are culled.
    //
    // TODO: Tiles are clamped to the left edge of the viewport, but lines fully to the left of the
    // viewport are not culled yet. These lines impact winding, and would need forwarding of
    // winding to the strip generation stage.
    pub fn make_tiles(&mut self, lines: &[Line], width: u16, height: u16) {
        self.reset();

        if width == 0 || height == 0 {
            return;
        }

        debug_assert!(
            lines.len() <= MAX_LINES_PER_PATH as usize,
            "Max. number of lines per path exceeded. Max is {}, got {}.",
            MAX_LINES_PER_PATH,
            lines.len()
        );

        let tile_columns = width.div_ceil(Tile::WIDTH);
        let tile_rows = height.div_ceil(Tile::HEIGHT);

        for (line_idx, line) in lines.iter().take(MAX_LINES_PER_PATH as usize).enumerate() {
            let line_idx = line_idx as u32;

            let p0_x = line.p0.x / f32::from(Tile::WIDTH);
            let p0_y = line.p0.y / f32::from(Tile::HEIGHT);
            let p1_x = line.p1.x / f32::from(Tile::WIDTH);
            let p1_y = line.p1.y / f32::from(Tile::HEIGHT);

            let (line_left_x, line_right_x) = if p0_x < p1_x {
                (p0_x, p1_x)
            } else {
                (p1_x, p0_x)
            };

            // Lines whose left-most endpoint exceed the right edge of the viewport are culled
            if line_left_x > tile_columns as f32 {
                continue;
            }

            let (line_top_y, line_top_x, line_bottom_y, line_bottom_x) = if p0_y < p1_y {
                (p0_y, p0_x, p1_y, p1_x)
            } else {
                (p1_y, p1_x, p0_y, p0_x)
            };

            // The `as u16` casts here intentionally clamp negative coordinates to 0.
            let y_top_tiles = (line_top_y as u16).min(tile_rows);
            let line_bottom_y_ceil = line_bottom_y.ceil();
            let y_bottom_tiles = (line_bottom_y_ceil as u16).min(tile_rows);

            // Get tile coordinates for start/end points, use i32 to preserve negative coordinates
            let p0_tile_x = line_top_x.floor() as i32;
            let p0_tile_y = line_top_y.floor() as i32;
            let p1_tile_x = line_bottom_x.floor() as i32;

            // Because our vertical loop is exclusive, tiles which result from an endpoint touching
            // the exact bottom edge of a tile do not get produced. This exclusive behavior is
            // correct, the line is not actually crossing, and producing an empty tile breaks the
            // rendering. However, a side effect of this is that the bottommost tile produced by
            // this case will not count itself as end_tile. This in turn results in recording a
            // bottom edge crossing. In isolation, this is fine, but another linesegment may match
            // that endpoint. If that is the case, if that linesegment traverses downwards, then,
            // its top edge starting point will not be an intersection. Because it is not an
            // intersection, the rasterizing algorithm will use the raw point, which can potentially
            // lead to a watertightness difference.
            //
            // The solution, here and in the general case, is to check whether the bottommost point
            // in the line is perfectly axis-aligned, and if that is true, to adjust the location
            // of the bottom tile, such that the bottom point is not considered an intersection.
            let p1_tile_y = if line_bottom_y == line_bottom_y_ceil {
                line_bottom_y as i32 - 1
            } else {
                line_bottom_y.floor() as i32
            };

            // special-case out lines which are fully contained within a tile.
            let not_same_tile = p0_tile_y != p1_tile_y || p0_tile_x != p1_tile_x;
            if not_same_tile {
                // For ease of logic, special-case purely vertical tiles.
                if line_left_x == line_right_x {
                    // If we're here (not_same), and we have the same top and bottom tile, they must
                    // have been culled, so exit.
                    if y_top_tiles < y_bottom_tiles {
                        let x = (line_left_x as u16).min(tile_columns.saturating_sub(1));

                        // Row Start, not culled.
                        let is_start_culled = line_top_y < 0.0;
                        if !is_start_culled {
                            let y = f32::from(y_top_tiles);
                            // A vertical line is never considered as intersecting horizontal edges,
                            // so line start/end is always a single unique intersection.
                            let is_start_or_end = (y_top_tiles as i32) == p0_tile_y
                                || (y_top_tiles as i32) == p1_tile_y;
                            let intersection_data = 2 | (is_start_or_end as u32) << 4;

                            let tile = Tile::new_clamped(
                                x,
                                y_top_tiles,
                                line_idx,
                                y >= line_top_y,
                                intersection_data,
                            );
                            self.tile_buf.push(tile);
                        }

                        // Middle
                        // If the start was culled, the first tile inside the viewport is a middle
                        let y_start = if is_start_culled {
                            y_top_tiles
                        } else {
                            y_top_tiles + 1
                        };
                        let line_bottom_floor = line_bottom_y.floor();
                        let y_end_idx = (line_bottom_floor as u16).min(tile_rows);

                        // Perfect touching B case
                        if y_start < y_end_idx {
                            let y_last = y_end_idx - 1;
                            for y_idx in y_start..y_last {
                                let tile = Tile::new_clamped(
                                    x,
                                    y_idx,
                                    line_idx,
                                    f32::from(y_idx) >= line_top_y,
                                    0b11,
                                );
                                self.tile_buf.push(tile);
                            }

                            let is_end_tile = ((y_last as i32) == p1_tile_y) as u32;
                            let intersection_data =
                                0b1 | ((1 ^ is_end_tile) << 1) | (is_end_tile << 4);
                            let tile = Tile::new_clamped(
                                x,
                                y_last,
                                line_idx,
                                f32::from(y_last) >= line_top_y,
                                intersection_data,
                            );
                            self.tile_buf.push(tile);
                        }

                        // Row End, handle the final tile (y_end_idx), but *only* if the line does
                        // not perfectly end on the top edge of the tile. In the case that it does,
                        // it gets handled by the middle logic above.
                        if line_bottom_y != line_bottom_floor && y_end_idx < tile_rows {
                            let y = f32::from(y_end_idx);
                            let is_start_or_end =
                                (y_end_idx as i32) == p0_tile_y || (y_end_idx as i32) == p1_tile_y;
                            let intersection_data = 1 | (is_start_or_end as u32) << 4;
                            let tile = Tile::new_clamped(
                                x,
                                y_end_idx,
                                line_idx,
                                y >= line_top_y,
                                intersection_data,
                            );
                            self.tile_buf.push(tile);
                        }
                    }
                } else {
                    let dx = p1_x - p0_x;
                    let dy = p1_y - p0_y;
                    let x_slope = dx / dy;
                    let dx_dir = (line_bottom_x >= line_top_x) as u32;
                    let not_dx_dir = dx_dir ^ 1;
                    let is_start_tile = |x_idx: u16, y_idx: u16| -> bool {
                        (x_idx as i32 == p0_tile_x) && (y_idx as i32 == p0_tile_y)
                    };
                    let is_end_tile = |x_idx: u16, y_idx: u16| -> bool {
                        (x_idx as i32 == p1_tile_x) && (y_idx as i32 == p1_tile_y)
                    };

                    // Line walks rows top to bottom, left to right, y-exclusive, x-inclusive
                    for y_idx in y_top_tiles..y_bottom_tiles {
                        let y = f32::from(y_idx);

                        let line_row_top_y = line_top_y.max(y).min(y + 1.);
                        let line_row_bottom_y = line_bottom_y.max(y).min(y + 1.);

                        let line_row_top_x = p0_x + (line_row_top_y - p0_y) * x_slope;
                        let line_row_bottom_x = p0_x + (line_row_bottom_y - p0_y) * x_slope;

                        let line_row_left_x =
                            f32::min(line_row_top_x, line_row_bottom_x).max(line_left_x);
                        let line_row_right_x =
                            f32::max(line_row_top_x, line_row_bottom_x).min(line_right_x);

                        // Floor so we don't truncate towards zero
                        let cannonical_x_start = line_row_left_x.floor() as i32;
                        let x_start = line_row_left_x as u16;
                        let cannonical_x_end = line_row_right_x as u16;
                        let x_end = cannonical_x_end.min(tile_columns.saturating_sub(1));

                        // Row start, but not necessarily the cannonical start of a row.
                        if x_start <= x_end {
                            // Check if we are the row start/end unculled.
                            let unc_row_start = (x_start as i32 == cannonical_x_start) as u32;
                            let unc_row_end = (x_start == cannonical_x_end) as u32;
                            let cannonical_row_start =
                                (dx_dir & unc_row_start) | (not_dx_dir & unc_row_end);
                            let cannonical_row_end =
                                (not_dx_dir & unc_row_start) | (dx_dir & unc_row_end);
                            let start_tile = is_start_tile(x_start, y_idx) as u32;
                            let end_tile = is_end_tile(x_start, y_idx) as u32;

                            // Entrant
                            let vert_entrant = cannonical_row_start & (1 ^ start_tile);
                            let hor_entrant = 1 ^ cannonical_row_start;

                            let mut intersection_data = vert_entrant;
                            intersection_data |= hor_entrant << not_dx_dir << 2;

                            // Exit
                            let vert_exit = cannonical_row_end & (1 ^ end_tile);
                            let hor_exit = 1 ^ cannonical_row_end;
                            intersection_data |= vert_exit << 1;
                            intersection_data |= hor_exit << dx_dir << 2;

                            // Check if the line passes through any of the four corners of this tile.
                            // It passes through a corner if the x-intersection with the top or bottom
                            // edge equals either the left (x_start) or right (x_start + 1) tile boundary.
                            let x_start_f = x_start as f32;
                            let x_right_f = x_start_f + 1.0;

                            let top_corner =
                                (line_row_top_x == x_start_f || line_row_top_x == x_right_f) as u32;
                            let bottom_corner = (line_row_bottom_x == x_start_f
                                || line_row_bottom_x == x_right_f)
                                as u32;

                            // Perfect bit is set if we hit exactly one corner,
                            // or if it's a start/end tile.
                            let perfect_bit = (top_corner ^ bottom_corner) | start_tile | end_tile;

                            intersection_data |= perfect_bit << 4;

                            let tile = Tile::new_clamped(
                                x_start,
                                y_idx,
                                line_idx,
                                y >= line_top_y && (dx_dir != 0 || x_start == x_end),
                                intersection_data,
                            );
                            self.tile_buf.push(tile);
                        }

                        // Middle
                        for x_idx in x_start + 1..x_end {
                            let intersection_data = 0b1100; // RL
                            let tile =
                                Tile::new_clamped(x_idx, y_idx, line_idx, false, intersection_data);
                            self.tile_buf.push(tile);
                        }

                        // Row End
                        // A single tile row would have been handled in the row start clause,
                        // so there is no ambiguity as to whether this is the start or end of a row
                        // except from culling.
                        if x_start < x_end {
                            // Note: must be lt for clipping instead of neq.
                            let unc_row_end = (x_end == cannonical_x_end) as u32;
                            let cannonical_row_start = not_dx_dir & unc_row_end;
                            let cannonical_row_end = dx_dir & unc_row_end;
                            let start_tile = is_start_tile(x_end, y_idx) as u32;
                            let end_tile = is_end_tile(x_end, y_idx) as u32;

                            // Entrant
                            let vert_entrant = cannonical_row_start & (1 ^ start_tile);
                            let hor_entrant = 1 ^ cannonical_row_start;
                            let mut intersection_data = vert_entrant;
                            intersection_data |= hor_entrant << not_dx_dir << 2;

                            // Exit
                            let vert_exit = cannonical_row_end & (1 ^ end_tile);
                            let hor_exit = 1 ^ cannonical_row_end;
                            intersection_data |= vert_exit << 1;
                            intersection_data |= hor_exit << dx_dir << 2;

                            // Perfect_bit
                            let x_end_f = x_end as f32;
                            let x_right_f = x_end_f + 1.0;
                            let top_corner =
                                (line_row_top_x == x_end_f || line_row_top_x == x_right_f) as u32;
                            let bottom_corner = (line_row_bottom_x == x_end_f
                                || line_row_bottom_x == x_right_f)
                                as u32;
                            let perfect_bit = (top_corner ^ bottom_corner) | start_tile | end_tile;
                            intersection_data |= perfect_bit << 4;

                            let tile = Tile::new_clamped(
                                x_end,
                                y_idx,
                                line_idx,
                                y >= line_top_y && not_dx_dir != 0,
                                intersection_data,
                            );
                            self.tile_buf.push(tile);
                        }
                    }
                }
            } else {
                // Case: Line is fully contained within a single tile.
                // Must exactly match the general case
                if y_top_tiles < y_bottom_tiles {
                    let y = f32::from(y_top_tiles);
                    let line_row_top_y = line_top_y.max(y).min(y + 1.);
                    let line_row_bottom_y = line_bottom_y.max(y).min(y + 1.);
                    let x_slope = (p1_x - p0_x) / (p1_y - p0_y);
                    let line_row_top_x = p0_x + ((line_row_top_y) - p0_y) * x_slope;
                    let line_row_bottom_x = p0_x + (line_row_bottom_y - p0_y) * x_slope;
                    let x_idx = (f32::min(line_row_top_x, line_row_bottom_x).max(line_left_x)
                        as u16)
                        .min(tile_columns + 1);
                    let tile = Tile::new_clamped(x_idx, y_top_tiles, line_idx, y >= line_top_y, 0);
                    self.tile_buf.push(tile);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::flatten::{FlattenCtx, Line, Point, fill};
    use crate::kurbo::{Affine, BezPath};
    use crate::tile::{Tile, Tiles};
    use fearless_simd::Level;
    use std::vec;

    const P: u32 = 0b10000;
    const R: u32 = 0b01000;
    const L: u32 = 0b00100;
    const B: u32 = 0b00010;
    const T: u32 = 0b00001;

    const VIEW_HEIGHT: u16 = 100;
    const F_V_HEIGHT: f32 = VIEW_HEIGHT as f32;

    //==============================================================================================
    // Culled Lines
    //==============================================================================================
    #[test]
    fn cull_sloped_outside_lines() {
        let lines = [
            Line {
                p0: Point { x: 1.0, y: -7.0 },
                p1: Point { x: 3.0, y: -1.0 },
            },
            Line {
                p0: Point { x: 1.0, y: -11.0 },
                p1: Point { x: 3.0, y: -1.0 },
            },
            Line {
                p0: Point { x: 101.0, y: 50.0 },
                p1: Point { x: 103.0, y: 70.0 },
            },
            Line {
                p0: Point { x: 1.0, y: 101.0 },
                p1: Point { x: 3.0, y: 107.0 },
            },
            Line {
                p0: Point { x: 1.0, y: 101.0 },
                p1: Point { x: 3.0, y: 113.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);

        assert!(tiles.is_empty());
    }

    #[test]
    fn sloped_line_crossing_top() {
        let lines = [
            Line {
                p0: Point { x: -2.0, y: -3.0 },
                p1: Point { x: 2.0, y: 1.0 },
            },
            Line {
                p0: Point { x: 6.0, y: -1.0 },
                p1: Point { x: 5.0, y: 2.0 },
            },
            Line {
                p0: Point { x: 9.0, y: -10.0 },
                p1: Point { x: 10.0, y: 3.0 },
            },
            Line {
                p0: Point { x: 2.0, y: 1.0 },
                p1: Point { x: -2.0, y: -3.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, true, P | T),
                Tile::new(0, 0, 3, true, P | T),
                Tile::new(1, 0, 1, true, P | T),
                Tile::new(2, 0, 2, true, P | T),
            ]
        );
    }

    #[test]
    fn sloped_line_crossing_bot() {
        let lines = [
            Line {
                p0: Point { x: 5.0, y: 103.0 },
                p1: Point { x: 6.0, y: 98.0 },
            },
            Line {
                p0: Point { x: 10.0, y: 101.0 },
                p1: Point { x: 9.0, y: 99.0 },
            },
            Line {
                p0: Point { x: 2.0, y: 98.0 },
                p1: Point { x: 3.0, y: 103.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 24, 2, false, P | B),
                Tile::new(1, 24, 0, false, P | B),
                Tile::new(2, 24, 1, false, P | B),
            ]
        );
    }

    #[test]
    fn sloped_line_crossing_top_multi_tile() {
        let lines = [
            Line {
                p0: Point { x: 1.0, y: -5.0 },
                p1: Point { x: 6.0, y: 7.0 },
            },
            Line {
                p0: Point { x: 2.5, y: -10.0 },
                p1: Point { x: 3.5, y: 6.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 1, true, T | B),
                Tile::new(0, 0, 0, true, T | R),
                Tile::new(1, 0, 0, false, L | B),
                Tile::new(0, 1, 1, true, T | P),
                Tile::new(1, 1, 0, true, T | P),
            ]
        );
    }

    #[test]
    fn sloped_line_crossing_bot_multi_tile() {
        let lines = [
            Line {
                p0: Point { x: 12.0, y: 110.0 },
                p1: Point { x: 2.0, y: 94.0 },
            },
            Line {
                p0: Point { x: 1.5, y: 105.0 },
                p1: Point { x: 3.5, y: 94.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 23, 0, false, B | P),
                Tile::new(0, 23, 1, false, B | P),
                Tile::new(0, 24, 1, true, T | B),
                Tile::new(0, 24, 0, true, T | R),
                Tile::new(1, 24, 0, false, B | L),
            ]
        );
    }

    #[test]
    fn sloped_line_crossing_right() {
        let lines = [
            Line {
                p0: Point { x: 97.0, y: 1.0 },
                p1: Point { x: 101.0, y: 2.0 },
            },
            Line {
                p0: Point { x: 93.0, y: 1.0 },
                p1: Point { x: 105.0, y: 2.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(23, 0, 1, false, P | R),
                Tile::new(24, 0, 1, false, R | L),
                Tile::new(24, 0, 0, false, P | R),
            ]
        );
    }

    #[test]
    fn sloped_line_crossing_left() {
        let lines = [
            Line {
                p0: Point { x: -5.0, y: 1.0 },
                p1: Point { x: 1.0, y: 2.0 },
            },
            Line {
                p0: Point { x: -5.0, y: 1.0 },
                p1: Point { x: 5.0, y: 2.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 1, false, L | R),
                Tile::new(0, 0, 0, false, P | L),
                Tile::new(1, 0, 1, false, P | L),
            ]
        );
    }

    #[test]
    fn horizontal_line_above_viewport() {
        let lines = [Line {
            p0: Point { x: 10.0, y: -5.0 },
            p1: Point { x: 90.0, y: -5.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);

        assert!(tiles.is_empty());
    }

    #[test]
    fn horizontal_line_below_viewport() {
        let lines = [Line {
            p0: Point { x: 10.0, y: 105.0 },
            p1: Point { x: 90.0, y: 105.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);

        assert!(tiles.is_empty());
    }

    #[test]
    fn horizontal_line_crossing_left_viewport() {
        let lines = [Line {
            p0: Point { x: -10.0, y: 10.0 },
            p1: Point { x: 10.0, y: 10.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        let expected = [
            Tile::new(0, 2, 0, false, L | R),
            Tile::new(1, 2, 0, false, L | R),
            Tile::new(2, 2, 0, false, P | L),
        ];

        assert_eq!(tiles.tile_buf, expected);
    }

    #[test]
    fn horizontal_line_crossing_right_viewport() {
        let lines = [Line {
            p0: Point { x: 15.0, y: 10.0 },
            p1: Point { x: 25.0, y: 10.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 20, 100);
        tiles.sort_tiles();

        let expected = [
            Tile::new(3, 2, 0, false, P | R),
            Tile::new(4, 2, 0, false, L | R),
        ];

        assert_eq!(tiles.tile_buf, expected);
    }

    #[test]
    fn vertical_lines_outside_viewport() {
        let lines = [
            Line {
                p0: Point { x: 1.0, y: -5.0 },
                p1: Point { x: 1.0, y: -1.0 },
            },
            Line {
                p0: Point {
                    x: 1.0,
                    y: F_V_HEIGHT + 1.0,
                },
                p1: Point {
                    x: 1.0,
                    y: F_V_HEIGHT + 5.0,
                },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, VIEW_HEIGHT);
        tiles.sort_tiles();
        assert_eq!(tiles.tile_buf, []);
    }

    #[test]
    fn vertical_path_on_the_right_of_viewport() {
        let path = BezPath::from_svg("M261,0 L78848,0 L78848,4 L261,4 Z").unwrap();
        let mut line_buf = vec![];
        fill(
            Level::try_detect().unwrap_or(Level::fallback()),
            &path,
            Affine::IDENTITY,
            &mut line_buf,
            &mut FlattenCtx::default(),
        );

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&line_buf, 10, 10);
        assert!(tiles.is_empty());
    }

    #[test]
    fn vertical_line_crossing_top_viewport() {
        let lines = [
            Line {
                p0: Point { x: 1.0, y: -7.0 },
                p1: Point { x: 1.0, y: 3.0 },
            },
            Line {
                p0: Point { x: 1.0, y: -7.0 },
                p1: Point { x: 1.0, y: 7.0 },
            },
            Line {
                p0: Point { x: 1.0, y: -7.0 },
                p1: Point { x: 1.0, y: 8.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 1, true, B | T),
                Tile::new(0, 0, 2, true, B | T),
                Tile::new(0, 0, 0, true, P | T),
                Tile::new(0, 1, 1, true, P | T),
                Tile::new(0, 1, 2, true, P | T),
            ]
        );
    }

    #[test]
    fn vertical_line_crossing_bot_viewport() {
        let lines = [
            Line {
                p0: Point {
                    x: 1.0,
                    y: F_V_HEIGHT - 1.0,
                },
                p1: Point {
                    x: 1.0,
                    y: F_V_HEIGHT + 5.0,
                },
            },
            Line {
                p0: Point {
                    x: 1.0,
                    y: F_V_HEIGHT - 5.0,
                },
                p1: Point {
                    x: 1.0,
                    y: F_V_HEIGHT + 5.0,
                },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, VIEW_HEIGHT);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 23, 1, false, P | B),
                Tile::new(0, 24, 0, false, P | B),
                Tile::new(0, 24, 1, true, T | B),
            ]
        );
    }

    #[test]
    fn clip_top_left_corner() {
        let lines = [Line {
            p0: Point { x: -1.0, y: 2.0 },
            p1: Point { x: 2.0, y: -1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();
        assert_eq!(tiles.tile_buf, [Tile::new(0, 0, 0, true, L | T)]);
    }

    #[test]
    fn clip_bottom_right_corner() {
        let lines = [Line {
            p0: Point {
                x: F_V_HEIGHT + 1.0,
                y: F_V_HEIGHT - 2.0,
            },
            p1: Point {
                x: F_V_HEIGHT - 2.0,
                y: F_V_HEIGHT + 1.0,
            },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();
        assert_eq!(tiles.tile_buf, [Tile::new(24, 24, 0, false, R | B)]);
    }

    //==============================================================================================
    // Axis-aligned lines
    //==============================================================================================
    #[test]
    fn horizontal_line_left_to_right_three_tile() {
        let lines = [Line {
            p0: Point { x: 1.5, y: 1.0 },
            p1: Point { x: 8.5, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false, P | R),
                Tile::new(1, 0, 0, false, R | L),
                Tile::new(2, 0, 0, false, P | L),
            ]
        );
    }

    #[test]
    fn horizontal_line_right_to_left_three_tile() {
        let lines = [Line {
            p0: Point { x: 8.5, y: 1.0 },
            p1: Point { x: 1.5, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false, P | R),
                Tile::new(1, 0, 0, false, R | L),
                Tile::new(2, 0, 0, false, P | L),
            ]
        );
    }

    #[test]
    fn horizontal_line_multi_tile() {
        let lines = [Line {
            p0: Point { x: 1.5, y: 1.0 },
            p1: Point { x: 12.5, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false, P | R),
                Tile::new(1, 0, 0, false, R | L),
                Tile::new(2, 0, 0, false, R | L),
                Tile::new(3, 0, 0, false, P | L),
            ]
        );
    }

    #[test]
    fn vertical_line_down_three_tile() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.5 },
            p1: Point { x: 1.0, y: 8.5 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false, P | B),
                Tile::new(0, 1, 0, true, T | B),
                Tile::new(0, 2, 0, true, P | T),
            ]
        );
    }

    #[test]
    fn vertical_line_down_multi_tile() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.0 },
            p1: Point { x: 1.0, y: 13.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false, P | B),
                Tile::new(0, 1, 0, true, T | B),
                Tile::new(0, 2, 0, true, T | B),
                Tile::new(0, 3, 0, true, P | T),
            ]
        );
    }

    #[test]
    fn vertical_line_up_three_tile() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 13.0 },
            p1: Point { x: 1.0, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false, P | B),
                Tile::new(0, 1, 0, true, T | B),
                Tile::new(0, 2, 0, true, T | B),
                Tile::new(0, 3, 0, true, P | T),
            ]
        );
    }

    #[test]
    fn vertical_line_up_multi_tile() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 8.5 },
            p1: Point { x: 1.0, y: 1.5 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false, P | B),
                Tile::new(0, 1, 0, true, T | B),
                Tile::new(0, 2, 0, true, P | T),
            ]
        );
    }

    // Exclusive to the bottom edge, no P required.
    #[test]
    fn vertical_line_touching_bot() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.0 },
            p1: Point { x: 1.0, y: 8.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false, P | B),
                Tile::new(0, 1, 0, true, P | T),
            ]
        );
    }

    #[test]
    fn vertical_line_touching_top() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 0.0 },
            p1: Point { x: 1.0, y: 7.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, true, P | B),
                Tile::new(0, 1, 0, true, P | T),
            ]
        );
    }

    //==============================================================================================
    // Sloped Lines
    //==============================================================================================
    #[test]
    fn top_left_to_bottom_right() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.0 },
            p1: Point { x: 11.0, y: 9.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false, P | R),
                Tile::new(1, 0, 0, false, L | B),
                Tile::new(1, 1, 0, true, R | T),
                Tile::new(2, 1, 0, false, L | B),
                Tile::new(2, 2, 0, true, P | T),
            ]
        );
    }

    #[test]
    fn bottom_right_to_top_left() {
        let lines = [Line {
            p0: Point { x: 11.0, y: 9.0 },
            p1: Point { x: 1.0, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false, P | R),
                Tile::new(1, 0, 0, false, L | B),
                Tile::new(1, 1, 0, true, R | T),
                Tile::new(2, 1, 0, false, L | B),
                Tile::new(2, 2, 0, true, P | T),
            ]
        );
    }

    #[test]
    fn bottom_left_to_top_right() {
        let lines = [Line {
            p0: Point { x: 2.0, y: 11.0 },
            p1: Point { x: 14.0, y: 6.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(2, 1, 0, false, R | B),
                Tile::new(3, 1, 0, false, P | L),
                Tile::new(0, 2, 0, false, P | R),
                Tile::new(1, 2, 0, false, R | L),
                Tile::new(2, 2, 0, true, L | T),
            ]
        );
    }

    #[test]
    fn top_right_to_bottom_left() {
        let lines = [Line {
            p0: Point { x: 14.0, y: 6.0 },
            p1: Point { x: 2.0, y: 11.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        // geometrically identical to above
        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(2, 1, 0, false, R | B),
                Tile::new(3, 1, 0, false, P | L),
                Tile::new(0, 2, 0, false, P | R),
                Tile::new(1, 2, 0, false, R | L),
                Tile::new(2, 2, 0, true, L | T),
            ]
        );
    }

    #[test]
    fn two_lines_in_single_tile() {
        let lines = [
            Line {
                p0: Point { x: 1.0, y: 3.0 },
                p1: Point { x: 3.0, y: 3.0 },
            },
            Line {
                p0: Point { x: 3.0, y: 3.0 },
                p1: Point { x: 0.0, y: 1.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);

        // Both lines are entirely within tile (0,0).
        assert_eq!(
            tiles.tile_buf,
            [Tile::new(0, 0, 0, false, 0), Tile::new(0, 0, 1, false, 0)]
        );
    }

    #[test]
    fn intersection_data_diagonal_cross_corner() {
        let lines = [Line {
            p0: Point { x: 3.0, y: 5.0 },
            p1: Point { x: 5.0, y: 3.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(1, 0, 0, false, P | B),
                Tile::new(0, 1, 0, false, P | R),
                Tile::new(1, 1, 0, true, P | L | T),
            ]
        );
    }

    #[test]
    fn diagonal_down_slope_tiles() {
        let lines = [Line {
            p0: Point { x: 5.0, y: 5.0 },
            p1: Point { x: 9.0, y: 9.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        let expected = [
            Tile::new(1, 1, 0, false, P | R),
            Tile::new(2, 1, 0, false, P | L | B),
            Tile::new(2, 2, 0, true, P | T),
        ];

        assert_eq!(tiles.tile_buf, expected);
    }

    #[test]
    fn diagonal_up_slope_tiles() {
        let lines = [Line {
            p0: Point { x: 5.0, y: 9.0 },
            p1: Point { x: 9.0, y: 5.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        let expected = [
            Tile::new(1, 1, 0, false, R | B),
            Tile::new(2, 1, 0, false, P | L),
            Tile::new(1, 2, 0, true, P | T),
        ];

        assert_eq!(tiles.tile_buf, expected);
    }

    #[test]
    fn diagonal_down_one_tile() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 0.0 },
            p1: Point { x: 4.0, y: 4.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        let expected = [
            Tile::new(0, 0, 0, true, P | R),
            Tile::new(1, 0, 0, false, P | L),
        ];

        assert_eq!(tiles.tile_buf, expected);
    }

    #[test]
    fn diagonal_up_one_tile() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 4.0 },
            p1: Point { x: 4.0, y: 0.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        let expected = [
            Tile::new(0, 0, 0, false, P | R),
            Tile::new(1, 0, 0, true, P | L),
        ];

        assert_eq!(tiles.tile_buf, expected);
    }

    #[test]
    fn diagonal_down_two_tile() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 0.0 },
            p1: Point { x: 8.0, y: 8.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        let expected = [
            Tile::new(0, 0, 0, true, P | R),
            Tile::new(1, 0, 0, false, P | L | B),
            Tile::new(1, 1, 0, true, R | T),
            Tile::new(2, 1, 0, false, P | L),
        ];

        assert_eq!(tiles.tile_buf, expected);
    }

    #[test]
    fn diagonal_up_two_tile() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 8.0 },
            p1: Point { x: 8.0, y: 0.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        let expected = [
            Tile::new(1, 0, 0, false, R | B),
            Tile::new(2, 0, 0, true, P | L),
            Tile::new(0, 1, 0, false, P | R),
            Tile::new(1, 1, 0, true, P | L | T),
        ];

        assert_eq!(tiles.tile_buf, expected);
    }

    #[test]
    fn sloped_ending_right() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.0 },
            p1: Point { x: 8.0, y: 2.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);
        tiles.sort_tiles();

        let expected = [
            Tile::new(0, 0, 0, false, P | R),
            Tile::new(1, 0, 0, false, L | R),
            Tile::new(2, 0, 0, false, P | L),
        ];

        assert_eq!(tiles.tile_buf, expected);
    }

    //==============================================================================================
    // Same Tile Cases
    //==============================================================================================
    #[test]
    fn same_tile() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.0 },
            p1: Point { x: 3.0, y: 3.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);

        assert_eq!(tiles.tile_buf, [Tile::new(0, 0, 0, false, 0)]);
    }

    #[test]
    fn same_tile_left() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 1.0 },
            p1: Point { x: 3.0, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);

        assert_eq!(tiles.tile_buf, [Tile::new(0, 0, 0, false, 0)]);
    }

    #[test]
    fn same_tile_top() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 0.0 },
            p1: Point { x: 1.0, y: 3.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);

        assert_eq!(tiles.tile_buf, [Tile::new(0, 0, 0, true, 0)]);
    }

    #[test]
    fn same_tile_right() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.0 },
            p1: Point { x: 4.0, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);

        let expected = [
            Tile::new(0, 0, 0, false, P | R),
            Tile::new(1, 0, 0, false, P | L),
        ];

        assert_eq!(tiles.tile_buf, expected);
    }

    #[test]
    fn same_tile_bottom() {
        let lines = [
            Line {
                p0: Point { x: 1.0, y: 1.0 },
                p1: Point { x: 1.0, y: 4.0 },
            },
            Line {
                p0: Point { x: 1.0, y: 1.0 },
                p1: Point { x: 2.0, y: 4.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);

        let expected = [Tile::new(0, 0, 0, false, 0), Tile::new(0, 0, 1, false, 0)];

        assert_eq!(tiles.tile_buf, expected);
    }

    #[test]
    fn same_tile_top_left() {
        let lines = [
            Line {
                p0: Point { x: 0.0, y: 1.0 },
                p1: Point { x: 1.0, y: 0.0 },
            },
            Line {
                p0: Point { x: 0.0, y: 0.0001 },
                p1: Point { x: 0.0001, y: 0.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&lines, 100, 100);

        let expected = [Tile::new(0, 0, 0, true, 0), Tile::new(0, 0, 1, true, 0)];

        assert_eq!(tiles.tile_buf, expected);
    }

    //==============================================================================================
    // Special Special Cases
    //==============================================================================================
    #[test]
    // See https://github.com/LaurenzV/cpu-sparse-experiments/issues/46.
    fn infinite_loop() {
        let line = Line {
            p0: Point { x: 22.0, y: 552.0 },
            p1: Point { x: 224.0, y: 388.0 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line], 600, 600);
    }
}
