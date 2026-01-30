// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Primitives for creating tiles.

use crate::flatten::Line;
use alloc::vec;
use alloc::vec::Vec;
use fearless_simd::Level;
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

/// T-op bit
const T: u32 = 0b00001;
/// B-ottom bit
const B: u32 = 0b00010;
/// L-eft bit
const L: u32 = 0b00100;
/// R-ight bit
const R: u32 = 0b01000;
/// W-inding bit
const W: u32 = 0b10000;

/// Shift amount corresponding to the bottom bit.
const BOT_SHIFT: u32 = B.trailing_zeros();
/// Shift amount corresponding to the left bit.
const LEFT_SHIFT: u32 = L.trailing_zeros();
/// Shift amount corresponding to the right bit.
const RIGHT_SHIFT: u32 = R.trailing_zeros();
/// Shift amount corresponding to the winding bit.
const WINDING_SHIFT: u32 = W.trailing_zeros();

/// Mask for all intersection and winding bits (Bits 0-4).
const INTERSECTION_MASK: u32 = W | R | L | B | T;
/// Shift amount corresponding to the intersection bits.
const INT_MASK_SHIFT: u32 = INTERSECTION_MASK.count_ones();

/// The max number of lines per path.
///
/// Trying to render a path with more lines than this may result in visual artifacts.
pub const MAX_LINES_PER_PATH: u32 = 1 << (32 - INT_MASK_SHIFT);

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
    /// and winding data packed together.
    ///
    /// The layout is:
    /// - **Bits 0-4 (5 bits):** Intersection and Winding Mask (`W | R | L | B | T`).
    ///   - Bit 0 (mask `0b00001`): Intersects top edge (T)
    ///   - Bit 1 (mask `0b00010`): Intersects bottom edge (B)
    ///   - Bit 2 (mask `0b00100`): Intersects left edge (L)
    ///   - Bit 3 (mask `0b01000`): Intersects right edge (R)
    ///   - Bit 4 (mask `0b10000`): Winding (W) - 1 if crosses top edge.
    /// - **Bits 5-31 (27 bits):** The line index (`line_idx`).
    ///
    /// **Sorting Note:** The `line_idx` occupies the higher bits to ensure that when sorting
    /// tiles with the same (x, y) coordinates, they are sorted by their line index first,
    /// and then by their intersection mask.
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
    pub fn new_clamped(x: u16, y: u16, line_idx: u32, intersection_mask: u32) -> Self {
        Self::new(
            // Make sure that x and y stay in range when multiplying
            // with the tile width and height during strips generation.
            x.min(u16::MAX / Self::WIDTH),
            y.min(u16::MAX / Self::HEIGHT),
            line_idx,
            intersection_mask,
        )
    }

    /// The base tile constructor
    ///
    /// Unlike [`Self::new_clamped`], this constructor stores `x` and `y` exactly as provided.
    /// Callers must ensure these coordinates do not exceed the limits required by downstream
    /// processing (typically `u16::MAX / WIDTH` and `u16::MAX / HEIGHT`).
    #[inline]
    pub const fn new(x: u16, y: u16, line_idx: u32, intersection_mask: u32) -> Self {
        #[cfg(debug_assertions)]
        if line_idx >= MAX_LINES_PER_PATH {
            panic!("Max. number of lines per path exceeded.");
        }
        // The intersection_mask is expected to contain bits 0-4 (T, B, L, R, W).
        // We pack line_idx into the high bits (5-31) and intersection_mask into low bits (0-4).
        Self {
            x,
            y,
            packed_winding_line_idx: (line_idx << INT_MASK_SHIFT) | intersection_mask,
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
    ///
    /// Returns the high 27 bits.
    #[inline]
    pub const fn line_idx(&self) -> u32 {
        self.packed_winding_line_idx >> INT_MASK_SHIFT
    }

    /// Whether the line crosses the top edge of the tile.
    ///
    /// Lines making this crossing increment or decrement the coarse tile winding, depending on the
    /// line direction.
    ///
    /// Checks Bit 4 (Winding).
    #[inline]
    pub const fn winding(&self) -> bool {
        (self.packed_winding_line_idx & W) != 0
    }

    /// The 5 bits of intersection and winding data.
    #[inline]
    pub const fn intersection_mask(&self) -> u32 {
        self.packed_winding_line_idx & INTERSECTION_MASK
    }

    /// Whether the line intersects the top edge of the tile.
    #[inline]
    pub const fn intersects_top(&self) -> bool {
        (self.intersection_mask() & T) != 0
    }

    /// Whether the line intersects the bottom edge of the tile.
    #[inline]
    pub const fn intersects_bottom(&self) -> bool {
        (self.intersection_mask() & B) != 0
    }

    /// Whether the line intersects the left edge of the tile.
    #[inline]
    pub const fn intersects_left(&self) -> bool {
        (self.intersection_mask() & L) != 0
    }

    /// Whether the line intersects the right edge of the tile.
    #[inline]
    pub const fn intersects_right(&self) -> bool {
        (self.intersection_mask() & R) != 0
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

    /// Generates tile commands for Analytic Anti-Aliasing rasterization. Unlike the MSAA path, this
    /// function performs "coarse binning" to simply identify every tile a line segment traverses.
    /// It encodes the line index and winding direction, delegating the precise calculation of pixel
    /// coverage to `strip::render`.
    //
    // TODO: Tiles are clamped to the left edge of the viewport, but lines fully to the left of the
    // viewport are not culled yet. These lines impact winding, and would need forwarding of
    // winding to the strip generation stage.
    pub fn make_tiles_analytic_aa(&mut self, lines: &[Line], width: u16, height: u16) {
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

            // If y_top_tiles == y_bottom_tiles, then the line is either completely above or below
            // the viewport OR it is perfectly horizontal and aligned to the tile grid, contributing
            // no winding. In either case, it should be culled.
            if y_top_tiles >= y_bottom_tiles {
                // Technically, the `>` part of the `>=` is unnecessary due to clamping, but this
                // gives stronger signal
                continue;
            }

            // Get tile coordinates for start/end points, use i32 to preserve negative coordinates
            let p0_tile_x = line_top_x.floor() as i32;
            let p0_tile_y = line_top_y.floor() as i32;
            let p1_tile_x = line_bottom_x.floor() as i32;
            let p1_tile_y = line_bottom_y.floor() as i32;

            // Special-case out lines which are fully contained within a tile.
            let not_same_tile = p0_tile_y != p1_tile_y || p0_tile_x != p1_tile_x;
            if not_same_tile {
                // For ease of logic, special-case purely vertical tiles.
                if line_left_x == line_right_x {
                    let x = (line_left_x as u16).min(tile_columns.saturating_sub(1));

                    // Row Start, not culled.
                    let is_start_culled = line_top_y < 0.0;
                    if !is_start_culled {
                        let winding =
                            ((f32::from(y_top_tiles) >= line_top_y) as u32) << WINDING_SHIFT;
                        let tile = Tile::new_clamped(x, y_top_tiles, line_idx, winding);
                        self.tile_buf.push(tile);
                    }

                    // Middle
                    // If the start was culled, the first tile inside the viewport is a middle.
                    let y_start = if is_start_culled {
                        y_top_tiles
                    } else {
                        y_top_tiles + 1
                    };
                    let line_bottom_floor = line_bottom_y.floor();
                    let y_end_idx = (line_bottom_floor as u16).min(tile_rows);

                    for y_idx in y_start..y_end_idx {
                        let tile = Tile::new_clamped(x, y_idx, line_idx, W);
                        self.tile_buf.push(tile);
                    }

                    // Row End, handle the final tile (y_end_idx), but *only* if the line does
                    // not perfectly end on the top edge of the tile. In the case that it does,
                    // it gets handled by the middle logic above.
                    if line_bottom_y != line_bottom_floor && y_end_idx < tile_rows {
                        let tile = Tile::new_clamped(x, y_end_idx, line_idx, W);
                        self.tile_buf.push(tile);
                    }
                } else {
                    let dx = p1_x - p0_x;
                    let dy = p1_y - p0_y;
                    let x_slope = dx / dy;
                    let dx_dir = (line_bottom_x >= line_top_x) as u32;
                    let not_dx_dir = dx_dir ^ 1;

                    let w_start_base = dx_dir << WINDING_SHIFT;
                    let w_end_base = not_dx_dir << WINDING_SHIFT;

                    let mut push_row = |y_idx: u16,
                                        row_top_y: f32,
                                        row_bottom_y: f32,
                                        w_start: u32,
                                        w_end: u32,
                                        w_single: u32| {
                        let row_top_x = p0_x + (row_top_y - p0_y) * x_slope;
                        let row_bottom_x = p0_x + (row_bottom_y - p0_y) * x_slope;

                        let row_left_x = f32::min(row_top_x, row_bottom_x).max(line_left_x);
                        let row_right_x = f32::max(row_top_x, row_bottom_x).min(line_right_x);

                        let x_start = row_left_x as u16;
                        let x_end = (row_right_x as u16).min(tile_columns.saturating_sub(1));

                        if x_start <= x_end {
                            let winding = if x_start == x_end { w_single } else { w_start };

                            self.tile_buf
                                .push(Tile::new(x_start, y_idx, line_idx, winding));

                            for x_idx in x_start.saturating_add(1)..x_end {
                                self.tile_buf.push(Tile::new(x_idx, y_idx, line_idx, 0));
                            }

                            if x_start < x_end {
                                self.tile_buf.push(Tile::new(x_end, y_idx, line_idx, w_end));
                            }
                        }
                    };

                    let is_start_culled = line_top_y < 0.0;
                    if !is_start_culled {
                        let y = f32::from(y_top_tiles);
                        let row_bottom_y = (y + 1.0).min(line_bottom_y);
                        let mask = ((y >= line_top_y) as u32) << WINDING_SHIFT;
                        push_row(
                            y_top_tiles,
                            line_top_y,
                            row_bottom_y,
                            w_start_base & mask,
                            w_end_base & mask,
                            W & mask,
                        );
                    }

                    let y_start_middle = if is_start_culled {
                        y_top_tiles
                    } else {
                        y_top_tiles + 1
                    };

                    let line_bottom_floor = line_bottom_y.floor();
                    let y_end_middle = (line_bottom_floor as u16).min(tile_rows);
                    for y_idx in y_start_middle..y_end_middle {
                        let y = f32::from(y_idx);
                        let row_bottom_y = (y + 1.0).min(line_bottom_y);
                        push_row(y_idx, y, row_bottom_y, w_start_base, w_end_base, W);
                    }

                    if line_bottom_y != line_bottom_floor
                        && y_end_middle < tile_rows
                        && (is_start_culled || y_end_middle != y_top_tiles)
                    {
                        let y_idx = y_end_middle;
                        let y = f32::from(y_idx);
                        push_row(y_idx, y, line_bottom_y, w_start_base, w_end_base, W);
                    }
                }
            } else {
                // Case: Line is fully contained within a single tile.
                let tile = Tile::new_clamped(
                    (line_left_x as u16).min(tile_columns + 1),
                    y_top_tiles,
                    line_idx,
                    ((f32::from(y_top_tiles) >= line_top_y) as u32) << WINDING_SHIFT,
                );
                self.tile_buf.push(tile);
            }
        }
    }

    /// Generates tile commands for MSAA (Multisample Anti-Aliasing) rasterization.
    ///
    /// [ Architecture & Watertightness ]
    /// The primary goal of this function is to establish a source of "ground truth" for line-tile
    /// intersections. Because the downstream rasterization (MSAA) occurs in parallel, it is
    /// critical that intersections are "watertight." If Thread A handles Tile (0,0) and Thread B
    /// handles Tile (1,0), they must agree exactly on whether a line crosses the shared edge.
    ///
    /// While calculating exact intersection coordinates here is feasible, it is computationally
    /// expensive. Instead, we defer the heavy math to the GPU/rasterizer and produce a lightweight
    /// Intersection Bitmask. This mask unambiguously defines which edges of a tile a line segment
    /// touches or crosses.
    ///
    /// [ The Intersection Bitmask (5 bits) ]
    /// The bitmask encodes winding information and edge intersections. A line is said to
    /// "intersect" an edge if it touches that edge AND continues into the neighboring tile.
    ///
    /// Bit representation:
    /// Bit: 4 | 3 | 2 | 1 | 0
    /// Val: W | R | L | B | T
    ///
    /// - W (Winding): Tracks whether the line touched the top edge of the tile.
    /// - R/L/B/T: Right, Left, Bottom, and Top edge intersections.
    pub fn make_tiles_msaa(&mut self, lines: &[Line], width: u16, height: u16) {
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

            // If y_top_tiles == y_bottom_tiles, then the line is either completely above or below
            // the viewport OR it is perfectly horizontal and aligned to the tile grid, contributing
            // no winding. In either case, it should be culled.
            if y_top_tiles >= y_bottom_tiles {
                continue;
            }

            // Get tile coordinates for start/end points, use i32 to preserve negative coordinates
            let p0_tile_x = line_top_x.floor() as i32;
            let p0_tile_y = line_top_y.floor() as i32;
            let p1_tile_x = line_bottom_x.floor() as i32;

            let p1_tile_y = if line_bottom_y == line_bottom_y_ceil {
                line_bottom_y as i32 - 1
            } else {
                line_bottom_y.floor() as i32
            };

            // Special-case out lines which are fully contained within a tile.
            let not_same_tile = p0_tile_y != p1_tile_y || p0_tile_x != p1_tile_x;
            if not_same_tile {
                // For ease of logic, special-case purely vertical tiles.
                if line_left_x == line_right_x {
                    let x = (line_left_x as u16).min(tile_columns.saturating_sub(1));

                    // Row Start, not culled.
                    let is_start_culled = line_top_y < 0.0;
                    if !is_start_culled {
                        let winding =
                            ((f32::from(y_top_tiles) >= line_top_y) as u32) << WINDING_SHIFT;
                        let intersection_mask = B | winding;
                        let tile = Tile::new_clamped(x, y_top_tiles, line_idx, intersection_mask);
                        self.tile_buf.push(tile);
                    }

                    // Middle
                    // If the start was culled, the first tile inside the viewport is a middle.
                    let y_start = if is_start_culled {
                        y_top_tiles
                    } else {
                        y_top_tiles + 1
                    };
                    let line_bottom_floor = line_bottom_y.floor();
                    let y_end_idx = (line_bottom_floor as u16).min(tile_rows);

                    if y_start < y_end_idx {
                        let y_last = y_end_idx - 1;
                        for y_idx in y_start..y_last {
                            let intersection_mask = W | B | T;
                            let tile = Tile::new_clamped(x, y_idx, line_idx, intersection_mask);
                            self.tile_buf.push(tile);
                        }

                        // Perfect touching B case.
                        {
                            let is_end_tile = ((y_last as i32) == p1_tile_y) as u32;
                            let intersection_mask = W | T | ((1 ^ is_end_tile) << BOT_SHIFT);
                            let tile = Tile::new_clamped(x, y_last, line_idx, intersection_mask);
                            self.tile_buf.push(tile);
                        }
                    }

                    // Row End, handle the final tile (y_end_idx), but *only* if the line does
                    // not perfectly end on the top edge of the tile. In the case that it does,
                    // it gets handled by the middle logic above.
                    if line_bottom_y != line_bottom_floor && y_end_idx < tile_rows {
                        let intersection_mask = W | T;
                        let tile = Tile::new_clamped(x, y_end_idx, line_idx, intersection_mask);
                        self.tile_buf.push(tile);
                    }
                } else {
                    let dx = p1_x - p0_x;
                    let dy = p1_y - p0_y;
                    let x_slope = dx / dy;
                    let dx_dir = (line_bottom_x >= line_top_x) as u32;
                    let not_dx_dir = dx_dir ^ 1;

                    let w_start_base = dx_dir << WINDING_SHIFT;
                    let w_end_base = not_dx_dir << WINDING_SHIFT;

                    // Check if the line is fully within the horizontal viewport bounds. If it is,
                    // we can skip the min/max clamping per row.
                    let min_x = p0_x.min(p1_x);
                    let max_x = p0_x.max(p1_x);
                    // Note: We use >= on the right edge to ensure strictly safe integer truncation
                    let needs_clamping = min_x < line_left_x || max_x >= line_right_x;

                    // Handles the bitmask logic for start/end tiles. Invariant to clamping.
                    macro_rules! push_edge {
                        ($x:expr, $y:expr, $row_top_x:expr, $row_bottom_x:expr,
                         $canonical_start:expr, $canonical_end:expr, $winding_input:expr,
                         $check_s:tt, $check_e:expr) => {{
                            let x_idx = $x;

                            let unc_row_start = (x_idx as i32 == $canonical_start) as u32;
                            let unc_row_end = (x_idx == $canonical_end) as u32;

                            let canonical_row_start =
                                (dx_dir & unc_row_start) | (not_dx_dir & unc_row_end);
                            let canonical_row_end =
                                (not_dx_dir & unc_row_start) | (dx_dir & unc_row_end);

                            let start_tile = if $check_s {
                                ((x_idx as i32 == p0_tile_x) && ($y as i32 == p0_tile_y)) as u32
                            } else {
                                0
                            };

                            let end_tile = if $check_e {
                                ((x_idx as i32 == p1_tile_x) && ($y as i32 == p1_tile_y)) as u32
                            } else {
                                0
                            };

                            let mut mask = $winding_input;

                            // Entrant/Exit
                            mask |= canonical_row_start & (1 ^ start_tile);
                            mask |= (1 ^ canonical_row_start) << not_dx_dir << LEFT_SHIFT;
                            mask |= (canonical_row_end & (1 ^ end_tile)) << BOT_SHIFT;
                            mask |= (1 ^ canonical_row_end) << dx_dir << LEFT_SHIFT;

                            // Corner
                            let x_left_f = x_idx as f32;
                            let x_right_f = (x_idx + 1) as f32;
                            let trc = (($row_top_x == x_right_f) as u32) & (1 ^ start_tile);
                            let tlc = (($row_top_x == x_left_f) as u32) & (1 ^ start_tile);
                            let brc = (($row_bottom_x == x_right_f) as u32) & (1 ^ end_tile);
                            let blc = (($row_bottom_x == x_left_f) as u32) & (1 ^ end_tile);
                            // Top left is handled specially
                            let tie_break = tlc & (canonical_row_start ^ 1);

                            mask |= (tie_break | blc) << LEFT_SHIFT;
                            mask |= (trc | brc) << RIGHT_SHIFT;
                            mask &= !(tie_break | trc);
                            mask &= !((blc | brc) << BOT_SHIFT);

                            self.tile_buf.push(Tile::new(x_idx, $y, line_idx, mask));
                        }};
                    }

                    // Handles row geometry and clamping logic.
                    macro_rules! process_row {
                        ($y_idx:expr, $row_top_y:expr, $row_bottom_y:expr, $w_mask:expr,
                     $check_s:tt, $check_e:tt, $clamped:tt) => {{
                            let row_top_x = p0_x + ($row_top_y - p0_y) * x_slope;
                            let row_bottom_x = p0_x + ($row_bottom_y - p0_y) * x_slope;

                            let (row_left_x, row_right_x, x_end) = if $clamped {
                                let lx = f32::min(row_top_x, row_bottom_x).max(line_left_x);
                                let rx = f32::max(row_top_x, row_bottom_x).min(line_right_x);
                                let xe = (rx as u16).min(tile_columns.saturating_sub(1));
                                (lx, rx, xe)
                            } else {
                                let lx = f32::min(row_top_x, row_bottom_x);
                                let rx = f32::max(row_top_x, row_bottom_x);
                                let xe = rx as u16; // Safe because we checked bounds earlier
                                (lx, rx, xe)
                            };

                            let canonical_x_start = row_left_x.floor() as i32;
                            let canonical_x_end = row_right_x as u16;
                            let x_start = row_left_x as u16;

                            if x_start <= x_end {
                                let is_single = (x_start == x_end) as u32;
                                let w_left = (w_start_base | (is_single << 4)) & $w_mask;

                                push_edge!(
                                    x_start,
                                    $y_idx,
                                    row_top_x,
                                    row_bottom_x,
                                    canonical_x_start,
                                    canonical_x_end,
                                    w_left,
                                    $check_s,
                                    $check_e
                                );

                                for x_idx in x_start.saturating_add(1)..x_end {
                                    self.tile_buf
                                        .push(Tile::new(x_idx, $y_idx, line_idx, R | L));
                                }

                                if x_start < x_end {
                                    let w_right = w_end_base & $w_mask;
                                    push_edge!(
                                        x_end,
                                        $y_idx,
                                        row_top_x,
                                        row_bottom_x,
                                        canonical_x_start,
                                        canonical_x_end,
                                        w_right,
                                        $check_s,
                                        $check_e
                                    );
                                }
                            }
                        }};
                    }

                    // Central macro
                    macro_rules! run_loops {
                        ($clamped:tt) => {{
                            // Top Row
                            let is_start_culled = line_top_y < 0.0;
                            if !is_start_culled {
                                let y = f32::from(y_top_tiles);
                                let row_bottom_y = (y + 1.0).min(line_bottom_y);
                                let mask = ((y >= line_top_y) as u32) << WINDING_SHIFT;
                                process_row!(
                                    y_top_tiles,
                                    line_top_y,
                                    row_bottom_y,
                                    mask,
                                    true,
                                    true,
                                    $clamped
                                );
                            }

                            let y_start_middle = if is_start_culled {
                                y_top_tiles
                            } else {
                                y_top_tiles + 1
                            };
                            let line_bottom_floor = line_bottom_y.floor();
                            let y_end_middle = (line_bottom_floor as u16).min(tile_rows);
                            let has_separate_bottom_row = line_bottom_y != line_bottom_floor
                                && y_end_middle < tile_rows
                                && (is_start_culled || y_end_middle != y_top_tiles);

                            if y_start_middle < y_end_middle {
                                for y_idx in y_start_middle..y_end_middle {
                                    let y = f32::from(y_idx);
                                    let row_bottom_y = (y + 1.0).min(line_bottom_y);
                                    let is_last_middle = y_idx == y_end_middle - 1;
                                    let check_end = is_last_middle && !has_separate_bottom_row;

                                    process_row!(
                                        y_idx,
                                        y,
                                        row_bottom_y,
                                        u32::MAX,
                                        false,
                                        check_end,
                                        $clamped
                                    );
                                }
                            }

                            // Bottom Row
                            if has_separate_bottom_row {
                                let y_idx = y_end_middle;
                                let y = f32::from(y_idx);
                                process_row!(
                                    y_idx,
                                    y,
                                    line_bottom_y,
                                    u32::MAX,
                                    false,
                                    true,
                                    $clamped
                                );
                            }
                        }};
                    }

                    if needs_clamping {
                        run_loops!(true);
                    } else {
                        run_loops!(false);
                    }
                }
            } else {
                // Case: Line is fully contained within a single tile.
                let tile = Tile::new_clamped(
                    (line_left_x as u16).min(tile_columns + 1),
                    y_top_tiles,
                    line_idx,
                    ((f32::from(y_top_tiles) >= line_top_y) as u32) << WINDING_SHIFT,
                );
                self.tile_buf.push(tile);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::flatten::{FlattenCtx, Line, Point, fill};
    use crate::kurbo::{Affine, BezPath};
    use crate::tile::{B, L, R, T, Tile, Tiles, W};
    use fearless_simd::Level;
    use std::vec;

    const VIEW_DIM: u16 = 100;
    const F_V_DIM: f32 = VIEW_DIM as f32;

    impl Tiles {
        fn assert_tiles_match(
            &mut self,
            lines: &[Line],
            width: u16,
            height: u16,
            expected: &[Tile],
        ) {
            self.make_tiles_msaa(lines, width, height);
            assert_eq!(self.tile_buf, expected, "MSAA: Tile buffer mismatch");

            self.make_tiles_analytic_aa(lines, width, height);
            check_analytic_aa_matches(&self.tile_buf, expected);
        }
    }

    fn check_analytic_aa_matches(actual: &[Tile], expected: &[Tile]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "Analytic AA: Tile count mismatch."
        );

        for (i, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got.x, want.x, "Analytic AA: Tile[{}] X mismatch", i);
            assert_eq!(got.y, want.y, "Analytic AA: Tile[{}] Y mismatch", i);
            assert_eq!(
                got.line_idx(),
                want.line_idx(),
                "Analytic AA: Tile[{}] Line Index mismatch",
                i
            );

            let got_winding = got.packed_winding_line_idx & W;
            let want_winding = want.packed_winding_line_idx & W;
            assert_eq!(
                got_winding, want_winding,
                "Analytic AA: Tile[{}] Winding mismatch",
                i
            );
        }
    }

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
                p0: Point {
                    x: F_V_DIM + 1.0,
                    y: 50.0,
                },
                p1: Point {
                    x: F_V_DIM + 3.0,
                    y: 70.0,
                },
            },
            Line {
                p0: Point {
                    x: 1.0,
                    y: F_V_DIM + 1.0,
                },
                p1: Point {
                    x: 3.0,
                    y: F_V_DIM + 7.0,
                },
            },
            Line {
                p0: Point {
                    x: 1.0,
                    y: F_V_DIM + 1.0,
                },
                p1: Point {
                    x: 3.0,
                    y: F_V_DIM + 13.0,
                },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &[]);
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
        let expected = [
            Tile::new(0, 0, 0, W | T),
            Tile::new(1, 0, 1, W | T),
            Tile::new(2, 0, 2, W | T),
            Tile::new(0, 0, 3, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn sloped_line_crossing_bot() {
        let lines = [
            Line {
                p0: Point {
                    x: 5.0,
                    y: F_V_DIM + 3.0,
                },
                p1: Point {
                    x: 6.0,
                    y: F_V_DIM - 2.0,
                },
            },
            Line {
                p0: Point {
                    x: 10.0,
                    y: F_V_DIM + 1.0,
                },
                p1: Point {
                    x: 9.0,
                    y: F_V_DIM - 1.0,
                },
            },
            Line {
                p0: Point {
                    x: 2.0,
                    y: F_V_DIM - 2.0,
                },
                p1: Point {
                    x: 3.0,
                    y: F_V_DIM + 3.0,
                },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(1, 24, 0, B),
            Tile::new(2, 24, 1, B),
            Tile::new(0, 24, 2, B),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
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
        let expected = [
            Tile::new(0, 0, 0, W | T | R),
            Tile::new(1, 0, 0, L | B),
            Tile::new(1, 1, 0, W | T),
            Tile::new(0, 0, 1, W | T | B),
            Tile::new(0, 1, 1, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn sloped_line_crossing_bot_multi_tile() {
        let lines = [
            Line {
                p0: Point {
                    x: 12.0,
                    y: F_V_DIM + 10.0,
                },
                p1: Point { x: 2.0, y: 94.0 },
            },
            Line {
                p0: Point {
                    x: 1.5,
                    y: F_V_DIM + 5.0,
                },
                p1: Point { x: 3.5, y: 94.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 23, 0, B),
            Tile::new(0, 24, 0, W | T | R),
            Tile::new(1, 24, 0, B | L),
            Tile::new(0, 23, 1, B),
            Tile::new(0, 24, 1, W | T | B),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn sloped_line_crossing_right() {
        let lines = [
            Line {
                p0: Point { x: 97.0, y: 1.0 },
                p1: Point {
                    x: F_V_DIM + 1.0,
                    y: 2.0,
                },
            },
            Line {
                p0: Point { x: 93.0, y: 1.0 },
                p1: Point {
                    x: F_V_DIM + 5.0,
                    y: 2.0,
                },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(24, 0, 0, R),
            Tile::new(23, 0, 1, R),
            Tile::new(24, 0, 1, R | L),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
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
            Line {
                p0: Point { x: -5.0, y: 1.0 },
                p1: Point { x: 13.0, y: 9.0 },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, L),
            Tile::new(0, 0, 1, L | R),
            Tile::new(1, 0, 1, L),
            Tile::new(0, 0, 2, L | B),
            Tile::new(0, 1, 2, W | R | T),
            Tile::new(1, 1, 2, R | L),
            Tile::new(2, 1, 2, L | B),
            Tile::new(2, 2, 2, W | R | T),
            Tile::new(3, 2, 2, L),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn horizontal_line_above_viewport() {
        let lines = [Line {
            p0: Point { x: 10.0, y: -5.0 },
            p1: Point { x: 90.0, y: -5.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &[]);
    }

    #[test]
    fn horizontal_line_below_viewport() {
        let lines = [Line {
            p0: Point {
                x: 10.0,
                y: F_V_DIM + 5.0,
            },
            p1: Point {
                x: 90.0,
                y: F_V_DIM + 5.0,
            },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &[]);
    }

    #[test]
    fn horizontal_line_crossing_left_viewport() {
        let lines = [Line {
            p0: Point { x: -10.0, y: 10.0 },
            p1: Point { x: 10.0, y: 10.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 2, 0, L | R),
            Tile::new(1, 2, 0, L | R),
            Tile::new(2, 2, 0, L),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn horizontal_line_crossing_right_viewport() {
        let lines = [Line {
            p0: Point {
                x: F_V_DIM - 5.0,
                y: 10.0,
            },
            p1: Point {
                x: F_V_DIM + 5.0,
                y: 10.0,
            },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [Tile::new(23, 2, 0, R), Tile::new(24, 2, 0, L | R)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
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
                    y: F_V_DIM + 1.0,
                },
                p1: Point {
                    x: 1.0,
                    y: F_V_DIM + 5.0,
                },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &[]);
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
        tiles.assert_tiles_match(&line_buf, 10, 10, &[]);
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
        let expected = [
            Tile::new(0, 0, 0, W | T),
            Tile::new(0, 0, 1, W | B | T),
            Tile::new(0, 1, 1, W | T),
            Tile::new(0, 0, 2, W | B | T),
            Tile::new(0, 1, 2, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn vertical_line_crossing_bot_viewport() {
        let lines = [
            Line {
                p0: Point {
                    x: 1.0,
                    y: F_V_DIM - 1.0,
                },
                p1: Point {
                    x: 1.0,
                    y: F_V_DIM + 5.0,
                },
            },
            Line {
                p0: Point {
                    x: 1.0,
                    y: F_V_DIM - 5.0,
                },
                p1: Point {
                    x: 1.0,
                    y: F_V_DIM + 5.0,
                },
            },
        ];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 24, 0, B),
            Tile::new(0, 23, 1, B),
            Tile::new(0, 24, 1, W | T | B),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn clip_top_left_corner() {
        let lines = [Line {
            p0: Point { x: -1.0, y: 2.0 },
            p1: Point { x: 2.0, y: -1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [Tile::new(0, 0, 0, W | L | T)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn clip_bottom_right_corner() {
        let lines = [Line {
            p0: Point {
                x: F_V_DIM + 1.0,
                y: F_V_DIM - 2.0,
            },
            p1: Point {
                x: F_V_DIM - 2.0,
                y: F_V_DIM + 1.0,
            },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [Tile::new(24, 24, 0, R | B)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
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
        let expected = [
            Tile::new(0, 0, 0, R),
            Tile::new(1, 0, 0, R | L),
            Tile::new(2, 0, 0, L),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn horizontal_line_right_to_left_three_tile() {
        let lines = [Line {
            p0: Point { x: 8.5, y: 1.0 },
            p1: Point { x: 1.5, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, R),
            Tile::new(1, 0, 0, R | L),
            Tile::new(2, 0, 0, L),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn horizontal_line_multi_tile() {
        let lines = [Line {
            p0: Point { x: 1.5, y: 1.0 },
            p1: Point { x: 12.5, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, R),
            Tile::new(1, 0, 0, R | L),
            Tile::new(2, 0, 0, R | L),
            Tile::new(3, 0, 0, L),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn vertical_line_down_three_tile() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.5 },
            p1: Point { x: 1.0, y: 8.5 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, B),
            Tile::new(0, 1, 0, W | T | B),
            Tile::new(0, 2, 0, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn vertical_line_down_multi_tile() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.0 },
            p1: Point { x: 1.0, y: 13.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, B),
            Tile::new(0, 1, 0, W | T | B),
            Tile::new(0, 2, 0, W | T | B),
            Tile::new(0, 3, 0, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn vertical_line_up_three_tile() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 13.0 },
            p1: Point { x: 1.0, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, B),
            Tile::new(0, 1, 0, W | T | B),
            Tile::new(0, 2, 0, W | T | B),
            Tile::new(0, 3, 0, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn vertical_line_up_multi_tile() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 8.5 },
            p1: Point { x: 1.0, y: 1.5 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, B),
            Tile::new(0, 1, 0, W | T | B),
            Tile::new(0, 2, 0, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    // Exclusive to the bottom edge, no P required.
    #[test]
    fn vertical_line_touching_bot() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.0 },
            p1: Point { x: 1.0, y: 8.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [Tile::new(0, 0, 0, B), Tile::new(0, 1, 0, W | T)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn vertical_line_touching_top() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 0.0 },
            p1: Point { x: 1.0, y: 7.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [Tile::new(0, 0, 0, W | B), Tile::new(0, 1, 0, W | T)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
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
        let expected = [
            Tile::new(0, 0, 0, R),
            Tile::new(1, 0, 0, L | B),
            Tile::new(1, 1, 0, W | R | T),
            Tile::new(2, 1, 0, L | B),
            Tile::new(2, 2, 0, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn bottom_right_to_top_left() {
        let lines = [Line {
            p0: Point { x: 11.0, y: 9.0 },
            p1: Point { x: 1.0, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, R),
            Tile::new(1, 0, 0, L | B),
            Tile::new(1, 1, 0, W | R | T),
            Tile::new(2, 1, 0, L | B),
            Tile::new(2, 2, 0, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn bottom_left_to_top_right() {
        let lines = [Line {
            p0: Point { x: 2.0, y: 11.0 },
            p1: Point { x: 14.0, y: 6.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(2, 1, 0, R | B),
            Tile::new(3, 1, 0, L),
            Tile::new(0, 2, 0, R),
            Tile::new(1, 2, 0, R | L),
            Tile::new(2, 2, 0, W | L | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn top_right_to_bottom_left() {
        let lines = [Line {
            p0: Point { x: 14.0, y: 6.0 },
            p1: Point { x: 2.0, y: 11.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(2, 1, 0, R | B),
            Tile::new(3, 1, 0, L),
            Tile::new(0, 2, 0, R),
            Tile::new(1, 2, 0, R | L),
            Tile::new(2, 2, 0, W | L | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
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
        let expected = [Tile::new(0, 0, 0, 0), Tile::new(0, 0, 1, 0)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn diagonal_cross_corner() {
        let lines = [Line {
            p0: Point { x: 3.0, y: 5.0 },
            p1: Point { x: 5.0, y: 3.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(1, 0, 0, L),
            Tile::new(0, 1, 0, R),
            Tile::new(1, 1, 0, W | L | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn diagonal_cross_corner_two() {
        let lines = [Line {
            p0: Point { x: 7.9, y: 7.9 },
            p1: Point { x: 0.1, y: 0.1 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, R),
            Tile::new(1, 0, 0, L),
            Tile::new(1, 1, 0, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn diagonal_down_slope_tiles() {
        let lines = [Line {
            p0: Point { x: 5.0, y: 5.0 },
            p1: Point { x: 9.0, y: 9.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(1, 1, 0, R),
            Tile::new(2, 1, 0, L),
            Tile::new(2, 2, 0, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn diagonal_up_slope_tiles() {
        let lines = [Line {
            p0: Point { x: 5.0, y: 9.0 },
            p1: Point { x: 9.0, y: 5.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(1, 1, 0, R | B),
            Tile::new(2, 1, 0, L),
            Tile::new(1, 2, 0, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn diagonal_down_one_tile() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 0.0 },
            p1: Point { x: 4.0, y: 4.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [Tile::new(0, 0, 0, W | R), Tile::new(1, 0, 0, L)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn diagonal_up_one_tile() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 4.0 },
            p1: Point { x: 4.0, y: 0.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [Tile::new(0, 0, 0, R), Tile::new(1, 0, 0, W | L)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn diagonal_down_two_tile() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 0.0 },
            p1: Point { x: 8.0, y: 8.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, W | R),
            Tile::new(1, 0, 0, L),
            Tile::new(1, 1, 0, W | R | T),
            Tile::new(2, 1, 0, L),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn diagonal_up_two_tile() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 8.0 },
            p1: Point { x: 8.0, y: 0.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(1, 0, 0, R | L),
            Tile::new(2, 0, 0, W | L),
            Tile::new(0, 1, 0, R),
            Tile::new(1, 1, 0, W | L | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn sloped_ending_right() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.0 },
            p1: Point { x: 8.0, y: 2.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, R),
            Tile::new(1, 0, 0, R | L),
            Tile::new(2, 0, 0, L),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn sloped_touching_top() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 8.0 },
            p1: Point { x: 4.0, y: 0.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, R | B),
            Tile::new(1, 0, 0, W | L),
            Tile::new(0, 1, 0, W | T),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn sloped_touching_bot() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 0.0 },
            p1: Point { x: 4.0, y: 8.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [
            Tile::new(0, 0, 0, W | B),
            Tile::new(0, 1, 0, W | R | T),
            Tile::new(1, 1, 0, L),
        ];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
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
        let expected = [Tile::new(0, 0, 0, 0)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn same_tile_left() {
        let lines = [Line {
            p0: Point { x: 0.0, y: 1.0 },
            p1: Point { x: 3.0, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [Tile::new(0, 0, 0, 0)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn same_tile_top() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 0.0 },
            p1: Point { x: 1.0, y: 3.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [Tile::new(0, 0, 0, W)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    #[test]
    fn same_tile_right() {
        let lines = [Line {
            p0: Point { x: 1.0, y: 1.0 },
            p1: Point { x: 4.0, y: 1.0 },
        }];

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        let expected = [Tile::new(0, 0, 0, R), Tile::new(1, 0, 0, L)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
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
        let expected = [Tile::new(0, 0, 0, 0), Tile::new(0, 0, 1, 0)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
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
        let expected = [Tile::new(0, 0, 0, W), Tile::new(0, 0, 1, W)];

        tiles.assert_tiles_match(&lines, VIEW_DIM, VIEW_DIM, &expected);
    }

    //==============================================================================================
    // Miscellaneous Cases
    //==============================================================================================
    #[test]
    // See https://github.com/LaurenzV/cpu-sparse-experiments/issues/46.
    fn infinite_loop() {
        let line = Line {
            p0: Point { x: 22.0, y: 552.0 },
            p1: Point { x: 224.0, y: 388.0 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles_msaa(&[line], 600, 600);
        tiles.make_tiles_analytic_aa(&[line], 600, 600);
    }

    #[test]
    // See https://github.com/linebender/vello/issues/1321
    fn overflow() {
        let line = Line {
            p0: Point {
                x: 59.60001,
                y: 40.78,
            },
            p1: Point {
                x: 520599.6,
                y: 100.18,
            },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles_analytic_aa(&[line], 200, 100);
        tiles.make_tiles_msaa(&[line], 200, 100);
    }

    #[test]
    fn sort_test() {
        let mut lines = vec![];
        let mut tiles = Tiles::new(Level::fallback());

        let step = 4.0;
        let mut y = F_V_DIM - 10.0;
        while y > 10.0 {
            lines.push(Line {
                p0: Point {
                    x: F_V_DIM - 10.0,
                    y,
                },
                p1: Point { x: 10.0, y },
            });

            lines.push(Line {
                p0: Point {
                    x: F_V_DIM - 12.0,
                    y,
                },
                p1: Point { x: 12.0, y },
            });

            y -= step;
        }

        tiles.make_tiles_msaa(&lines, VIEW_DIM, VIEW_DIM);
        assert!(tiles.tile_buf.first().unwrap().y > tiles.tile_buf.last().unwrap().y);
        tiles.sort_tiles();
        check_sorted(&tiles.tile_buf);

        tiles.make_tiles_analytic_aa(&lines, VIEW_DIM, VIEW_DIM);
        assert!(tiles.tile_buf.first().unwrap().y > tiles.tile_buf.last().unwrap().y);
        tiles.sort_tiles();
        check_sorted(&tiles.tile_buf);
    }

    fn check_sorted(buf: &[Tile]) {
        for i in 0..buf.len() - 1 {
            let current = buf[i];
            let next = buf[i + 1];

            if current.y > next.y {
                panic!(
                    "Sort Failure [Y]: Tile[{}] (y={}) > Tile[{}] (y={})",
                    i,
                    current.y,
                    i + 1,
                    next.y
                );
            }

            if current.y == next.y {
                if current.x > next.x {
                    panic!(
                        "Sort Failure [X]: at Row y={}, Tile[{}] (x={}) > Tile[{}] (x={})",
                        current.y,
                        i,
                        current.x,
                        i + 1,
                        next.x
                    );
                }

                if current.x == next.x
                    && current.packed_winding_line_idx > next.packed_winding_line_idx
                {
                    panic!(
                        "Sort Failure [Payload]: at {}x{}, Tile[{}] (val={}) > Tile[{}] (val={})",
                        current.x,
                        current.y,
                        i,
                        current.packed_winding_line_idx,
                        i + 1,
                        next.packed_winding_line_idx
                    );
                }
            }
        }
    }
}
