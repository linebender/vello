// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Primitives for creating tiles.

use crate::flatten::Line;

/// The max number of lines per path.
///
/// Trying to render a path with more lines than this may result in visual artifacts.
pub const MAX_LINES_PER_PATH: u32 = 1 << 31;

/// A tile represents an aligned area on the pixmap, used to subdivide the viewport into sub-areas
/// (currently 4x4) and analyze line intersections inside each such area.
///
/// Keep in mind that it is possible to have multiple tiles with the same index,
/// namely if we have multiple lines crossing the same 4x4 area!
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
#[repr(C)]
pub struct Tile(
    /// The packed tile data.
    ///
    /// The layout is as follows, with the bit indices from least to most significant.
    ///
    /// [ 0, 31): The 31-bit index of the line this tile belongs to into the line buffer, plus
    /// whether the line crosses the top edge of the tile, packed together;
    /// [31, 32): The 1-bit coarse winding of the tile. This is `1` if and only if the lines
    /// crosses the tile's top edge. Lines making this crossing increment or decrement the coarse
    /// tile winding, depending on the line direction;
    /// [32, 48): The 16-bit x-coordinate; and
    /// [48, 64): The 16-bit y-coordinate.
    ///
    /// Note the byte layout in memory depends on the endianness of the compilation target.
    pub u64,
);

impl Tile {
    /// The width of a tile in pixels.
    pub const WIDTH: u16 = 4;

    /// The height of a tile in pixels.
    pub const HEIGHT: u16 = 4;

    /// The x-coordinate of this tile, in tiles.
    #[inline]
    pub const fn x(&self) -> u16 {
        // This cast explicitly overflows, dropping the high order bits.
        (self.0 >> 32) as u16
    }

    /// The y-coordinate of this tile, in tiles.
    #[inline]
    pub const fn y(&self) -> u16 {
        (self.0 >> 48) as u16
    }

    /// Create a new tile.
    #[inline]
    pub const fn new(x: u16, y: u16, line_idx: u32, winding: bool) -> Self {
        #[cfg(debug_assertions)]
        if line_idx >= MAX_LINES_PER_PATH {
            panic!("Max. number of lines per path exceeded.");
        }
        Self(((y as u64) << 48) | ((x as u64) << 32) | ((winding as u64) << 31) | (line_idx as u64))
    }

    /// Check whether two tiles are at the same location.
    #[inline]
    pub const fn same_loc(&self, other: &Self) -> bool {
        self.same_row(other) && self.x() == other.x()
    }

    /// Check whether `self` is adjacent to the left of `other`.
    #[inline]
    pub const fn prev_loc(&self, other: &Self) -> bool {
        self.same_row(other) && self.x() + 1 == other.x()
    }

    /// Check whether two tiles are on the same row.
    #[inline]
    pub const fn same_row(&self, other: &Self) -> bool {
        self.y() == other.y()
    }

    #[inline(never)]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }

    /// The index of the line this tile belongs to into the line buffer.
    #[inline]
    pub const fn line_idx(&self) -> u32 {
        // This cast explicitly overflows, dropping the high order bits.
        (self.0 as u32) & ((1 << 31) - 1)
    }

    /// Whether the line crosses the top edge of the tile.
    ///
    /// Lines making this crossing increment or decrement the coarse tile winding, depending on the
    /// line direction.
    #[inline]
    pub const fn winding(&self) -> bool {
        (self.0 & (1 << 31)) != 0
    }
}

/// Handles the tiling of paths.
#[derive(Clone, Debug)]
pub struct Tiles {
    tile_buf: Vec<Tile>,
    sorted: bool,
}

impl Default for Tiles {
    fn default() -> Self {
        Self::new()
    }
}

impl Tiles {
    /// Create a new tiles container.
    pub fn new() -> Self {
        Self {
            tile_buf: vec![],
            sorted: false,
        }
    }

    /// Get the number of tiles in the container.
    pub fn len(&self) -> u32 {
        self.tile_buf.len() as u32
    }

    /// Returns true if the container has no tiles.
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
        self.tile_buf.sort_unstable_by(Tile::cmp);
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

    /// Populate the tiles' container with a buffer of lines.
    ///
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

        for (line_idx, line) in lines
            .iter()
            .take((MAX_LINES_PER_PATH as usize).saturating_add(1))
            .enumerate()
        {
            let line_idx = line_idx as u32;

            let p0_x = line.p0.x / Tile::WIDTH as f32;
            let p0_y = line.p0.y / Tile::HEIGHT as f32;
            let p1_x = line.p1.x / Tile::WIDTH as f32;
            let p1_y = line.p1.y / Tile::HEIGHT as f32;

            let (line_left_x, line_right_x) = if p0_x < p1_x {
                (p0_x, p1_x)
            } else {
                (p1_x, p0_x)
            };
            let (line_top_y, line_top_x, line_bottom_y, line_bottom_x) = if p0_y < p1_y {
                (p0_y, p0_x, p1_y, p1_x)
            } else {
                (p1_y, p1_x, p0_y, p0_x)
            };

            // For ease of logic, special-case purely vertical tiles.
            if line_left_x == line_right_x {
                let y_top_tiles = (line_top_y as u16).min(tile_rows);
                let y_bottom_tiles = (line_bottom_y.ceil() as u16).min(tile_rows);

                let x = line_left_x as u16;
                for y_idx in y_top_tiles..y_bottom_tiles {
                    let y = y_idx as f32;

                    let tile = Tile::new(x, y_idx, line_idx, y >= line_top_y);
                    self.tile_buf.push(tile);
                }
            } else {
                let x_slope = (p1_x - p0_x) / (p1_y - p0_y);

                let y_top_tiles = (line_top_y as u16).min(tile_rows);
                let y_bottom_tiles = (line_bottom_y.ceil() as u16).min(tile_rows);

                for y_idx in y_top_tiles..y_bottom_tiles {
                    let y = y_idx as f32;

                    // The line's y-coordinates at the line's top-and bottom-most points within the
                    // tile row.
                    let line_row_top_y = line_top_y.max(y).min(y + 1.);
                    let line_row_bottom_y = line_bottom_y.max(y).min(y + 1.);

                    // The line's x-coordinates at the line's top- and bottom-most points within the
                    // tile row.
                    let line_row_top_x = p0_x + (line_row_top_y - p0_y) * x_slope;
                    let line_row_bottom_x = p0_x + (line_row_bottom_y - p0_y) * x_slope;

                    // The line's x-coordinates at the line's left- and right-most points within the
                    // tile row.
                    let line_row_left_x =
                        f32::min(line_row_top_x, line_row_bottom_x).max(line_left_x);
                    let line_row_right_x =
                        f32::max(line_row_top_x, line_row_bottom_x).min(line_right_x);

                    let winding_x = if line_top_x < line_bottom_x {
                        line_row_left_x as u16
                    } else {
                        line_row_right_x as u16
                    };

                    for x_idx in
                        line_row_left_x as u16..=(line_row_right_x as u16).min(tile_columns - 1)
                    {
                        let tile = Tile::new(
                            x_idx,
                            y_idx,
                            line_idx,
                            y >= line_top_y && x_idx == winding_x,
                        );
                        self.tile_buf.push(tile);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
const _: () = if Tile::WIDTH != 4 || Tile::HEIGHT != 4 {
    panic!("Can only handle 4x4 tiles for now.");
};

#[cfg(test)]
mod tests {
    use crate::flatten::{Line, Point};
    use crate::tile::Tiles;

    #[test]
    fn issue_46_infinite_loop() {
        let line = Line {
            p0: Point { x: 22.0, y: 552.0 },
            p1: Point { x: 224.0, y: 388.0 },
        };

        let mut tiles = Tiles::new();
        tiles.make_tiles(&[line], 600, 600);
    }
}
