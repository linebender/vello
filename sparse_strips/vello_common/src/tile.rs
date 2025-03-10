// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Primitives for creating tiles.

use crate::flatten::Line;

/// A tile represents an aligned area on the pixmap, used to subdivide the viewport into sub-areas
/// (currently 4x4) and analyze line intersections inside each such area.
///
/// Keep in mind that it is possible to have multiple tiles with the same index,
/// namely if we have multiple lines crossing the same 4x4 area!
#[derive(Debug, Clone, Copy)]
pub struct Tile {
    /// The index of the tile in the x direction.
    pub x: i32,
    /// The index of the tile in the y direction.
    pub y: u16,
    /// The index of the line this tile belongs to into the line buffer.
    pub line_idx: u32,
    /// Whether the line crosses the top edge of the tile.
    ///
    /// Lines making this crossing increment or decrement the coarse tile winding, depending on the
    /// line direction.
    pub winding: bool,
}

impl Tile {
    /// The width of a tile in pixels.
    pub const WIDTH: u16 = 4;

    /// The height of a tile in pixels.
    pub const HEIGHT: u16 = 4;

    /// Create a new tile.
    pub fn new(x: i32, y: u16, line_idx: u32, winding: bool) -> Self {
        Self {
            x,
            y,
            line_idx,
            winding,
        }
    }

    /// Check whether two tiles are at the same location.
    pub fn same_loc(&self, other: &Self) -> bool {
        self.x == other.x && self.same_row(other)
    }

    /// Check whether `self` is adjacent to the left of `other`.
    pub fn prev_loc(&self, other: &Self) -> bool {
        self.same_row(other) && self.x + 1 == other.x
    }

    /// Check whether two tiles are on the same row.
    pub fn same_row(&self, other: &Self) -> bool {
        self.y == other.y
    }
}

/// Handles the tiling of paths.
#[derive(Clone, Debug)]
pub struct Tiles {
    tile_buf: Vec<Tile>,
    tile_index_buf: Vec<TileIndex>,
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
            tile_index_buf: vec![],
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
        self.tile_index_buf.clear();
        self.sorted = false;
    }

    /// Sort the tiles in the container.
    pub fn sort_tiles(&mut self) {
        self.sorted = true;
        self.tile_index_buf.sort_unstable_by(TileIndex::cmp);
    }

    /// Get the tile at a certain index.
    ///
    /// Panics if the container hasn't been sorted before.
    pub fn get(&self, index: u32) -> &Tile {
        assert!(
            self.sorted,
            "attempted to call `get` before sorting the tile container."
        );

        &self.tile_buf[self.tile_index_buf[index as usize].index()]
    }

    /// Iterate over the tiles in sorted order.
    ///
    /// Panics if the container hasn't been sorted before.
    pub fn iter(&self) -> impl Iterator<Item = &Tile> {
        assert!(
            self.sorted,
            "attempted to call `iter` before sorting the tile container."
        );

        self.tile_index_buf
            .iter()
            .map(|idx| &self.tile_buf[idx.index()])
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
            lines.len() < (u32::MAX as usize).saturating_add(1),
            "Max. number of lines per path exceeded. Max is {}, got {}.",
            u32::MAX,
            lines.len()
        );

        let tile_columns = width.div_ceil(Tile::WIDTH);
        let tile_rows = height.div_ceil(Tile::HEIGHT);

        for (line_idx, line) in lines
            .iter()
            .take((u32::MAX as usize).saturating_add(1))
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

                    let tile = Tile::new(x as i32, y_idx, line_idx, y >= line_top_y);
                    self.tile_index_buf
                        .push(TileIndex::from_tile(self.tile_buf.len() as u32, &tile));
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
                            x_idx as i32,
                            y_idx,
                            line_idx,
                            y >= line_top_y && x_idx == winding_x,
                        );
                        self.tile_index_buf
                            .push(TileIndex::from_tile(self.tile_buf.len() as u32, &tile));
                        self.tile_buf.push(tile);
                    }
                }
            }
        }
    }
}

/// An index into a sorted tile buffer.
#[derive(Clone, Debug)]
struct TileIndex {
    x: u16,
    y: u16,
    index: u32,
}

impl TileIndex {
    pub(crate) fn from_tile(index: u32, tile: &Tile) -> Self {
        let x = (tile.x + 1).max(0) as u16;
        let y = tile.y;

        Self { x, y, index }
    }

    pub(crate) fn cmp(&self, b: &Self) -> std::cmp::Ordering {
        let xya = ((self.y as u32) << 16) + (self.x as u32);
        let xyb = ((b.y as u32) << 16) + (b.x as u32);
        xya.cmp(&xyb)
    }

    pub(crate) fn index(&self) -> usize {
        self.index as usize
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
