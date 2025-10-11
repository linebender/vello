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
pub const MAX_LINES_PER_PATH: u32 = 1 << 31;

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

    /// The index of the line this tile belongs to into the line buffer, plus whether the line
    /// crosses the top edge of the tile, packed together.
    ///
    /// The index is the unsigned number in the 31 least significant bits of this value.
    ///
    /// The last bit is 1 if and only if the lines crosses the tile's top edge. Lines making this
    /// crossing increment or decrement the coarse tile winding, depending on the line direction.
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
    ///
    /// `line_idx` must be smaller than [`MAX_LINES_PER_PATH`].
    #[inline]
    pub fn new(x: u16, y: u16, line_idx: u32, winding: bool) -> Self {
        Self::new_const(
            // Make sure that x and y stay in range when multiplying
            // with the tile width and height during strips generation.
            x.min(u16::MAX / Tile::WIDTH),
            y.min(u16::MAX / Tile::HEIGHT),
            line_idx,
            winding,
        )
    }

    #[inline]
    pub(crate) const fn new_const(x: u16, y: u16, line_idx: u32, winding: bool) -> Self {
        #[cfg(debug_assertions)]
        if line_idx >= MAX_LINES_PER_PATH {
            panic!("Max. number of lines per path exceeded.");
        }
        Self {
            x,
            y,
            packed_winding_line_idx: ((winding as u32) << 31) | line_idx,
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
        (self.packed_winding_line_idx & MAX_LINES_PER_PATH) != 0
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
            let (line_top_y, line_top_x, line_bottom_y, line_bottom_x) = if p0_y < p1_y {
                (p0_y, p0_x, p1_y, p1_x)
            } else {
                (p1_y, p1_x, p0_y, p0_x)
            };

            // For ease of logic, special-case purely vertical tiles.
            if line_left_x == line_right_x {
                let y_top_tiles = (line_top_y as u16).min(tile_rows);
                let y_bottom_tiles = (line_bottom_y.ceil() as u16).min(tile_rows);

                // Clamp all tiles that are strictly on the right of the viewport to the tile x coordinate
                // right next to the outside of the viewport. If we don't do this, we might end up
                // with too big tile coordinates, which will cause overflows in strip rendering.
                // TODO: in principle it is possible to cull right-of-viewport tiles, but it was causing some
                // issues, and we are choosing to do the less efficient but working thing for now.
                // See <https://github.com/linebender/vello/pull/1189> and
                // <https://github.com/linebender/vello/issues/1126>.
                let x = (line_left_x as u16).min(tile_columns + 1);

                for y_idx in y_top_tiles..y_bottom_tiles {
                    let y = f32::from(y_idx);

                    let tile = Tile::new(x, y_idx, line_idx, y >= line_top_y);
                    self.tile_buf.push(tile);
                }
            } else {
                let x_slope = (p1_x - p0_x) / (p1_y - p0_y);

                let y_top_tiles = (line_top_y as u16).min(tile_rows);
                let y_bottom_tiles = (line_bottom_y.ceil() as u16).min(tile_rows);

                for y_idx in y_top_tiles..y_bottom_tiles {
                    let y = f32::from(y_idx);

                    // The line's y-coordinates at the line's top- and bottom-most points within
                    // the tile row.
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
mod tests {
    use crate::flatten::{FlattenCtx, Line, Point, fill};
    use crate::kurbo::{Affine, BezPath};
    use crate::tile::{Tile, Tiles};
    use fearless_simd::Level;
    use std::vec;

    #[test]
    fn cull_line_at_top() {
        let line = Line {
            p0: Point { x: 3.0, y: -5.0 },
            p1: Point { x: 9.0, y: -1.0 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line], 100, 100);

        assert!(tiles.is_empty());
    }

    #[test]
    fn cull_line_at_right() {
        let line = Line {
            p0: Point { x: 101.0, y: 0.0 },
            p1: Point { x: 103.0, y: 20.0 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line], 100, 100);

        assert!(tiles.is_empty());
    }

    #[test]
    fn cull_line_at_bottom() {
        let line = Line {
            p0: Point { x: 30.0, y: 101.0 },
            p1: Point { x: 35.0, y: 105.0 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line], 100, 100);

        assert!(tiles.is_empty());
    }

    #[test]
    fn partially_cull_line_exceeding_viewport() {
        let line = Line {
            p0: Point { x: -2.0, y: -3.0 },
            p1: Point { x: 2.0, y: 1.0 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line], 100, 100);

        assert_eq!(tiles.tile_buf, [Tile::new(0, 0, 0, true)]);
    }

    #[test]
    fn horizontal_straight_line() {
        let line = Line {
            p0: Point { x: 1.5, y: 1.0 },
            p1: Point { x: 8.5, y: 1.0 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line], 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false),
                Tile::new(1, 0, 0, false),
                Tile::new(2, 0, 0, false),
            ]
        );
    }

    #[test]
    fn vertical_straight_line() {
        let line = Line {
            p0: Point { x: 1.0, y: 1.5 },
            p1: Point { x: 1.0, y: 8.5 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line], 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false),
                Tile::new(0, 1, 0, true),
                Tile::new(0, 2, 0, true),
            ]
        );
    }

    #[test]
    fn top_left_to_bottom_right() {
        let line = Line {
            p0: Point { x: 1.0, y: 1.0 },
            p1: Point { x: 11.0, y: 8.5 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line], 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false),
                Tile::new(1, 0, 0, false),
                Tile::new(1, 1, 0, true),
                Tile::new(2, 1, 0, false),
                Tile::new(2, 2, 0, true),
            ]
        );
    }

    #[test]
    fn bottom_right_to_top_left() {
        let line = Line {
            p0: Point { x: 11.0, y: 8.5 },
            p1: Point { x: 1.0, y: 1.0 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line], 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(0, 0, 0, false),
                Tile::new(1, 0, 0, false),
                Tile::new(1, 1, 0, true),
                Tile::new(2, 1, 0, false),
                Tile::new(2, 2, 0, true),
            ]
        );
    }

    #[test]
    fn bottom_left_to_top_right() {
        let line = Line {
            p0: Point { x: 2.0, y: 11.0 },
            p1: Point { x: 14.0, y: 6.0 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line], 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(2, 1, 0, false),
                Tile::new(3, 1, 0, false),
                Tile::new(0, 2, 0, false),
                Tile::new(1, 2, 0, false),
                Tile::new(2, 2, 0, true),
            ]
        );
    }

    #[test]
    fn top_right_to_bottom_left() {
        let line = Line {
            p0: Point { x: 14.0, y: 6.0 },
            p1: Point { x: 2.0, y: 11.0 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line], 100, 100);
        tiles.sort_tiles();

        assert_eq!(
            tiles.tile_buf,
            [
                Tile::new(2, 1, 0, false),
                Tile::new(3, 1, 0, false),
                Tile::new(0, 2, 0, false),
                Tile::new(1, 2, 0, false),
                Tile::new(2, 2, 0, true),
            ]
        );
    }

    #[test]
    fn two_lines_in_single_tile() {
        let line_1 = Line {
            p0: Point { x: 1.0, y: 3.0 },
            p1: Point { x: 3.0, y: 3.0 },
        };

        let line_2 = Line {
            p0: Point { x: 3.0, y: 3.0 },
            p1: Point { x: 0.0, y: 1.0 },
        };

        let mut tiles = Tiles::new(Level::try_detect().unwrap_or(Level::fallback()));
        tiles.make_tiles(&[line_1, line_2], 100, 100);

        assert_eq!(
            tiles.tile_buf,
            [Tile::new(0, 0, 0, false), Tile::new(0, 0, 1, false),]
        );
    }

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
        assert_eq!(tiles.tile_buf[0].x, 4);
        assert_eq!(tiles.tile_buf[1].x, 4);
    }
}
