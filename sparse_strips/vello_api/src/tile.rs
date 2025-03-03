//! Types for generating tiles.

use crate::flatten::Point;

/// The width of a tile.
pub const TILE_WIDTH: u32 = 4;
/// The height of a tile.
pub const TILE_HEIGHT: u32 = 4;

/// A tile represents an aligned area on the pixmap, used to subdivide the viewport into sub-areas
/// (currently 4x4) and analyze line intersections inside each such area.
///
/// Keep in mind that it is possible to have multiple tiles with the same index,
/// namely if we have multiple lines crossing the same 4x4 area!
#[derive(Debug, Clone)]
pub struct Tile {
    /// The index of the tile in the x direction.
    pub x: i32,
    /// The index of the tile in the y direction.
    pub y: u16,
    /// The start point of the line in that tile.
    pub p0: Point,
    /// The end point of the line in that tile.
    pub p1: Point,
}

impl Tile {
    /// Create a new tile.
    pub fn new(x: i32, y: u16, p0: Point, p1: Point) -> Self {
        Self {
            // We don't need to store the exact negative location, just that it is negative,
            // so that the winding number calculation is correct.
            x: x.max(-1),
            y,
            p0,
            p1,
        }
    }

    /// Check whether two tiles are at the same location.
    pub fn same_loc(&self, other: &Self) -> bool {
        self.x == other.x && self.same_row(other)
    }

    /// Check whether two tiles are on the same strip.
    pub fn same_strip(&self, other: &Self) -> bool {
        self.same_row(other) && (other.x - self.x).abs() <= 1
    }

    /// Check whether two tiles are on the same row.
    pub fn same_row(&self, other: &Self) -> bool {
        self.y == other.y
    }

    /// Return the delta of the tile.
    pub fn delta(&self) -> i32 {
        (self.p1.y == 0.0) as i32 - (self.p0.y == 0.0) as i32
    }
}
