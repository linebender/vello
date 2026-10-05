// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared types for the sparse strips port: lines, tiles and the winding histogram.
//!
//! These mirror Skia's `SparseStripsTypes.h`, `Tiler.h` (`struct Tile`) and
//! `WindingHistogram.h`. The bit layout of [`Tile`] is identical to Skia's.

/// A point in device pixel space.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    #[inline(always)]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// A flattened line segment in device pixel space.
///
/// Unlike Skia's `Polyline` (a NaN-separated point list), a path is a plain slice of lines.
/// [`Tile::line_idx`] indexes into that slice.
///
/// Invariants established by the front end:
/// - both points are finite,
/// - `p0 != p1` (zero-length lines are dropped),
/// - the lines of a path form closed loops (every point is the end of exactly as many
///   lines as it is the start of), with bit-identical shared endpoints.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Line {
    pub p0: Point,
    pub p1: Point,
}

impl Line {
    #[inline(always)]
    pub const fn new(p0: Point, p1: Point) -> Self {
        Self { p0, p1 }
    }
}

/// Integer rectangle in pixels, `[left, right) × [top, bottom)`. Port of Skia's `SkIRect`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl IRect {
    #[inline(always)]
    pub const fn from_wh(width: i32, height: i32) -> Self {
        Self {
            left: 0,
            top: 0,
            right: width,
            bottom: height,
        }
    }

    #[inline(always)]
    pub const fn width(&self) -> i32 {
        self.right - self.left
    }

    #[inline(always)]
    pub const fn height(&self) -> i32 {
        self.bottom - self.top
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.left >= self.right || self.top >= self.bottom
    }
}

/// A tile touched by one line, as produced by the tiler.
///
/// Port of Skia's `struct Tile`. Layout of `packed` (MSB to LSB):
///
/// ```text
/// 31------------------------------------------------------5|4|3|2|1|0|
/// |            Parent Line Index (27 bits)                 |W|R|L|B|T|
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tile {
    /// Tile column.
    pub x: u16,
    /// Tile row.
    pub y: u16,
    /// Parent line index and intersection/winding mask.
    pub packed: u32,
}

impl Tile {
    /// Top edge intersection.
    pub const T: u32 = 0b00001;
    /// Bottom edge intersection.
    pub const B: u32 = 0b00010;
    /// Left edge intersection.
    pub const L: u32 = 0b00100;
    /// Right edge intersection.
    pub const R: u32 = 0b01000;
    /// Coarse winding: the line touched the tile's top edge.
    pub const W: u32 = 0b10000;

    pub const BOT_SHIFT: u32 = 1;
    pub const LEFT_SHIFT: u32 = 2;
    pub const RIGHT_SHIFT: u32 = 3;
    pub const WINDING_SHIFT: u32 = 4;
    pub const INT_MASK_SHIFT: u32 = 5;

    pub const INTERSECTION_MASK: u32 = Self::W | Self::R | Self::L | Self::B | Self::T;
    pub const MAX_LINES_PER_PATH: u32 = 1 << (32 - Self::INT_MASK_SHIFT);

    #[inline(always)]
    pub fn new(x: u16, y: u16, line_idx: u32, intersection_mask: u32) -> Self {
        debug_assert!(intersection_mask < (1 << Self::INT_MASK_SHIFT));
        debug_assert!(line_idx < Self::MAX_LINES_PER_PATH);
        Self {
            x,
            y,
            packed: (line_idx << Self::INT_MASK_SHIFT) | intersection_mask,
        }
    }

    /// Sort key: rows first, then columns, then line index (for cache locality).
    #[inline(always)]
    pub fn to_bits(self) -> u64 {
        ((self.y as u64) << 48) | ((self.x as u64) << 32) | (self.packed as u64)
    }

    #[inline(always)]
    pub fn line_idx(self) -> u32 {
        self.packed >> Self::INT_MASK_SHIFT
    }

    #[inline(always)]
    pub fn intersection_mask(self) -> u32 {
        self.packed & Self::INTERSECTION_MASK
    }

    #[inline(always)]
    pub fn coarse_winding(self) -> bool {
        (self.packed & Self::W) != 0
    }

    #[inline(always)]
    pub fn has_left_intersection(self) -> bool {
        (self.packed & Self::L) != 0
    }
}

/// Per tile row winding contributed by lines culled to the left of the clip.
///
/// Port of Skia's `WindingHistogram`.
#[derive(Clone, Debug, Default)]
pub struct WindingHistogram {
    data: Vec<i16>,
}

impl WindingHistogram {
    pub fn new(rows: usize) -> Self {
        Self {
            data: vec![0; rows],
        }
    }

    pub fn resize(&mut self, rows: usize) {
        self.data.resize(rows, 0);
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline(always)]
    pub fn get(&self, row: usize) -> i16 {
        self.data[row]
    }

    #[inline(always)]
    pub fn add_winding(&mut self, row: usize, delta: i16) {
        self.data[row] = self.data[row].wrapping_add(delta);
    }

    #[inline(always)]
    pub fn add_winding_range(&mut self, start_row: usize, end_row: usize, delta: i16) {
        for w in &mut self.data[start_row..end_row] {
            *w = w.wrapping_add(delta);
        }
    }

    pub fn data(&self) -> &[i16] {
        &self.data
    }
}
