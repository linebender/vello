// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Geometry utilities.

/// An axis-aligned rectangle with `u16` coordinates, stored as two corners `(x0, y0)` and
/// `(x1, y1)`.
///
/// `(x0, y0)` is the top-left (minimum) corner and `(x1, y1)` is the bottom-right (maximum) corner.
/// The rectangle is considered to be empty when `x0 >= x1` or `y0 >= y1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RectU16 {
    /// The minimum x coordinate (left edge).
    pub x0: u16,
    /// The minimum y coordinate (top edge).
    pub y0: u16,
    /// The maximum x coordinate (right edge, exclusive).
    pub x1: u16,
    /// The maximum y coordinate (bottom edge, exclusive).
    pub y1: u16,
}

impl RectU16 {
    /// A rectangle with all coordinates set to zero.
    pub const ZERO: Self = Self {
        x0: 0,
        y0: 0,
        x1: 0,
        y1: 0,
    };

    /// An empty, maximally inverted rectangle, useful as a starting value for incremental union
    /// operations.
    ///
    /// Has `(x0, y0) = (u16::MAX, u16::MAX)` and `(x1, y1) = (0, 0)`.
    pub const INVERTED: Self = Self {
        x0: u16::MAX,
        y0: u16::MAX,
        x1: 0,
        y1: 0,
    };

    /// Create a new rectangle from its corner coordinates.
    #[inline(always)]
    pub const fn new(x0: u16, y0: u16, x1: u16, y1: u16) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// The width of the rectangle (`x1 - x0`), saturating at zero.
    #[inline(always)]
    pub const fn width(self) -> u16 {
        self.x1.saturating_sub(self.x0)
    }

    /// The height of the rectangle (`y1 - y0`), saturating at zero.
    #[inline(always)]
    pub const fn height(self) -> u16 {
        self.y1.saturating_sub(self.y0)
    }

    /// Returns `true` if the rectangle has zero area (`x0 >= x1` or `y0 >= y1`).
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    /// Check if a point `(x, y)` is contained within this rectangle.
    ///
    /// Returns `true` if `x0 <= x < x1` and `y0 <= y < y1`.
    #[inline(always)]
    pub const fn contains(self, x: u16, y: u16) -> bool {
        (x >= self.x0) & (x < self.x1) & (y >= self.y0) & (y < self.y1)
    }

    /// Compute the intersection of two rectangles.
    ///
    /// The result may be empty if the rectangles do not overlap.
    #[inline(always)]
    pub const fn intersect(self, other: Self) -> Self {
        Self {
            x0: const_max(self.x0, other.x0),
            y0: const_max(self.y0, other.y0),
            x1: const_min(self.x1, other.x1),
            y1: const_min(self.y1, other.y1),
        }
    }

    /// Expand this rectangle to also cover `other` (union in place).
    ///
    /// The union of `self` with a [`Self::INVERTED`] returns `self`.
    #[inline(always)]
    pub const fn union(&mut self, other: Self) {
        self.x0 = const_min(self.x0, other.x0);
        self.y0 = const_min(self.y0, other.y0);
        self.x1 = const_max(self.x1, other.x1);
        self.y1 = const_max(self.y1, other.y1);
    }
}

#[inline(always)]
const fn const_max(a: u16, b: u16) -> u16 {
    if a > b { a } else { b }
}

#[inline(always)]
const fn const_min(a: u16, b: u16) -> u16 {
    if a < b { a } else { b }
}
