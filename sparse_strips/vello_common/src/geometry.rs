// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Geometry utilities.

use crate::kurbo::Rect;
use bytemuck::{Pod, Zeroable};
use core::num::TryFromIntError;
use core::ops::Add;

/// A size represented by two 16-bit unsigned integers.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct SizeU16(pub [u16; 2]);

impl SizeU16 {
    /// A zero size.
    pub const ZERO: Self = Self::new(0);

    /// Create a new square size.
    pub const fn new(size: u16) -> Self {
        Self([size; 2])
    }

    /// Create a new size from its width and height.
    pub const fn from_wh(width: u16, height: u16) -> Self {
        Self([width, height])
    }

    /// The width of this size.
    pub const fn width(self) -> u16 {
        self.0[0]
    }

    /// The height of this size.
    pub const fn height(self) -> u16 {
        self.0[1]
    }

    /// Return the maximum of the two sizes.
    pub fn max(self, other: Self) -> Self {
        Self::from_wh(
            self.width().max(other.width()),
            self.height().max(other.height()),
        )
    }

    /// Return the minimum of the two sizes.
    pub fn min(self, other: Self) -> Self {
        Self::from_wh(
            self.width().min(other.width()),
            self.height().min(other.height()),
        )
    }

    /// Clamp both dimensions to the given range.
    pub fn clamp(self, min: u16, max: u16) -> Self {
        Self::from_wh(self.width().clamp(min, max), self.height().clamp(min, max))
    }
}

impl From<[u16; 2]> for SizeU16 {
    fn from(value: [u16; 2]) -> Self {
        Self(value)
    }
}

impl Add for SizeU16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_wh(self.width() + rhs.width(), self.height() + rhs.height())
    }
}

impl Add<u16> for SizeU16 {
    type Output = Self;

    fn add(self, rhs: u16) -> Self::Output {
        self + Self::new(rhs)
    }
}

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

    /// Expand this rectangle by the given left, top, right, and bottom padding.
    #[inline(always)]
    pub const fn expand(self, padding: Self) -> Self {
        Self {
            x0: self.x0.saturating_sub(padding.x0),
            y0: self.y0.saturating_sub(padding.y0),
            x1: self.x1.saturating_add(padding.x1),
            y1: self.y1.saturating_add(padding.y1),
        }
    }

    /// Return this rectangle relative to `origin`, clamping negative coordinates to zero.
    #[inline(always)]
    pub fn relative_to_origin(self, origin: (u16, u16)) -> Self {
        self.shift((-(origin.0 as i32), -(origin.1 as i32)))
    }

    /// Return a shifted version of the rectangle, clamping negative coordinates to zero.
    #[inline]
    pub fn shift(self, shift: (i32, i32)) -> Self {
        Self {
            x0: (self.x0 as i32)
                .saturating_add(shift.0)
                .clamp(0, u16::MAX as i32) as u16,
            y0: (self.y0 as i32)
                .saturating_add(shift.1)
                .clamp(0, u16::MAX as i32) as u16,
            x1: (self.x1 as i32)
                .saturating_add(shift.0)
                .clamp(0, u16::MAX as i32) as u16,
            y1: (self.y1 as i32)
                .saturating_add(shift.1)
                .clamp(0, u16::MAX as i32) as u16,
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

    /// Return the rect as a [`kurbo::Rect`].
    pub fn as_rect(self) -> Rect {
        Rect::new(
            self.x0 as f64,
            self.y0 as f64,
            self.x1 as f64,
            self.y1 as f64,
        )
    }
}

impl From<RectU16> for SizeU16 {
    fn from(rect: RectU16) -> Self {
        Self::from_wh(rect.width(), rect.height())
    }
}

// TODO: Remove these types once we've completely moved to u16 everywhere in Vello Hybrid

/// An offset represented by two 32-bit unsigned integers.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct OffsetU32(pub [u32; 2]);

impl OffsetU32 {
    /// A zero offset.
    pub const ZERO: Self = Self::new(0);

    /// Create a new offset with equal x and y coordinates.
    pub const fn new(offset: u32) -> Self {
        Self([offset; 2])
    }

    /// Create a new offset from its x and y coordinates.
    pub const fn from_xy(x: u32, y: u32) -> Self {
        Self([x, y])
    }

    /// The x coordinate of this offset.
    pub const fn x(self) -> u32 {
        self.0[0]
    }

    /// The y coordinate of this offset.
    pub const fn y(self) -> u32 {
        self.0[1]
    }
}

impl From<[u32; 2]> for OffsetU32 {
    fn from(value: [u32; 2]) -> Self {
        Self(value)
    }
}

/// A size represented by two 32-bit unsigned integers.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct SizeU32(pub [u32; 2]);

impl SizeU32 {
    /// A zero size.
    pub const ZERO: Self = Self::new(0);

    /// Create a new square size.
    pub const fn new(size: u32) -> Self {
        Self([size; 2])
    }

    /// Create a new size from its width and height.
    pub const fn from_wh(width: u32, height: u32) -> Self {
        Self([width, height])
    }

    /// The width of this size.
    pub const fn width(self) -> u32 {
        self.0[0]
    }

    /// The height of this size.
    pub const fn height(self) -> u32 {
        self.0[1]
    }

    /// Return the maximum of the two sizes.
    pub fn max(self, other: Self) -> Self {
        Self::from_wh(
            self.width().max(other.width()),
            self.height().max(other.height()),
        )
    }

    /// Return the minimum of the two sizes.
    pub fn min(self, other: Self) -> Self {
        Self::from_wh(
            self.width().min(other.width()),
            self.height().min(other.height()),
        )
    }

    /// Clamp both dimensions to the given range.
    pub fn clamp(self, min: u32, max: u32) -> Self {
        Self::from_wh(self.width().clamp(min, max), self.height().clamp(min, max))
    }
}

impl From<[u32; 2]> for SizeU32 {
    fn from(value: [u32; 2]) -> Self {
        Self(value)
    }
}

impl From<(u32, u32)> for SizeU32 {
    fn from((width, height): (u32, u32)) -> Self {
        Self::from_wh(width, height)
    }
}

impl From<SizeU32> for (u32, u32) {
    fn from(size: SizeU32) -> Self {
        (size.width(), size.height())
    }
}

impl From<SizeU16> for SizeU32 {
    fn from(size: SizeU16) -> Self {
        Self::from_wh(u32::from(size.width()), u32::from(size.height()))
    }
}

impl TryFrom<SizeU32> for SizeU16 {
    type Error = TryFromIntError;

    fn try_from(size: SizeU32) -> Result<Self, Self::Error> {
        Ok(Self::from_wh(
            u16::try_from(size.width())?,
            u16::try_from(size.height())?,
        ))
    }
}

impl Add for SizeU32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::from_wh(self.width() + rhs.width(), self.height() + rhs.height())
    }
}

impl Add<u32> for SizeU32 {
    type Output = Self;

    fn add(self, rhs: u32) -> Self::Output {
        self + Self::new(rhs)
    }
}

/// An axis-aligned rectangle with `u32` coordinates.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct RectU32 {
    /// The minimum x coordinate.
    pub x0: u32,
    /// The minimum y coordinate.
    pub y0: u32,
    /// The exclusive maximum x coordinate.
    pub x1: u32,
    /// The exclusive maximum y coordinate.
    pub y1: u32,
}

impl RectU32 {
    /// Create a new rectangle from its corner coordinates.
    pub const fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// The width of this rectangle.
    pub const fn width(self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }

    /// The height of this rectangle.
    pub const fn height(self) -> u32 {
        self.y1.saturating_sub(self.y0)
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

#[cfg(test)]
mod tests {
    use super::{SizeU16, SizeU32};

    #[test]
    fn size_addition() {
        assert_eq!(
            SizeU16::from_wh(2, 3) + SizeU16::from_wh(5, 7),
            SizeU16::from_wh(7, 10)
        );
        assert_eq!(SizeU16::from_wh(2, 3) + 5, SizeU16::from_wh(7, 8));
        assert_eq!(
            SizeU32::from_wh(11, 13) + SizeU32::from_wh(17, 19),
            SizeU32::from_wh(28, 32)
        );
        assert_eq!(SizeU32::from_wh(11, 13) + 17, SizeU32::from_wh(28, 30));
    }

    #[test]
    fn size_minimum() {
        assert_eq!(
            SizeU16::from_wh(2, 7).min(SizeU16::from_wh(5, 3)),
            SizeU16::from_wh(2, 3)
        );
        assert_eq!(
            SizeU32::from_wh(11, 19).min(SizeU32::from_wh(17, 13)),
            SizeU32::from_wh(11, 13)
        );
    }

    #[test]
    fn size_u32_to_u16_conversion() {
        assert_eq!(
            SizeU16::try_from(SizeU32::from_wh(11, 13)),
            Ok(SizeU16::from_wh(11, 13))
        );
        assert!(SizeU16::try_from(SizeU32::from_wh(u32::from(u16::MAX) + 1, 13)).is_err());
    }
}
