// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use crate::geometry::RectU16;
use crate::kurbo::PathEl;
use crate::math::FloatExt;
use crate::strip::Strip;
use crate::tile::Tile;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};
use fearless_simd::{
    Bytes, Simd, SimdBase, SimdFloat, f32x16, u8x16, u8x32, u16x16, u16x32, u32x16,
};
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;
use peniko::kurbo::{Affine, Rect};

/// Convert f32x16 to u8x16.
///
/// **Important note: The values need to be between 0.0 and 1.0, otherwise you might
/// get inconsistent results across different platforms.**
// We can't guarantee correctness for values < 0.0 due to a restriction in fearless_simd:
// https://github.com/linebender/fearless_simd/blob/3f4489389940b7c3c6ee1847a2d007a22494eeff/fearless_simd/src/generated/simd_types.rs#L1623
#[inline(always)]
pub fn f32_to_u8<S: Simd>(val: f32x16<S>) -> u8x16<S> {
    let simd = val.simd;
    let converted = val.to_int::<u32x16<S>>().to_bytes();

    let (x8_1, x8_2) = simd.split_u8x64(converted);
    let (p1, p2) = simd.split_u8x32(x8_1);
    let (p3, p4) = simd.split_u8x32(x8_2);

    let uzp1 = simd.unzip_low_u8x16(p1, p2);
    let uzp2 = simd.unzip_low_u8x16(p3, p4);
    simd.unzip_low_u8x16(uzp1, uzp2)
}

/// A trait for implementing a fast approximal division by 255 for integers.
pub trait Div255Ext {
    /// Divide by 255.
    fn div_255(self) -> Self;
}

impl<S: Simd> Div255Ext for u16x32<S> {
    #[inline(always)]
    fn div_255(self) -> Self {
        let p1 = Self::splat(self.simd, 255);
        let p2 = self + p1;
        p2 >> 8
    }
}

impl<S: Simd> Div255Ext for u16x16<S> {
    #[inline(always)]
    fn div_255(self) -> Self {
        let p1 = Self::splat(self.simd, 255);
        let p2 = self + p1;
        p2 >> 8
    }
}

/// Perform a normalized multiplication for u8x32.
#[inline(always)]
pub fn normalized_mul_u8x32<S: Simd>(a: u8x32<S>, b: u8x32<S>) -> u16x32<S> {
    (S::widen_u8x32(a.simd, a) * S::widen_u8x32(b.simd, b)).div_255()
}

/// Perform a normalized multiplication for u8x16.
#[inline(always)]
pub fn normalized_mul_u8x16<S: Simd>(a: u8x16<S>, b: u8x16<S>) -> u16x16<S> {
    (S::widen_u8x16(a.simd, a) * S::widen_u8x16(b.simd, b)).div_255()
}

/// Check if an affine transform is a pure integer translation.
///
/// Returns true if the transform only contains integer translation (no rotation,
/// skew, or scaling), meaning rectangles will remain pixel-aligned after transformation.
#[inline]
pub fn is_integer_translation(transform: &Affine) -> bool {
    let [a, b, c, d, e, f] = transform.as_coeffs();
    (a - 1.0).is_nearly_zero()
        && b.is_nearly_zero()
        && c.is_nearly_zero()
        && (d - 1.0).is_nearly_zero()
        && (e - e.round()).is_nearly_zero()
        && (f - f.round()).is_nearly_zero()
}
/// Check if an affine transform has no skewing (i.e. preserves axis alignment).
#[inline]
pub fn is_axis_aligned(transform: &Affine) -> bool {
    let [_, b, c, ..] = transform.as_coeffs();
    b.is_nearly_zero() && c.is_nearly_zero()
}

/// Extract scale factors from an affine transform using singular value decomposition.
///
/// Returns a tuple of (`scale_x`, `scale_y`) representing the scale along each axis.
/// This uses the same algorithm as kurbo's internal `svd()` method.
///
/// # Arguments
/// * `transform` - The affine transformation to extract scales from.
///
/// # Returns
/// A tuple `(scale_x, scale_y)` with minimum values clamped to 1e-6 to avoid division by zero.
///
/// # Note
/// TODO: Consider making `Affine::svd()` public in kurbo to avoid duplicating this code.
/// This implementation mirrors kurbo's internal SVD calculation for extracting scale factors
/// from arbitrary affine transformations.
#[inline]
pub fn extract_scales(transform: &Affine) -> (f32, f32) {
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let a = a as f32;
    let b = b as f32;
    let c = c as f32;
    let d = d as f32;

    // Compute singular values using the same formula as kurbo's svd()
    let a2 = a * a;
    let b2 = b * b;
    let c2 = c * c;
    let d2 = d * d;
    let s1 = a2 + b2 + c2 + d2;
    let s2 = ((a2 - b2 + c2 - d2).powi(2) + 4.0 * (a * b + c * d).powi(2)).sqrt();

    let scale_x = (0.5 * (s1 + s2)).sqrt();
    let scale_y = (0.5 * (s1 - s2)).sqrt();

    (scale_x.max(1e-6), scale_y.max(1e-6))
}

/// Extension methods for rectangles.
pub trait RectExt {
    /// Snap the rect to whole tile coordinates.
    fn snap_to_tile_coordinates(self) -> Self;
}

impl RectExt for Rect {
    #[inline]
    fn snap_to_tile_coordinates(self) -> Self {
        Self::new(
            snap_down(self.x0, Tile::WIDTH),
            snap_down(self.y0, Tile::HEIGHT),
            snap_up(self.x1, Tile::WIDTH),
            snap_up(self.y1, Tile::HEIGHT),
        )
    }
}

impl RectExt for RectU16 {
    #[inline]
    fn snap_to_tile_coordinates(self) -> Self {
        Self::new(
            (self.x0 / Tile::WIDTH) * Tile::WIDTH,
            (self.y0 / Tile::HEIGHT) * Tile::HEIGHT,
            self.x1
                .checked_next_multiple_of(Tile::WIDTH)
                .unwrap_or(u16::MAX),
            self.y1
                .checked_next_multiple_of(Tile::HEIGHT)
                .unwrap_or(u16::MAX),
        )
    }
}

#[inline]
fn snap_down(value: f64, step: u16) -> f64 {
    let step = f64::from(step);
    (value / step).floor() * step
}

#[inline]
fn snap_up(value: f64, step: u16) -> f64 {
    let step = f64::from(step);
    (value / step).ceil() * step
}

/// A type that can be cleared.
pub trait Clear {
    /// Clear the object to its default state.
    fn clear(&mut self);
}

impl<T> Clear for Vec<T> {
    fn clear(&mut self) {
        Self::clear(self);
    }
}

/// Pool for reusing allocations.
#[derive(Debug)]
pub struct Pool<T> {
    entries: Vec<T>,
    clear_on_submit: bool,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self::new(true)
    }
}

impl<T> Pool<T> {
    /// Create a new pool.
    ///
    /// `clear_on_submit` decides whether submitted values should
    /// be cleared when they are submitted or whether they should retain
    /// their original contents.
    pub fn new(clear_on_submit: bool) -> Self {
        Self {
            entries: Vec::new(),
            clear_on_submit,
        }
    }

    /// Take an object from the pool or create a new one.
    pub fn take(&mut self) -> T
    where
        T: Default,
    {
        self.entries.pop().unwrap_or_default()
    }

    /// Return an object to the pool.
    pub fn submit(&mut self, mut entry: T)
    where
        T: Clear,
    {
        if self.clear_on_submit {
            entry.clear();
        }

        self.entries.push(entry);
    }
}

/// Pool for reusing vector allocations.
pub type VecPool<T> = Pool<Vec<T>>;

/// A resizable vector that retains inner elements upon resizing.
#[derive(Debug)]
pub struct RetainVec<T> {
    inner: Vec<T>,
    len: usize,
}

impl<T: Clear> RetainVec<T> {
    /// Create an empty `RetainVec`.
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            len: 0,
        }
    }

    /// Create a `RetainVec` with `len` initialized entries.
    pub fn with_len(len: usize, mut init: impl FnMut() -> T) -> Self {
        let mut inner = Vec::with_capacity(len);
        inner.resize_with(len, &mut init);
        Self { inner, len }
    }

    /// Return the length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return `true` if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the entries as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.inner[..self.len]
    }

    /// Return the entries as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.inner[..self.len]
    }

    /// Iterate mutably over active entries.
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// Clear the elements in this vector.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Resize the vector.
    pub fn resize_with(&mut self, new_len: usize, mut init: impl FnMut() -> T) {
        let old_len = self.len;
        if new_len > self.inner.len() {
            self.inner.resize_with(new_len, &mut init);
        }
        self.len = new_len;

        // Make sure to actually reset the newly added values since they are not reset when shrinking
        // the vector.
        if new_len > old_len {
            for item in &mut self.inner[old_len..new_len] {
                item.clear();
            }
        }
    }
}

impl<T: Clear> Default for RetainVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Index<usize> for RetainVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[..self.len][index]
    }
}

impl<T> IndexMut<usize> for RetainVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[..self.len][index]
    }
}

/// Compute a conservative bounding box for the transformed path by computing the bounding box of
/// the transformed control points.
///
/// If `path` is empty, this returns an infinite, inversed [`Rect`] (`left` > `right` and `top` > `bottom`).
pub fn control_point_bbox(path: impl IntoIterator<Item = PathEl>, transform: Affine) -> Rect {
    // Start with an infinite, inversed rectangle. Adding the first point immediately collapses it
    // without branching.
    let mut bbox = Rect::new(
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for el in path {
        match el {
            PathEl::MoveTo(p) | PathEl::LineTo(p) => {
                bbox = bbox.union_pt(transform * p);
            }
            PathEl::QuadTo(p1, p2) => {
                bbox = bbox.union_pt(transform * p1);
                bbox = bbox.union_pt(transform * p2);
            }
            PathEl::CurveTo(p1, p2, p3) => {
                bbox = bbox.union_pt(transform * p1);
                bbox = bbox.union_pt(transform * p2);
                bbox = bbox.union_pt(transform * p3);
            }
            PathEl::ClosePath => {}
        }
    }
    bbox
}

/// Compute a conservative bounding box for the transformed path in pixel coordinates.
///
/// If `path` is empty, this returns an inverted [`RectU16`].
pub fn control_point_bbox_u16(
    path: impl IntoIterator<Item = PathEl>,
    transform: Affine,
) -> RectU16 {
    let bbox = control_point_bbox(path, transform);
    RectU16::new(
        bbox.x0 as u16,
        bbox.y0 as u16,
        bbox.x1.ceil() as u16,
        bbox.y1.ceil() as u16,
    )
}

pub(crate) fn strip_bbox(strips: &[Strip]) -> RectU16 {
    let mut bbox = RectU16::INVERTED;

    // Need at least one strip (and the sentinel one).
    if strips.len() < 2 {
        return bbox;
    }

    // TODO: It _feels_ like this "iterating over strips code" should be possible to
    // deduplicate with other locations (i.e. the code used to generate fill commands).

    for pair in strips.windows(2) {
        let strip = pair[0];
        let next_strip = pair[1];
        if strip.is_sentinel() {
            continue;
        }

        let strip_y = strip.strip_y();
        let row_y = strip_y.saturating_mul(Tile::HEIGHT);
        let row_y1 = row_y.saturating_add(Tile::HEIGHT);
        let strip_width = strip.width_to(&next_strip);
        let strip_x1 = strip.x.saturating_add(strip_width);

        if strip_width > 0 {
            bbox.union(RectU16::new(strip.x, row_y, strip_x1, row_y1));
        }

        if next_strip.fill_gap() && strip_y == next_strip.strip_y() {
            let fill_x1 = next_strip.x;
            if strip_x1 < fill_x1 {
                bbox.union(RectU16::new(strip_x1, row_y, fill_x1, row_y1));
            }
        }
    }

    bbox
}

#[cfg(test)]
mod tests {
    use super::{RectExt, RectU16, strip_bbox};
    use crate::strip::Strip;
    use crate::tile::Tile;
    use peniko::kurbo::Rect;

    #[test]
    fn snap_to_tile_coordinates_rounds_outward() {
        let rect = Rect::new(-4.1, -0.1, 4.1, 8.0).snap_to_tile_coordinates();
        assert_eq!(rect, Rect::new(-8.0, -4.0, 8.0, 8.0));
    }

    #[test]
    fn snap_u16_to_tile_coordinates_rounds_outward() {
        let rect = RectU16::new(5, 3, 9, 7).snap_to_tile_coordinates();
        assert_eq!(rect, RectU16::new(4, 0, 12, 8));
    }

    #[test]
    fn empty_strip_bbox() {
        let strips = [Strip::sentinel(0, 0), Strip::sentinel(0, 0)];

        assert_eq!(strip_bbox(&strips), RectU16::INVERTED);
    }

    #[test]
    fn single_strip_bbox() {
        let strips = [
            Strip::new(8, 4, 0, false),
            Strip::sentinel(4, u32::from(Tile::HEIGHT) * 4),
        ];

        assert_eq!(strip_bbox(&strips), RectU16::new(8, 4, 12, 8));
    }

    #[test]
    fn strip_with_fill_bbox() {
        let strips = [
            Strip::new(4, 0, 0, false),
            Strip::new(20, 0, u32::from(Tile::HEIGHT) * 4, true),
            Strip::sentinel(0, u32::from(Tile::HEIGHT) * 8),
        ];

        assert_eq!(strip_bbox(&strips), RectU16::new(4, 0, 24, 4));
    }

    #[test]
    fn strip_with_row_end_fill_gap_bbox_is_clamped_to_viewport() {
        let strips = [
            Strip::new(4, 0, 0, false),
            Strip::new(32, 0, u32::from(Tile::HEIGHT) * 4, true),
            Strip::sentinel(0, u32::from(Tile::HEIGHT) * 4),
        ];

        assert_eq!(strip_bbox(&strips), RectU16::new(4, 0, 32, 4));
    }

    #[test]
    fn strips_with_multiple_rows_bbox() {
        let strips = [
            Strip::new(12, 0, 0, false),
            Strip::new(4, 8, u32::from(Tile::HEIGHT) * 4, false),
            Strip::sentinel(8, u32::from(Tile::HEIGHT) * 8),
        ];

        assert_eq!(strip_bbox(&strips), RectU16::new(4, 0, 16, 12));
    }
}
