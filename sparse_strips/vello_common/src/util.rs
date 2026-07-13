// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions.

use crate::geometry::RectU16;
use crate::kurbo::PathEl;
use crate::math::FloatExt;
use crate::tile::Tile;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};
use fearless_simd::{Bytes, Simd, SimdBase, SimdFloat, f32x16, u8x16, u8x32, u16x16, u16x32};
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;
use peniko::kurbo::{Affine, Rect};

/// Convert f32x16 to u8x16.
#[inline(always)]
pub fn f32_to_u8<S: Simd>(val: f32x16<S>) -> u8x16<S> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use fearless_simd::i32x16;
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    use fearless_simd::u32x16;

    let simd = val.simd;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let converted = val
        .max(f32x16::splat(simd, 0.0))
        .min(f32x16::splat(simd, 255.0))
        .to_int::<i32x16<S>>()
        .to_bytes();

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let converted = val
        .min(f32x16::splat(simd, 255.0))
        .to_int::<u32x16<S>>()
        .to_bytes();

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

#[cfg(test)]
mod tests {
    use super::RectExt;
    use super::RectU16;
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
}
