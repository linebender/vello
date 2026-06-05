// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko::ImageQuality;
use vello_common::encode::EncodedImage;
use vello_common::fearless_simd::{Simd, SimdBase, f32x4, u8x32};
use vello_common::geometry::RectU16;
use vello_common::math::FloatExt;
use vello_common::tile::Tile;
use vello_common::util::Div255Ext;

pub(crate) mod scalar {
    /// Perform an approximate division by 255.
    ///
    /// There are three reasons for having this method.
    /// 1) Divisions are slower than shifting + adding, and the compiler does not seem to replace
    ///    divisions by 255 with an equivalent (this was verified by benchmarking; doing / 255 was
    ///    significantly slower).
    /// 2) Integer divisions are usually not available in SIMD, so this provides a good baseline
    ///    implementation.
    /// 3) There are two options for performing the division: One is to perform the division
    ///    in a way that completely preserves the rounding semantics of a integer division by
    ///    255. This could be achieved using the implementation `(val + 1 + (val >> 8)) >> 8`.
    ///    The second approach (used here) has slightly different rounding behavior to a
    ///    normal division by 255, but is much faster (see <https://github.com/linebender/vello/issues/904>)
    ///    and therefore preferable for the high-performance pipeline.
    ///
    /// Four properties worth mentioning:
    /// - This actually calculates the ceiling of `val / 256`.
    /// - Within the allowed range for `val`, rounding errors do not appear for values divisible by 255, i.e. any call `div_255(val * 255)` will always yield `val`.
    /// - If there is a discrepancy, this division will always yield a value 1 higher than the original.
    /// - This holds for values of `val` up to and including `65279`. You should not call this function with higher values.
    #[inline(always)]
    pub(crate) const fn div_255(val: u16) -> u16 {
        debug_assert!(
            val < 65280,
            "the properties of `div_255` do not hold for values of `65280` or greater"
        );
        (val + 255) >> 8
    }

    #[cfg(test)]
    mod tests {
        use crate::util::scalar::div_255;

        #[test]
        fn div_255_properties() {
            for i in 0_u16..256 * 255 {
                let expected = i / 255;
                let actual = div_255(i);

                assert!(
                    expected <= actual,
                    "In case of a discrepancy, the division should yield a value higher than the original."
                );

                let diff = expected.abs_diff(actual);
                assert!(diff <= 1, "Rounding error shouldn't be higher than 1.");

                if i % 255 == 0 {
                    assert_eq!(diff, 0, "Division should be accurate for multiples of 255.");
                }
            }
        }
    }
}

pub(crate) trait NormalizedMulExt {
    fn normalized_mul(self, other: Self) -> Self;
}

pub(crate) fn bbox_relative_to(bbox: RectU16, origin: (u16, u16)) -> RectU16 {
    RectU16::new(
        bbox.x0.saturating_sub(origin.0),
        bbox.y0.saturating_sub(origin.1),
        bbox.x1.saturating_sub(origin.0),
        bbox.y1.saturating_sub(origin.1),
    )
}

pub(crate) fn snap_bbox_to_tile_coordinates(bbox: RectU16) -> RectU16 {
    RectU16::new(
        (bbox.x0 / Tile::WIDTH) * Tile::WIDTH,
        (bbox.y0 / Tile::HEIGHT) * Tile::HEIGHT,
        bbox.x1
            .checked_next_multiple_of(Tile::WIDTH)
            .unwrap_or(u16::MAX),
        bbox.y1
            .checked_next_multiple_of(Tile::HEIGHT)
            .unwrap_or(u16::MAX),
    )
}

impl<S: Simd> NormalizedMulExt for u8x32<S> {
    #[inline(always)]
    fn normalized_mul(self, other: Self) -> Self {
        let divided = (self.simd.widen_u8x32(self) * other.simd.widen_u8x32(other)).div_255();
        self.simd.narrow_u16x32(divided)
    }
}

pub(crate) trait EncodedImageExt {
    fn has_skew(&self) -> bool;
    fn nearest_neighbor(&self) -> bool;
}

impl EncodedImageExt for EncodedImage {
    fn has_skew(&self) -> bool {
        !(self.x_advance.y as f32).is_nearly_zero() || !(self.y_advance.x as f32).is_nearly_zero()
    }

    fn nearest_neighbor(&self) -> bool {
        self.sampler.quality == ImageQuality::Low
    }
}

pub(crate) trait Premultiply {
    fn premultiply(self, alphas: Self) -> Self;
    fn unpremultiply(self, alphas: Self) -> Self;
}

impl<S: Simd> Premultiply for f32x4<S> {
    #[inline(always)]
    fn premultiply(self, alphas: Self) -> Self {
        self * alphas
    }

    #[inline(always)]
    fn unpremultiply(self, alphas: Self) -> Self {
        let zero = Self::splat(alphas.simd, 0.0);
        let divided = self / alphas;

        self.simd
            .select_f32x4(self.simd.simd_eq_f32x4(alphas, zero), zero, divided)
    }
}

/// A horizontal span in pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Span {
    /// The horizontal start position in pixels.
    x: u16,
    /// The horizontal span width in pixels.
    width: u16,
}

impl Span {
    /// Creates a span from pixel coordinates.
    pub(crate) fn new(x: u16, width: u16) -> Self {
        Self { x, width }
    }

    /// Creates a span from tile coordinates.
    pub(crate) fn new_tile(tile_x: u16, tile_width: u16) -> Self {
        Self {
            x: tile_x * Tile::WIDTH,
            width: tile_width * Tile::WIDTH,
        }
    }

    /// Returns the horizontal start position in tile coordinates.
    pub(crate) fn tile_x(self) -> u16 {
        self.x / Tile::WIDTH
    }

    /// Returns the exclusive horizontal end position in tile coordinates.
    pub(crate) fn tile_end(self) -> u16 {
        self.pixel_end().div_ceil(Tile::WIDTH)
    }

    /// Extends this span to include another span.
    pub(crate) fn extend(&mut self, other: Self) {
        let x = self.x.min(other.x);
        let end = self.pixel_end().max(other.pixel_end());
        *self = Self::new(x, end.saturating_sub(x));
    }

    /// Returns the intersection of this span with another span.
    pub(crate) fn intersect(self, other: Self) -> Option<Self> {
        let x = self.x.max(other.x);
        let end = self.pixel_end().min(other.pixel_end());
        (x < end).then(|| Self::new(x, end - x))
    }

    /// Returns the horizontal start position in pixels.
    pub(crate) fn pixel_x(self) -> u16 {
        self.x
    }

    /// Returns the horizontal span width in pixels.
    pub(crate) fn pixel_width(self) -> u16 {
        self.width
    }

    /// Returns the exclusive horizontal end position in pixels.
    pub(crate) fn pixel_end(self) -> u16 {
        self.pixel_x().saturating_add(self.pixel_width())
    }
}
