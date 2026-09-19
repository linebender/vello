// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko::ImageQuality;
use vello_common::encode::EncodedImage;
use vello_common::fearless_simd::{f32x4, prelude::*, u8x32};
use vello_common::math::FloatExt;
use vello_common::tile::Tile;
use vello_common::util::{narrow, normalized_mul_u8};

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

impl<S: Simd> NormalizedMulExt for u8x32<S> {
    #[inline(always)]
    fn normalized_mul(self, other: Self) -> Self {
        narrow(normalized_mul_u8(self, other))
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct Span {
    /// The horizontal start position in pixels.
    x: u16,
    /// The horizontal span width in pixels.
    width: u16,
}

impl Span {
    /// Creates a span from pixel coordinates.
    pub fn new(x: u16, width: u16) -> Self {
        Self { x, width }
    }

    /// Expands this span to the tiles covering it. Empty spans remain empty.
    ///
    /// Panics if the aligned end cannot be represented in pixel coordinates.
    pub fn tile_aligned(self) -> TileAlignedSpan {
        let start = self.tile_x();
        let count = if self.width == 0 {
            0
        } else {
            self.tile_end() - start
        };

        TileAlignedSpan::from_tiles(start, count)
    }

    /// Returns the horizontal start position in tile coordinates.
    pub fn tile_x(self) -> u16 {
        self.x / Tile::WIDTH
    }

    /// Returns the exclusive horizontal end position in tile coordinates.
    pub fn tile_end(self) -> u16 {
        self.pixel_end().div_ceil(Tile::WIDTH)
    }

    /// Extends this span to include another span.
    pub fn extend(&mut self, other: Self) {
        let x = self.x.min(other.x);
        let end = self.pixel_end().max(other.pixel_end());
        *self = Self::new(x, end.saturating_sub(x));
    }

    /// Returns the intersection of this span with another span.
    pub fn intersect(self, other: Self) -> Option<Self> {
        let x = self.x.max(other.x);
        let end = self.pixel_end().min(other.pixel_end());
        (x < end).then(|| Self::new(x, end - x))
    }

    /// Returns the horizontal start position in pixels.
    pub fn pixel_x(self) -> u16 {
        self.x
    }

    /// Returns the horizontal span width in pixels.
    pub fn pixel_width(self) -> u16 {
        self.width
    }

    /// Returns the exclusive horizontal end position in pixels.
    pub fn pixel_end(self) -> u16 {
        self.pixel_x().saturating_add(self.pixel_width())
    }
}

/// A horizontal pixel span aligned to tile coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct TileAlignedSpan(Span);

impl TileAlignedSpan {
    /// Creates a tile-aligned span from a tile start and tile count.
    pub fn from_tiles(start: u16, count: u16) -> Self {
        let x = start.checked_mul(Tile::WIDTH).unwrap();
        let width = count.checked_mul(Tile::WIDTH).unwrap();
        x.checked_add(width).unwrap();

        Self(Span::new(x, width))
    }

    /// Returns the underlying pixel span.
    pub fn as_span(self) -> Span {
        self.0
    }

    /// Returns the horizontal start position in pixels.
    pub fn pixel_x(self) -> u16 {
        self.0.pixel_x()
    }

    /// Returns the horizontal width in pixels.
    pub fn pixel_width(self) -> u16 {
        self.0.pixel_width()
    }

    /// Returns the exclusive horizontal end in pixels.
    pub fn pixel_end(self) -> u16 {
        self.0.pixel_end()
    }

    /// Returns the horizontal start in tile coordinates.
    pub fn tile_x(self) -> u16 {
        self.pixel_x() / Tile::WIDTH
    }

    /// Returns the exclusive horizontal end in tile coordinates.
    pub fn tile_end(self) -> u16 {
        self.pixel_end() / Tile::WIDTH
    }

    /// Extends this span to cover another tile-aligned span.
    pub fn extend(&mut self, other: Self) {
        self.0.extend(other.0);
    }

    /// Intersects two tile-aligned spans.
    pub fn intersect(self, other: Self) -> Option<Self> {
        self.0.intersect(other.0).map(Self)
    }
}

impl TryFrom<Span> for TileAlignedSpan {
    type Error = &'static str;

    fn try_from(span: Span) -> Result<Self, Self::Error> {
        if !span.x.is_multiple_of(Tile::WIDTH) || !span.width.is_multiple_of(Tile::WIDTH) {
            return Err("span is not tile-aligned");
        }

        if span.x.checked_add(span.width).is_none() {
            return Err("tile span end overflow");
        }

        Ok(Self::from_tiles(
            span.x / Tile::WIDTH,
            span.width / Tile::WIDTH,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{Span, TileAlignedSpan};
    use vello_common::tile::Tile;

    #[test]
    fn tile_aligned_rounds_outward() {
        let tile = Tile::WIDTH;
        for (span, expected) in [
            (Span::new(tile, tile), Span::new(tile, tile)),
            (Span::new(tile + 1, tile - 1), Span::new(tile, tile)),
            (Span::new(tile, tile + 1), Span::new(tile, 2 * tile)),
            (Span::new(tile + 1, tile), Span::new(tile, 2 * tile)),
        ] {
            assert_eq!(span.tile_aligned().as_span(), expected, "{span:?}");
        }
    }

    #[test]
    fn tile_aligned_keeps_empty_spans_empty() {
        for x in [0, Tile::WIDTH, Tile::WIDTH + 1] {
            let aligned = Span::new(x, 0).tile_aligned();
            assert_eq!(aligned.pixel_width(), 0, "x = {x}");
        }
    }

    #[test]
    fn try_from_preserves_aligned_bounds() {
        let max_aligned = u16::MAX / Tile::WIDTH * Tile::WIDTH;
        for span in [
            Span::new(Tile::WIDTH, 3 * Tile::WIDTH),
            Span::new(0, max_aligned),
            Span::new(max_aligned, 0),
        ] {
            assert_eq!(TileAlignedSpan::try_from(span).unwrap().as_span(), span);
        }
    }

    #[test]
    fn try_from_rejects_unaligned_bounds() {
        for span in [
            Span::new(Tile::WIDTH + 1, Tile::WIDTH),
            Span::new(Tile::WIDTH, Tile::WIDTH + 1),
            Span::new(Tile::WIDTH + 1, 0),
        ] {
            assert!(TileAlignedSpan::try_from(span).is_err(), "{span:?}");
        }
    }
}
