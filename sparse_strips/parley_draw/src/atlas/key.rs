// Copyright 2026 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Cache key for glyph bitmaps stored in the atlas.
//!
//! [`GlyphCacheKey`] captures every parameter that affects the visual appearance
//! of a rasterized glyph — font identity, size, hinting, subpixel position,
//! COLR context color, and variable-font coordinates. Two keys that compare
//! equal produce identical bitmaps and can safely share a single atlas entry.

use core::hash::{Hash, Hasher};
#[cfg(not(feature = "std"))]
use core_maths::CoreFloat as _;
use skrifa::instance::NormalizedCoord;
use smallvec::SmallVec;
use vello_common::color::{AlphaColor, Srgb};

/// Number of horizontal subpixel quantization buckets (valid range: 1–255).
///
/// Higher values improve rendering quality at the cost of more atlas entries
/// per glyph. Common values: 1 (disabled), 2, 4 (default), 8.
pub(crate) const SUBPIXEL_BUCKETS: u8 = 4;

/// Unique identifier for a cached glyph bitmap.
///
/// Two glyphs with the same key are visually identical and can share
/// the same cached bitmap. The key includes all parameters that affect
/// the glyph's appearance.
///
/// `var_coords` is deliberately excluded from `Hash`/`Eq` because the
/// [`GlyphAtlas`](crate::atlas::cache::GlyphAtlas) uses a two-level map
/// structure that already partitions entries by variation coordinates.
/// Callers that use a flat map must ensure equivalent `var_coords`
/// externally.
#[derive(Clone, Debug)]
pub struct GlyphCacheKey {
    /// Unique identifier for the font blob.
    pub font_id: u64,
    /// Index within font collection (for TTC files).
    pub font_index: u32,
    /// Glyph index within the font.
    pub glyph_id: u32,
    /// Font size as f32 bits (exact match, no quantization).
    pub size_bits: u32,
    /// Whether hinting was applied.
    pub hinted: bool,
    /// Horizontal subpixel position (0 to SUBPIXEL_BUCKETS-1).
    pub subpixel_x: u8,
    /// Context color for COLR glyphs. Only used for rendering, not for Hash/Eq.
    pub context_color: AlphaColor<Srgb>,
    /// Pre-packed context color (premultiplied RGBA8 as u32) used in Hash/Eq.
    pub context_color_packed: u32,
    /// Variation coordinates for variable fonts.
    pub var_coords: SmallVec<[NormalizedCoord; 4]>,
}

impl GlyphCacheKey {
    /// Creates a new cache key.
    ///
    /// `fractional_x` (the fractional pixel offset) is quantized into
    /// `SUBPIXEL_BUCKETS` buckets, so nearby positions share the same entry.
    #[inline]
    pub fn new(
        font_id: u64,
        font_index: u32,
        glyph_id: u32,
        size: f32,
        hinted: bool,
        fractional_x: f32,
        context_color: AlphaColor<Srgb>,
        context_color_packed: u32,
        var_coords: &[NormalizedCoord],
    ) -> Self {
        Self {
            font_id,
            font_index,
            glyph_id,
            size_bits: size.to_bits(),
            hinted,
            subpixel_x: quantize_subpixel(fractional_x),
            context_color,
            context_color_packed,
            var_coords: SmallVec::from_slice(var_coords),
        }
    }
}

/// Manual `Hash` and `PartialEq` use the pre-packed `context_color_packed` field
/// (a premultiplied RGBA8 `u32`) instead of `AlphaColor<Srgb>`, which doesn't
/// implement `Hash`/`Eq`. Packing once at construction avoids repeated work
/// during lookups. `glyph_id` is compared first for early short-circuit.
impl Hash for GlyphCacheKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.font_id.hash(state);
        self.font_index.hash(state);
        self.glyph_id.hash(state);
        self.size_bits.hash(state);
        self.hinted.hash(state);
        self.subpixel_x.hash(state);
        self.context_color_packed.hash(state);
    }
}

impl PartialEq for GlyphCacheKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.glyph_id == other.glyph_id
            && self.subpixel_x == other.subpixel_x
            && self.font_id == other.font_id
            && self.font_index == other.font_index
            && self.size_bits == other.size_bits
            && self.hinted == other.hinted
            && self.context_color_packed == other.context_color_packed
    }
}

impl Eq for GlyphCacheKey {}

/// Premultiply and pack an RGBA color into a `u32` for bitwise hashing/comparison.
#[inline]
pub(crate) fn pack_color(color: AlphaColor<Srgb>) -> u32 {
    color.premultiply().to_rgba8().to_u32()
}

/// Quantize a fractional pixel offset into one of [`SUBPIXEL_BUCKETS`] buckets.
///
/// Values near 1.0 (>= 0.875 with 4 buckets) are clamped to the last bucket
/// rather than wrapping to 0. Wrapping to bucket 0 without also incrementing the
/// integer pixel coordinate would shift the glyph by ~0.75px in the wrong
/// direction. Clamping keeps the worst-case error to 0.125px.
#[expect(
    clippy::cast_possible_truncation,
    reason = "result is clamped to SUBPIXEL_BUCKETS-1 which fits in u8"
)]
#[inline]
fn quantize_subpixel(frac: f32) -> u8 {
    let normalized = frac.fract();
    let normalized = if normalized < 0.0 {
        normalized + 1.0
    } else {
        normalized
    };
    ((normalized * SUBPIXEL_BUCKETS as f32).round() as u8).min(SUBPIXEL_BUCKETS - 1)
}

/// Convert a quantized bucket index back to the fractional pixel offset it represents.
#[inline]
pub fn subpixel_offset(quantized: u8) -> f32 {
    quantized as f32 / SUBPIXEL_BUCKETS as f32
}

#[cfg(test)]
mod tests {
    use vello_common::color::palette::css::BLACK;

    use super::*;

    #[test]
    fn test_quantize_subpixel() {
        // Test bucket boundaries
        assert_eq!(quantize_subpixel(0.0), 0);
        assert_eq!(quantize_subpixel(0.1), 0);
        assert_eq!(quantize_subpixel(0.2), 1);
        assert_eq!(quantize_subpixel(0.25), 1);
        assert_eq!(quantize_subpixel(0.4), 2);
        assert_eq!(quantize_subpixel(0.5), 2);
        assert_eq!(quantize_subpixel(0.6), 2);
        assert_eq!(quantize_subpixel(0.7), 3);
        assert_eq!(quantize_subpixel(0.75), 3);
        assert_eq!(quantize_subpixel(0.9), 3);
        assert_eq!(quantize_subpixel(1.0), 0);
    }

    #[test]
    fn test_subpixel_offset() {
        assert_eq!(subpixel_offset(0), 0.0);
        assert_eq!(subpixel_offset(1), 0.25);
        assert_eq!(subpixel_offset(2), 0.5);
        assert_eq!(subpixel_offset(3), 0.75);
    }

    #[test]
    fn test_key_equality() {
        let packed = pack_color(BLACK);
        let key1 = GlyphCacheKey::new(1, 0, 42, 16.0, true, 0.3, BLACK, packed, &[]);
        let key2 = GlyphCacheKey::new(1, 0, 42, 16.0, true, 0.3, BLACK, packed, &[]);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_var_coords_excluded_from_equality() {
        let packed = pack_color(BLACK);
        // var_coords is excluded from Hash/Eq (two-level map handles it),
        // so keys differing only in var_coords are considered equal.
        let key1 = GlyphCacheKey::new(1, 0, 42, 16.0, true, 0.3, BLACK, packed, &[]);
        let key2 = GlyphCacheKey::new(
            1,
            0,
            42,
            16.0,
            true,
            0.3,
            BLACK,
            packed,
            &[NormalizedCoord::from_bits(100)],
        );
        assert_eq!(key1, key2);
    }
}
