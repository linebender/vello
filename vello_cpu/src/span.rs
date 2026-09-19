// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Spans for delimiting horizontal regions.

use vello_common::tile::Tile;

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
    use crate::span::{Span, TileAlignedSpan};
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
