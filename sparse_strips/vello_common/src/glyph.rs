// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared glyph data types.

use core::fmt::{Debug, Formatter};
use skrifa::instance::LocationRef;
use skrifa::outline::OutlinePen;
use skrifa::FontRef;

use crate::kurbo::{Affine, BezPath, Rect};

/// Positioned glyph.
#[derive(Copy, Clone, Default, Debug)]
pub struct Glyph {
    /// The font-specific identifier for this glyph.
    ///
    /// This ID is specific to the font being used and corresponds to the
    /// glyph index within that font. It is *not* a Unicode code point.
    pub id: u32,
    /// X-offset in run, relative to transform.
    pub x: f32,
    /// Y-offset in run, relative to transform.
    pub y: f32,
}

/// A glyph defined by a COLR glyph description.
///
/// Clients are supposed to first draw the glyph into an intermediate image
/// texture/pixmap and then render that into the actual scene, in a similar
/// fashion to bitmap glyphs.
pub struct ColorGlyph<'a> {
    pub(crate) skrifa_glyph: skrifa::color::ColorGlyph<'a>,
    pub(crate) location: LocationRef<'a>,
    pub(crate) font_ref: &'a FontRef<'a>,
    pub(crate) draw_transform: Affine,
    /// The rectangular area that should be filled with the rendered representation of the
    /// COLR glyph when painting.
    pub area: Rect,
    /// The width of the pixmap/texture in pixels to which the glyph should be rendered to.
    pub pix_width: u16,
    /// The height of the pixmap/texture in pixels to which the glyph should be rendered to.
    pub pix_height: u16,
}

impl Debug for ColorGlyph<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "ColorGlyph")
    }
}

#[derive(Clone, Default)]
pub(crate) struct OutlinePath(pub(crate) BezPath);

impl OutlinePath {
    pub(crate) fn new() -> Self {
        Self(BezPath::new())
    }
}

impl OutlinePen for OutlinePath {
    #[inline]
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to((x, y));
    }

    #[inline]
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to((x, y));
    }

    #[inline]
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.curve_to((cx0, cy0), (cx1, cy1), (x, y));
    }

    #[inline]
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.0.quad_to((cx, cy), (x, y));
    }

    #[inline]
    fn close(&mut self) {
        self.0.close_path();
    }
}

/// A normalized variation coordinate (for variable fonts) in 2.14 fixed point format.
///
/// In most cases, this can be [cast](bytemuck::cast_slice) from the
/// normalised coords provided by your text layout library.
///
/// Equivalent to [`skrifa::instance::NormalizedCoord`], but defined
/// in Vello so that Skrifa is not part of Vello's public API.
pub type NormalizedCoord = i16;

#[cfg(test)]
mod tests {
    use super::*;

    const _NORMALISED_COORD_SIZE_MATCHES: () =
        assert!(size_of::<skrifa::instance::NormalizedCoord>() == size_of::<NormalizedCoord>());
}
