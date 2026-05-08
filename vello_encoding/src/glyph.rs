// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::ops::Range;

use peniko::{
    FontData, Style,
    kurbo::{Diagonal2, Join},
};

use super::{StreamOffsets, Transform};

/// Positioned glyph.
#[derive(Copy, Clone, Default, Debug)]
pub struct Glyph {
    /// Glyph identifier.
    pub id: u32,
    /// X-offset in run, relative to transform.
    pub x: f32,
    /// Y-offset in run, relative to transform.
    pub y: f32,
}

/// Synthetic embolden settings for a glyph run.
#[derive(Clone, Copy, Debug)]
pub struct FontEmbolden {
    /// Synthetic embolden amount.
    pub amount: Diagonal2,
    /// Join style used when expanding outlines.
    pub join: Join,
    /// Miter limit used when expanding outlines.
    pub miter_limit: f64,
    /// Tolerance used when expanding outlines.
    pub tolerance: f64,
}

impl FontEmbolden {
    /// Create synthetic embolden settings with default expansion controls.
    pub fn new(amount: Diagonal2) -> Self {
        Self {
            amount,
            ..Self::default()
        }
    }

    /// Set the join style used when expanding outlines.
    pub fn with_join(mut self, join: Join) -> Self {
        self.join = join;
        self
    }

    /// Set the miter limit used when expanding outlines.
    pub fn with_miter_limit(mut self, miter_limit: f64) -> Self {
        self.miter_limit = miter_limit;
        self
    }

    /// Set the tolerance used when expanding outlines.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl Default for FontEmbolden {
    fn default() -> Self {
        Self {
            amount: Diagonal2::new(0.0, 0.0),
            join: Join::Miter,
            miter_limit: 4.0,
            tolerance: 0.1,
        }
    }
}

/// Properties for a sequence of glyphs in an encoding.
#[derive(Clone)]
pub struct GlyphRun {
    /// Font for all glyphs in the run.
    pub font: FontData,
    /// Global run transform.
    pub transform: Transform,
    /// Per-glyph transform.
    pub glyph_transform: Option<Transform>,
    /// Size of the font in pixels per em.
    pub font_size: f32,
    /// Synthetic embolden settings.
    pub font_embolden: FontEmbolden,
    /// True if hinting is enabled.
    pub hint: bool,
    /// Range of normalized coordinates in the parent encoding.
    pub normalized_coords: Range<usize>,
    /// Fill or stroke style.
    pub style: Style,
    /// Range of glyphs in the parent encoding.
    pub glyphs: Range<usize>,
    /// Stream offsets where this glyph run should be inserted.
    pub stream_offsets: StreamOffsets,
}
