// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::ops::Range;

use peniko::{FontData, Style};

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
