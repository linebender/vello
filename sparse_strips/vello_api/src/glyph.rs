// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types for glyphs.

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
