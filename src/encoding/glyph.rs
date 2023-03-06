// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

use std::ops::Range;

use peniko::{Font, Style};

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
    pub font: Font,
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
