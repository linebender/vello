// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Embedded fonts and a minimal left-to-right layout.
//!
//! Layout deliberately avoids shaping: workloads only need realistic glyph streams, and keeping
//! layout simple keeps it out of the way when profiling Glifo itself.

use std::sync::Arc;

use glifo::Glyph;
use skrifa::instance::{LocationRef, Size};
use skrifa::{FontRef, MetadataProvider};
use vello_common::peniko::{Blob, FontData};

const ROBOTO: &[u8] = include_bytes!("../../../../assets/roboto/Roboto-Regular.ttf");
const INCONSOLATA: &[u8] = include_bytes!("../../../../assets/inconsolata/Inconsolata.ttf");
const NOTO_COLR: &[u8] =
    include_bytes!("../../../../assets/noto_color_emoji/NotoColorEmoji-Subset.ttf");
const NOTO_CBTF: &[u8] =
    include_bytes!("../../../../assets/noto_color_emoji/NotoColorEmoji-CBTF-Subset.ttf");

/// Index into [`Fonts::faces`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FaceId {
    Roboto = 0,
    Inconsolata = 1,
    NotoColr = 2,
    NotoBitmap = 3,
}

pub(crate) struct Face {
    pub(crate) name: &'static str,
    pub(crate) data: FontData,
}

impl Face {
    fn new(name: &'static str, bytes: &'static [u8]) -> Self {
        Self {
            name,
            data: FontData::new(Blob::new(Arc::new(bytes)), 0),
        }
    }

    fn font_ref(&self) -> FontRef<'_> {
        FontRef::from_index(self.data.data.data(), self.data.index).expect("embedded font is valid")
    }

    /// Returns the IDs of glyphs mapped from any code point, in code point order.
    pub(crate) fn mapped_glyph_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .font_ref()
            .charmap()
            .mappings()
            .map(|(_, glyph_id)| glyph_id.to_u32())
            .filter(|&id| id != 0)
            .collect();
        ids.dedup();
        ids
    }
}

pub(crate) struct Fonts {
    pub(crate) faces: [Face; 4],
}

impl Fonts {
    pub(crate) fn new() -> Self {
        Self {
            faces: [
                Face::new("Roboto", ROBOTO),
                Face::new("Inconsolata", INCONSOLATA),
                Face::new("Noto Color Emoji (COLR)", NOTO_COLR),
                Face::new("Noto Color Emoji (CBDT)", NOTO_CBTF),
            ],
        }
    }

    pub(crate) fn face(&self, id: FaceId) -> &Face {
        &self.faces[id as usize]
    }
}

/// Glyph advances for one face at one size.
pub(crate) struct Shaper<'a> {
    metrics: skrifa::metrics::GlyphMetrics<'a>,
    charmap: skrifa::charmap::Charmap<'a>,
    space_advance: f32,
}

impl<'a> Shaper<'a> {
    pub(crate) fn new(face: &'a Face, size: f32) -> Self {
        let font = face.font_ref();
        let metrics = font.glyph_metrics(Size::new(size), LocationRef::default());
        let charmap = font.charmap();
        let space_advance = charmap
            .map(' ')
            .and_then(|id| metrics.advance_width(id))
            .unwrap_or(size * 0.25);
        Self {
            metrics,
            charmap,
            space_advance,
        }
    }

    pub(crate) fn advance(&self, glyph_id: u32) -> f32 {
        self.metrics
            .advance_width(glyph_id.into())
            .unwrap_or(self.space_advance)
    }

    /// Appends the glyphs for `word` at `(x, y)` and returns the advance.
    fn push_word(&self, word: &str, x: f32, y: f32, out: &mut Vec<Glyph>) -> f32 {
        let mut pen = x;
        for ch in word.chars() {
            let id = self.charmap.map(ch).map_or(0, |id| id.to_u32());
            out.push(Glyph { id, x: pen, y });
            pen += self.advance(id);
        }
        pen - x
    }

    fn measure(&self, word: &str) -> f32 {
        word.chars()
            .map(|ch| self.advance(self.charmap.map(ch).map_or(0, |id| id.to_u32())))
            .sum()
    }

    /// Greedily wraps `text` into lines no wider than `max_width`, starting at baseline `y`.
    ///
    /// Returns one glyph vector per line together with its baseline.
    pub(crate) fn wrap(
        &self,
        text: &str,
        x: f32,
        mut y: f32,
        max_width: f32,
        line_height: f32,
    ) -> Vec<(Vec<Glyph>, f32)> {
        let mut lines = Vec::new();
        let mut line = Vec::new();
        let mut pen = 0.0;
        for word in text.split_whitespace() {
            let width = self.measure(word);
            if pen > 0.0 && pen + width > max_width {
                lines.push((std::mem::take(&mut line), y));
                y += line_height;
                pen = 0.0;
            }
            pen += self.push_word(word, x + pen, y, &mut line) + self.space_advance;
        }
        if !line.is_empty() {
            lines.push((line, y));
        }
        lines
    }
}
