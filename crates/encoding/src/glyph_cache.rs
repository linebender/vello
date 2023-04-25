// Copyright 2022 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::HashMap;

use super::{Encoding, StreamOffsets};

use fello::scale::Scaler;
use fello::GlyphId;
use peniko::{Fill, Style};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Default, Debug)]
pub struct GlyphKey {
    pub font_id: u64,
    pub font_index: u32,
    pub glyph_id: u32,
    pub font_size: u32,
    pub hint: bool,
}

#[derive(Default)]
pub struct GlyphCache {
    pub encoding: Encoding,
    glyphs: HashMap<GlyphKey, CachedRange>,
}

impl GlyphCache {
    pub fn clear(&mut self) {
        self.encoding.reset(true);
        self.glyphs.clear();
    }

    pub fn get_or_insert(
        &mut self,
        key: GlyphKey,
        style: &Style,
        scaler: &mut Scaler,
    ) -> Option<CachedRange> {
        let is_fill = matches!(style, Style::Fill(_));
        let is_var = !scaler.normalized_coords().is_empty();
        let encoding_cache = &mut self.encoding;
        let mut encode_glyph = || {
            let start = encoding_cache.stream_offsets();
            match style {
                Style::Fill(Fill::NonZero) => encoding_cache.encode_linewidth(-1.0),
                Style::Fill(Fill::EvenOdd) => encoding_cache.encode_linewidth(-2.0),
                Style::Stroke(stroke) => encoding_cache.encode_linewidth(stroke.width),
            }
            let mut path = encoding_cache.encode_path(is_fill);
            scaler
                .outline(GlyphId::new(key.glyph_id as u16), &mut path)
                .ok()?;
            if path.finish(false) == 0 {
                return None;
            }
            let end = encoding_cache.stream_offsets();
            Some(CachedRange { start, end })
        };
        // For now, only cache non-zero filled, non-variable glyphs so we don't need to keep style
        // as part of the key.
        let range = if matches!(style, Style::Fill(Fill::NonZero)) && !is_var {
            use std::collections::hash_map::Entry;
            match self.glyphs.entry(key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => *entry.insert(encode_glyph()?),
            }
        } else {
            encode_glyph()?
        };
        Some(range)
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct CachedRange {
    pub start: StreamOffsets,
    pub end: StreamOffsets,
}

impl CachedRange {
    pub fn len(&self) -> StreamOffsets {
        StreamOffsets {
            path_tags: self.end.path_tags - self.start.path_tags,
            path_data: self.end.path_data - self.start.path_data,
            draw_tags: self.end.draw_tags - self.start.draw_tags,
            draw_data: self.end.draw_data - self.start.draw_data,
            transforms: self.end.transforms - self.start.transforms,
            linewidths: self.end.linewidths - self.start.linewidths,
        }
    }
}
