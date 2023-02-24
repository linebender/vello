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

use std::collections::HashMap;

use super::{Encoding, StreamOffsets};
use crate::glyph::GlyphProvider;

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
        scaler: &mut GlyphProvider,
    ) -> Option<CachedRange> {
        // For now, only cache non-zero filled glyphs so we don't need to keep style
        // as part of the key.
        let is_nz_fill = matches!(style, Style::Fill(Fill::NonZero));
        if is_nz_fill {
            if let Some(range) = self.glyphs.get(&key) {
                return Some(*range);
            }
        }
        let start = self.encoding.stream_offsets();
        scaler.encode_glyph(key.glyph_id as u16, style, &mut self.encoding)?;
        let end = self.encoding.stream_offsets();
        let range = CachedRange { start, end };
        if is_nz_fill {
            self.glyphs.insert(key, range);
        }
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
