// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::HashMap;

use super::{Encoding, StreamOffsets};

use peniko::kurbo::{BezPath, Shape};
use peniko::{Fill, Style};
use skrifa::instance::{NormalizedCoord, Size};
use skrifa::outline::{HintingInstance, HintingMode, LcdLayout, OutlineGlyphFormat, OutlinePen};
use skrifa::{GlyphId, OutlineGlyphCollection};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Default, Debug)]
pub struct GlyphKey {
    pub font_id: u64,
    pub font_index: u32,
    pub glyph_id: u32,
    pub font_size_bits: u32,
    pub hint: bool,
}

#[derive(Default)]
pub struct GlyphCache {
    pub encoding: Encoding,
    glyphs: HashMap<GlyphKey, CachedRange>,
    hinting: HintCache,
}

impl GlyphCache {
    pub fn clear(&mut self) {
        self.encoding.reset();
        self.glyphs.clear();
        // No need to clear the hinting cache
    }

    pub fn get_or_insert(
        &mut self,
        outlines: &OutlineGlyphCollection,
        key: GlyphKey,
        style: &Style,
        font_size: f32,
        coords: &[NormalizedCoord],
    ) -> Option<CachedRange> {
        let size = skrifa::instance::Size::new(font_size);
        let is_var = !coords.is_empty();
        let encoding_cache = &mut self.encoding;
        let hinting_cache = &mut self.hinting;
        let mut encode_glyph = || {
            let start = encoding_cache.stream_offsets();
            let fill = match style {
                Style::Fill(fill) => *fill,
                Style::Stroke(_) => Fill::NonZero,
            };
            // Make sure each glyph gets encoded with a style.
            // TODO: can probably optimize by setting style per run
            encoding_cache.force_next_transform_and_style();
            encoding_cache.encode_fill_style(fill);
            let mut path = encoding_cache.encode_path(true);
            let outline = outlines.get(GlyphId::new(key.glyph_id as u16))?;
            use skrifa::outline::DrawSettings;
            let draw_settings = if key.hint {
                if let Some(hint_instance) =
                    hinting_cache.get(&HintKey::new(outlines, &key, font_size, coords))
                {
                    DrawSettings::hinted(hint_instance, false)
                } else {
                    DrawSettings::unhinted(size, coords)
                }
            } else {
                DrawSettings::unhinted(size, coords)
            };
            match style {
                Style::Fill(_) => {
                    outline.draw(draw_settings, &mut path).ok()?;
                }
                Style::Stroke(stroke) => {
                    const STROKE_TOLERANCE: f64 = 0.01;
                    let mut pen = BezPathPen::default();
                    outline.draw(draw_settings, &mut pen).ok()?;
                    let stroked = peniko::kurbo::stroke(
                        pen.0.path_elements(STROKE_TOLERANCE),
                        stroke,
                        &Default::default(),
                        STROKE_TOLERANCE,
                    );
                    path.shape(&stroked);
                }
            }
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
            styles: self.end.styles - self.start.styles,
        }
    }
}

// A wrapper newtype so we can implement the `OutlinePen` trait.
#[derive(Default)]
struct BezPathPen(BezPath);

impl OutlinePen for BezPathPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to((x as f64, y as f64));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to((x as f64, y as f64));
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.0
            .quad_to((cx0 as f64, cy0 as f64), (x as f64, y as f64));
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.curve_to(
            (cx0 as f64, cy0 as f64),
            (cx1 as f64, cy1 as f64),
            (x as f64, y as f64),
        );
    }

    fn close(&mut self) {
        self.0.close_path();
    }
}

/// We keep this small to enable a simple LRU cache with a linear
/// search. Regenerating hinting data is low to medium cost so it's fine
/// to redo it occassionally.
const MAX_CACHED_HINT_INSTANCES: usize = 8;

pub struct HintKey<'a> {
    font_id: u64,
    font_index: u32,
    outlines: &'a OutlineGlyphCollection<'a>,
    size: Size,
    coords: &'a [NormalizedCoord],
}

impl<'a> HintKey<'a> {
    fn new(
        outlines: &'a OutlineGlyphCollection<'a>,
        glyph_key: &GlyphKey,
        size: f32,
        coords: &'a [NormalizedCoord],
    ) -> Self {
        Self {
            font_id: glyph_key.font_id,
            font_index: glyph_key.font_index,
            outlines,
            size: Size::new(size),
            coords,
        }
    }

    fn instance(&self) -> Option<HintingInstance> {
        HintingInstance::new(self.outlines, self.size, self.coords, HINTING_MODE).ok()
    }
}

const HINTING_MODE: HintingMode = HintingMode::Smooth {
    lcd_subpixel: Some(LcdLayout::Horizontal),
    preserve_linear_metrics: true,
};

#[derive(Default)]
struct HintCache {
    // Split caches for glyf/cff because the instance type can reuse
    // internal memory when reconfigured for the same format.
    glyf_entries: Vec<HintEntry>,
    cff_entries: Vec<HintEntry>,
    serial: u64,
}

impl HintCache {
    fn get(&mut self, key: &HintKey) -> Option<&HintingInstance> {
        let entries = match key.outlines.format()? {
            OutlineGlyphFormat::Glyf => &mut self.glyf_entries,
            OutlineGlyphFormat::Cff | OutlineGlyphFormat::Cff2 => &mut self.cff_entries,
        };
        let (entry_ix, is_current) = find_hint_entry(entries, key)?;
        let entry = entries.get_mut(entry_ix)?;
        self.serial += 1;
        entry.serial = self.serial;
        if !is_current {
            entry.font_id = key.font_id;
            entry.font_index = key.font_index;
            entry
                .instance
                .reconfigure(key.outlines, key.size, key.coords, HINTING_MODE)
                .ok()?;
        }
        Some(&entry.instance)
    }
}

struct HintEntry {
    font_id: u64,
    font_index: u32,
    instance: HintingInstance,
    serial: u64,
}

fn find_hint_entry(entries: &mut Vec<HintEntry>, key: &HintKey) -> Option<(usize, bool)> {
    let mut found_serial = u64::MAX;
    let mut found_index = 0;
    for (ix, entry) in entries.iter().enumerate() {
        if entry.font_id == key.font_id
            && entry.font_index == key.font_index
            && entry.instance.size() == key.size
            && entry.instance.location().coords() == key.coords
        {
            return Some((ix, true));
        }
        if entry.serial < found_serial {
            found_serial = entry.serial;
            found_index = ix;
        }
    }
    if entries.len() < MAX_CACHED_HINT_INSTANCES {
        let instance = key.instance()?;
        let ix = entries.len();
        entries.push(HintEntry {
            font_id: key.font_id,
            font_index: key.font_index,
            instance,
            serial: 0,
        });
        Some((ix, true))
    } else {
        Some((found_index, false))
    }
}
