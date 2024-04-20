// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::HashMap;

use super::{Encoding, StreamOffsets};

use peniko::{Font, Style};
use skrifa::instance::{NormalizedCoord, Size};
use skrifa::outline::{HintingInstance, HintingMode, LcdLayout, OutlineGlyphFormat};
use skrifa::{GlyphId, MetadataProvider, OutlineGlyphCollection};

/// A rough upper bound to limit the amount of memory we retain for the free
/// list.
///
/// TODO: make this more sophisticated taking actual memory size into account.
const MAX_FREE_LIST_LEN: usize = 64;

#[derive(Default)]
pub struct GlyphCache {
    glyphs: Vec<EncodedGlyph>,
    free_list: Vec<usize>,
    map: GlyphMap,
    var_map: HashMap<VarKey, GlyphMap>,
    hinting: HintCache,
    serial: u64,
}

impl GlyphCache {
    pub fn glyphs(&self) -> &[EncodedGlyph] {
        &self.glyphs
    }

    pub fn session<'a>(
        &'a mut self,
        font: &'a Font,
        coords: &'a [NormalizedCoord],
        size: f32,
        hint: bool,
        style: &'a Style,
    ) -> Option<GlyphCacheSession<'a>> {
        let font_id = font.data.id();
        let font_index = font.index;
        let font = skrifa::FontRef::from_index(font.data.as_ref(), font.index).ok()?;
        let map = if !coords.is_empty() {
            // This is still ugly in rust. Choices are:
            // 1. multiple lookups in the hashmap (implemented here)
            // 2. always allocate and copy the key
            // 3. use unsafe
            // Pick 1 bad option :(
            if self.var_map.contains_key(coords) {
                self.var_map.get_mut(coords).unwrap()
            } else {
                self.var_map.entry(coords.into()).or_default()
            }
        } else {
            &mut self.map
        };
        let outlines = font.outline_glyphs();
        let size = Size::new(size);
        let hinter = if hint {
            let key = HintKey {
                font_id,
                font_index,
                outlines: &outlines,
                size,
                coords,
            };
            self.hinting.get(&key)
        } else {
            None
        };
        // TODO: we're ignoring dashing for now
        let style_bits = match style {
            Style::Fill(fill) => super::path::Style::from_fill(*fill),
            Style::Stroke(stroke) => super::path::Style::from_stroke(stroke),
        };
        let style_bits: [u32; 2] = bytemuck::cast(style_bits);
        Some(GlyphCacheSession {
            glyphs: &mut self.glyphs,
            free_list: &mut self.free_list,
            map,
            font_id,
            font_index,
            coords,
            size,
            size_bits: size.ppem().unwrap().to_bits(),
            style,
            style_bits,
            outlines,
            hinter,
            serial: self.serial,
        })
    }

    pub fn prune(&mut self, max_age: u64) {
        let free_list = &mut self.free_list;
        let serial = self.serial;
        self.serial += 1;
        self.map.retain(|_, v| {
            if serial - v.serial > max_age {
                free_list.push(v.index);
                false
            } else {
                true
            }
        });
        self.var_map.retain(|_, map| {
            map.retain(|_, v| {
                if serial - v.serial > max_age {
                    free_list.push(v.index);
                    false
                } else {
                    true
                }
            });
            !map.is_empty()
        });
        // If we have a sufficient number of entries on the free list,
        // release the memory for some of them.
        if free_list.len() > MAX_FREE_LIST_LEN {
            let want_to_free = free_list.len() - MAX_FREE_LIST_LEN;
            let mut freed = 0;
            for ix in free_list {
                let glyph = &mut self.glyphs[*ix];
                if glyph.encoding.path_data.capacity() > 0 {
                    glyph.encoding = Encoding::default();
                    freed += 1;
                }
                if freed >= want_to_free {
                    break;
                }
            }
        }
    }
}

pub struct GlyphCacheSession<'a> {
    glyphs: &'a mut Vec<EncodedGlyph>,
    free_list: &'a mut Vec<usize>,
    map: &'a mut GlyphMap,
    font_id: u64,
    font_index: u32,
    coords: &'a [NormalizedCoord],
    size: Size,
    size_bits: u32,
    style: &'a Style,
    style_bits: [u32; 2],
    outlines: OutlineGlyphCollection<'a>,
    hinter: Option<&'a HintingInstance>,
    serial: u64,
}

impl<'a> GlyphCacheSession<'a> {
    pub fn get_or_insert(&mut self, glyph_id: u32) -> Option<(usize, StreamOffsets)> {
        let key = GlyphKey {
            font_id: self.font_id,
            font_index: self.font_index,
            glyph_id,
            font_size_bits: self.size_bits,
            style_bits: self.style_bits,
            hint: self.hinter.is_some(),
        };
        if let Some(entry) = self.map.get_mut(&key) {
            entry.serial = self.serial;
            let stream_sizes = self.glyphs.get(entry.index)?.stream_sizes;
            return Some((entry.index, stream_sizes));
        }
        let outline = self.outlines.get(GlyphId::new(key.glyph_id as u16))?;
        let index = if let Some(index) = self.free_list.pop() {
            index
        } else {
            let index = self.glyphs.len();
            self.glyphs.push(Default::default());
            index
        };
        let glyph = self.glyphs.get_mut(index)?;
        let encoding = &mut glyph.encoding;
        encoding.reset();
        glyph.stream_sizes = Default::default();
        let is_fill = match &self.style {
            Style::Fill(fill) => {
                encoding.encode_fill_style(*fill);
                true
            }
            Style::Stroke(stroke) => {
                encoding.encode_stroke_style(stroke);
                false
            }
        };
        use skrifa::outline::DrawSettings;
        let mut path = encoding.encode_path(is_fill);
        let draw_settings = if key.hint {
            if let Some(hinter) = self.hinter {
                DrawSettings::hinted(hinter, false)
            } else {
                DrawSettings::unhinted(self.size, self.coords)
            }
        } else {
            DrawSettings::unhinted(self.size, self.coords)
        };
        outline.draw(draw_settings, &mut path).ok()?;
        if path.finish(false) == 0 {
            encoding.reset();
        }
        let stream_sizes = encoding.stream_offsets();
        glyph.stream_sizes = stream_sizes;
        self.map.insert(
            key,
            GlyphEntry {
                index,
                serial: self.serial,
            },
        );
        Some((index, stream_sizes))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Default, Debug)]
struct GlyphKey {
    font_id: u64,
    font_index: u32,
    glyph_id: u32,
    font_size_bits: u32,
    style_bits: [u32; 2],
    hint: bool,
}

/// Outer level key for variable font caches.
type VarKey = smallvec::SmallVec<[NormalizedCoord; 4]>;

type GlyphMap = HashMap<GlyphKey, GlyphEntry>;

#[derive(Copy, Clone, Default)]
struct GlyphEntry {
    /// Index into glyphs vector.
    index: usize,
    /// Last use of this entry.
    serial: u64,
}

/// A cached encoding of a single glyph.
#[derive(Default)]
pub struct EncodedGlyph {
    pub encoding: Encoding,
    pub stream_sizes: StreamOffsets,
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
