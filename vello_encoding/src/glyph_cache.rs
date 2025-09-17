// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::HashMap;
use std::sync::Arc;

use super::{Encoding, StreamOffsets};

use peniko::{FontData, Style};
use skrifa::instance::{NormalizedCoord, Size};
use skrifa::outline::{HintingInstance, HintingOptions, OutlineGlyphFormat};
use skrifa::{GlyphId, MetadataProvider, OutlineGlyphCollection};

#[derive(Default)]
pub(crate) struct GlyphCache {
    free_list: Vec<Arc<Encoding>>,
    map: GlyphMap,
    var_map: HashMap<VarKey, GlyphMap>,
    cached_count: usize,
    hinting: HintCache,
    serial: u64,
    last_prune_serial: u64,
}

impl GlyphCache {
    pub(crate) fn session<'a>(
        &'a mut self,
        font: &'a FontData,
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
            Style::Stroke(stroke) => super::path::Style::from_stroke(stroke)?,
        };
        let style_bits: [u32; 2] = bytemuck::cast(style_bits);
        Some(GlyphCacheSession {
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
            cached_count: &mut self.cached_count,
        })
    }

    pub(crate) fn maintain(&mut self) {
        // Maximum number of resolve phases where we'll retain an unused glyph
        const MAX_ENTRY_AGE: u64 = 64;
        // Maximum number of resolve phases before we force a prune
        const PRUNE_FREQUENCY: u64 = 64;
        // Always prune if the cached count is greater than this value
        const CACHED_COUNT_THRESHOLD: usize = 256;
        // Number of encoding buffers we'll keep on the free list
        const MAX_FREE_LIST_SIZE: usize = 32;
        let free_list = &mut self.free_list;
        let serial = self.serial;
        self.serial += 1;
        // Don't iterate over the whole cache every frame
        if serial - self.last_prune_serial < PRUNE_FREQUENCY
            && self.cached_count < CACHED_COUNT_THRESHOLD
        {
            return;
        }
        self.last_prune_serial = serial;
        self.map.retain(|_, entry| {
            if serial - entry.serial > MAX_ENTRY_AGE {
                if free_list.len() < MAX_FREE_LIST_SIZE {
                    free_list.push(entry.encoding.clone());
                }
                self.cached_count -= 1;
                false
            } else {
                true
            }
        });
        self.var_map.retain(|_, map| {
            map.retain(|_, entry| {
                if serial - entry.serial > MAX_ENTRY_AGE {
                    if free_list.len() < MAX_FREE_LIST_SIZE {
                        free_list.push(entry.encoding.clone());
                    }
                    self.cached_count -= 1;
                    false
                } else {
                    true
                }
            });
            !map.is_empty()
        });
    }
}

pub(crate) struct GlyphCacheSession<'a> {
    free_list: &'a mut Vec<Arc<Encoding>>,
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
    cached_count: &'a mut usize,
}

impl GlyphCacheSession<'_> {
    pub(crate) fn get_or_insert(
        &mut self,
        glyph_id: u32,
    ) -> Option<(Arc<Encoding>, StreamOffsets)> {
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
            return Some((entry.encoding.clone(), entry.stream_sizes));
        }
        let outline = self.outlines.get(GlyphId::new(key.glyph_id))?;
        let mut encoding = self.free_list.pop().unwrap_or_default();
        let encoding_ptr = Arc::make_mut(&mut encoding);
        encoding_ptr.reset();
        let is_fill = match &self.style {
            Style::Fill(fill) => {
                encoding_ptr.encode_fill_style(*fill);
                true
            }
            Style::Stroke(stroke) => {
                let encoded_stroke = encoding_ptr.encode_stroke_style(stroke);
                debug_assert!(encoded_stroke, "Stroke width is non-zero");
                false
            }
        };
        use skrifa::outline::DrawSettings;
        let mut path = encoding_ptr.encode_path(is_fill);
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
            encoding_ptr.reset();
        }
        let stream_sizes = encoding_ptr.stream_offsets();
        self.map.insert(
            key,
            GlyphEntry {
                encoding: encoding.clone(),
                stream_sizes,
                serial: self.serial,
            },
        );
        *self.cached_count += 1;
        Some((encoding, stream_sizes))
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
///
/// Inline size of 8 maximizes the internal storage of the small vec.
type VarKey = smallvec::SmallVec<[NormalizedCoord; 8]>;

type GlyphMap = HashMap<GlyphKey, GlyphEntry>;

#[derive(Clone, Default)]
struct GlyphEntry {
    encoding: Arc<Encoding>,
    stream_sizes: StreamOffsets,
    /// Last use of this entry.
    serial: u64,
}

/// We keep this small to enable a simple LRU cache with a linear
/// search. Regenerating hinting data is low to medium cost so it's fine
/// to redo it occasionally.
const MAX_CACHED_HINT_INSTANCES: usize = 8;

pub(crate) struct HintKey<'a> {
    font_id: u64,
    font_index: u32,
    outlines: &'a OutlineGlyphCollection<'a>,
    size: Size,
    coords: &'a [NormalizedCoord],
}

impl HintKey<'_> {
    fn instance(&self) -> Option<HintingInstance> {
        HintingInstance::new(self.outlines, self.size, self.coords, HINTING_OPTIONS).ok()
    }
}

// TODO: We might want to expose making these configurable in future.
// However, these options are probably fine for most users.
const HINTING_OPTIONS: HintingOptions = HintingOptions {
    engine: skrifa::outline::Engine::AutoFallback,
    target: skrifa::outline::Target::Smooth {
        mode: skrifa::outline::SmoothMode::Lcd,
        symmetric_rendering: false,
        preserve_linear_metrics: true,
    },
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
    fn get(&mut self, key: &HintKey<'_>) -> Option<&HintingInstance> {
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
                .reconfigure(key.outlines, key.size, key.coords, HINTING_OPTIONS)
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

fn find_hint_entry(entries: &mut Vec<HintEntry>, key: &HintKey<'_>) -> Option<(usize, bool)> {
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
