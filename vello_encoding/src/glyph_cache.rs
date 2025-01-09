// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::HashMap;
use std::sync::Arc;

use super::{Encoding, StreamOffsets};

use peniko::{Font, Style};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use skrifa::instance::{NormalizedCoord, Size};
use skrifa::outline::{HintingInstance, HintingOptions, OutlineGlyphFormat};
use skrifa::{GlyphId, MetadataProvider, OutlineGlyphCollection};

#[derive(Default)]
pub(crate) struct GlyphCache {
    map: GlyphMap,
    var_map: HashMap<VarKey, GlyphMap>,
    cached_count: usize,
    serial: u64,
    last_prune_serial: u64,
    unresolved: Vec<(VarKey, GlyphKey, Font)>,
}

impl GlyphCache {
    pub(crate) fn session<'a>(
        &'a mut self,
        font: &'a Font,
        coords: &'a [NormalizedCoord],
        size: f32,
        hint: bool,
        style: &'a Style,
    ) -> Option<GlyphCacheSession<'a>> {
        let font_id = font.data.id();
        let font_index = font.index;
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
        let size = Size::new(size);
        // TODO: we're ignoring dashing for now
        let style_bits = match style {
            Style::Fill(fill) => super::path::Style::from_fill(*fill),
            Style::Stroke(stroke) => super::path::Style::from_stroke(stroke),
        };
        let style_bits: [u32; 2] = bytemuck::cast(style_bits);
        Some(GlyphCacheSession {
            map,
            font_id,
            font_index,
            coords,
            size_bits: size.ppem().unwrap().to_bits(),
            style_bits,
            hint,
            font,
            serial: self.serial,
            unresolved: &mut self.unresolved,
        })
    }

    pub(crate) fn maintain(&mut self) {
        self.unresolved.clear();
        // Maximum number of resolve phases where we'll retain an unused glyph
        const MAX_ENTRY_AGE: u64 = 64;
        // Maximum number of resolve phases before we force a prune
        const PRUNE_FREQUENCY: u64 = 64;
        // Always prune if the cached count is greater than this value
        const CACHED_COUNT_THRESHOLD: usize = 256;
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
                self.cached_count -= 1;
                false
            } else {
                true
            }
        });
        self.var_map.retain(|_, map| {
            map.retain(|_, entry| {
                if serial - entry.serial > MAX_ENTRY_AGE {
                    self.cached_count -= 1;
                    false
                } else {
                    true
                }
            });
            !map.is_empty()
        });
    }
    pub(crate) fn resolve_in_parallel(&mut self) {
        let _span = tracing::info_span!("Resolving glyph outlines", count = self.unresolved.len())
            .entered();
        let result = self
            .unresolved
            .par_iter()
            .map_init(HintCache::default, |hint_cache, (coords, glyph, font)| {
                let _span = tracing::trace_span!("Resolving single glyph").entered();
                (
                    resolve_single_glyph(hint_cache, coords, glyph, font),
                    coords,
                    glyph,
                )
            })
            .collect_vec_list();
        for (result, coords, glyph) in result.into_iter().flatten() {
            let map = if !coords.is_empty() {
                self.var_map.get_mut(coords).unwrap()
            } else {
                &mut self.map
            };
            let (encoding, offsets) = result.unwrap_or_default();
            map.get_mut(glyph).unwrap().status = GlyphEntryStatus::Resolved {
                encoding,
                stream_sizes: offsets,
            };
        }
    }
    pub(crate) fn get_resolved(&mut self, idx: usize) -> (Arc<Encoding>, StreamOffsets) {
        let (coords, glyph, _) = &mut self.unresolved[idx];
        let map = if !coords.is_empty() {
            self.var_map.get_mut(coords).unwrap()
        } else {
            &mut self.map
        };
        match &mut map.get_mut(glyph).unwrap().status {
            GlyphEntryStatus::Resolved {
                encoding,
                stream_sizes,
            } => (encoding.clone(), *stream_sizes),
            GlyphEntryStatus::Unresolved { .. } => unreachable!(),
        }
    }
}

fn resolve_single_glyph(
    hint_cache: &mut HintCache,
    coords: &smallvec::SmallVec<[skrifa::raw::types::F2Dot14; 8]>,
    glyph: &GlyphKey,
    font: &Font,
) -> Option<(Arc<Encoding>, StreamOffsets)> {
    let font_id = font.data.id();
    let font_index = font.index;
    let font = skrifa::FontRef::from_index(font.data.as_ref(), font.index).ok()?;

    let outlines = {
        let _span = tracing::trace_span!("Getting font outline builder").entered();
        font.outline_glyphs()
    };
    let size = Size::new(f32::from_bits(glyph.font_size_bits));
    let outline = {
        let _span = tracing::trace_span!("Getting Glyph Outline").entered();
        outlines.get(GlyphId::new(glyph.glyph_id))?
    };
    let mut encoding = Encoding::default();
    encoding.reset();
    let style: crate::Style = bytemuck::cast(glyph.style_bits);
    encoding.encode_style(style);
    let is_fill = style.is_fill();
    use skrifa::outline::DrawSettings;
    let mut path = {
        let _span = tracing::trace_span!("Encoding path").entered();
        encoding.encode_path(is_fill)
    };
    let hinter = if glyph.hint {
        let key = HintKey {
            font_id,
            font_index,
            outlines: &outlines,
            size,
            coords,
        };
        hint_cache.get(&key)
    } else {
        None
    };
    let draw_settings = if let Some(hinter) = hinter {
        DrawSettings::hinted(hinter, false)
    } else {
        DrawSettings::unhinted(size, &**coords)
    };
    {
        let _span = tracing::trace_span!("Drawing span").entered();
        outline.draw(draw_settings, &mut path).ok()?;
    }
    if path.finish(false) == 0 {
        encoding.reset();
    }
    let stream_sizes = encoding.stream_offsets();
    let arc_encoding = Arc::new(encoding);
    Some((arc_encoding, stream_sizes))
}

pub(crate) struct GlyphCacheSession<'a> {
    map: &'a mut GlyphMap,
    font_id: u64,
    font_index: u32,
    coords: &'a [NormalizedCoord],
    size_bits: u32,
    style_bits: [u32; 2],
    font: &'a Font,
    serial: u64,
    hint: bool,
    unresolved: &'a mut Vec<(VarKey, GlyphKey, Font)>,
}

impl GlyphCacheSession<'_> {
    pub(crate) fn get_or_insert(&mut self, glyph_id: u32) -> GlyphEntryStatus {
        let key = GlyphKey {
            font_id: self.font_id,
            font_index: self.font_index,
            glyph_id,
            font_size_bits: self.size_bits,
            style_bits: self.style_bits,
            hint: self.hint,
        };
        if let Some(entry) = self.map.get_mut(&key) {
            entry.serial = self.serial;
            return entry.status.clone();
        }
        let index = self.unresolved.len();
        self.unresolved
            .push((self.coords.into(), key, self.font.clone()));
        let result = GlyphEntryStatus::Unresolved { index };
        self.map.insert(
            key,
            GlyphEntry {
                status: result.clone(),
                serial: self.serial,
            },
        );
        result

        // let outline = self.outlines.get(GlyphId::new(key.glyph_id))?;
        // let mut encoding = Encoding::default();
        // encoding.reset();
        // let is_fill = match &self.style {
        //     Style::Fill(fill) => {
        //         encoding.encode_fill_style(*fill);
        //         true
        //     }
        //     Style::Stroke(stroke) => {
        //         encoding.encode_stroke_style(stroke);
        //         false
        //     }
        // };
        // use skrifa::outline::DrawSettings;
        // let mut path = encoding.encode_path(is_fill);
        // let draw_settings = if key.hint {
        //     if let Some(hinter) = self.hinter {
        //         DrawSettings::hinted(hinter, false)
        //     } else {
        //         DrawSettings::unhinted(self.size, self.coords)
        //     }
        // } else {
        //     DrawSettings::unhinted(self.size, self.coords)
        // };
        // outline.draw(draw_settings, &mut path).ok()?;
        // if path.finish(false) == 0 {
        //     encoding.reset();
        // }
        // let stream_sizes = encoding.stream_offsets();
        // let arc_encoding = Arc::new(encoding);
        // self.map.insert(
        //     key,
        //     GlyphEntry {
        //         encoding: arc_encoding.clone(),
        //         stream_sizes,
        //         serial: self.serial,
        //     },
        // );
        // *self.cached_count += 1;
        // Some((arc_encoding, stream_sizes))
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

// #[derive(Clone, Default)]
struct GlyphEntry {
    status: GlyphEntryStatus,
    /// Last use of this entry.
    serial: u64,
}

#[derive(Clone)]
pub(crate) enum GlyphEntryStatus {
    Resolved {
        encoding: Arc<Encoding>,
        stream_sizes: StreamOffsets,
    },
    Unresolved {
        index: usize,
    },
}

/// We keep this small to enable a simple LRU cache with a linear
/// search. Regenerating hinting data is low to medium cost so it's fine
/// to redo it occasionally.
const MAX_CACHED_HINT_INSTANCES: usize = 2;

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
pub(crate) struct HintCache {
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
