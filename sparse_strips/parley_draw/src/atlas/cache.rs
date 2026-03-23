// Copyright 2026 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glyph atlas cache with LRU eviction.

use super::commands::AtlasCommandRecorder;
use super::key::GlyphCacheKey;
use super::key::SUBPIXEL_BUCKETS;
use super::region::{AtlasSlot, RasterMetrics};
use crate::Pixmap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::{Debug, Formatter};
use foldhash::fast::FixedState;
use hashbrown::HashMap;
use hashbrown::HashSet;
use hashbrown::hash_map::RawEntryMut;
use smallvec::SmallVec;
pub use vello_common::image_cache::ImageCache;
pub use vello_common::multi_atlas::AtlasConfig;
use vello_common::paint::ImageId;

/// Deterministic hash map type alias.
///
/// Uses `foldhash::fast::FixedState` instead of the default random-seeded hasher
/// so that iteration order is identical across processes. This ensures that LRU
/// eviction deallocates atlas regions in a deterministic order, producing
/// reproducible atlas packing regardless of which binary (CPU / hybrid) runs.
type FixedHashMap<K, V> = HashMap<K, V, FixedState>;

/// Fixed seed for deterministic hashing across all glyph cache maps.
const HASH_SEED: u64 = 0;
/// Compile-time empty map for static (non-variable) glyph entries.
const EMPTY_GLYPH_MAP: FixedHashMap<GlyphCacheKey, GlyphCacheEntry> =
    FixedHashMap::with_hasher(FixedState::with_seed(HASH_SEED));
/// Compile-time empty map for variable-font glyph entries, keyed by variation coordinates.
const EMPTY_VAR_MAP: FixedHashMap<VarKey, FixedHashMap<GlyphCacheKey, GlyphCacheEntry>> =
    FixedHashMap::with_hasher(FixedState::with_seed(HASH_SEED));

/// Padding in pixels added to each side of a glyph to prevent texture bleeding.
///
/// The hybrid (GPU) renderer samples atlas sub-images via `Extend::Pad`, which
/// clamps out-of-bounds coordinates to the edge texel. Without at least 1px of
/// transparent padding, strip-rasteriser overshoot at glyph boundaries would
/// either duplicate the edge row/column or bleed in content from a neighbouring
/// glyph allocation. 1px is sufficient: the overshoot is sub-pixel, and the
/// transparent padding absorbs it. This padding also enables a future switch to
/// native bilinear sampling in the hybrid renderer.
pub const GLYPH_PADDING: u16 = 1;

/// Configuration for glyph cache behavior.
#[derive(Clone, Debug)]
pub struct GlyphCacheConfig {
    /// Maximum age (in frames) before an unused entry is evicted.
    pub max_entry_age: u64,
    /// How often (in frames) to run the eviction pass.
    pub eviction_frequency: u64,
    /// Maximum font size (in ppem) that will be cached. Glyphs rendered at
    /// sizes above this threshold are drawn directly each frame, since very
    /// large glyphs consume disproportionate atlas space.
    pub max_cached_font_size: f32,
}

impl Default for GlyphCacheConfig {
    fn default() -> Self {
        Self {
            max_entry_age: 64,
            eviction_frequency: 64,
            max_cached_font_size: 128.0,
        }
    }
}

/// Common interface for glyph atlas caches.
///
/// Rendering code is generic over this trait, so different backends can provide
/// different storage strategies without duplicating orchestration logic.
pub trait GlyphCache {
    /// Look up a cached glyph.
    ///
    /// Returns `Some(AtlasSlot)` on cache hit (copy), `None` on miss.
    /// Updates the entry's access time on hit.
    fn get(&mut self, key: &GlyphCacheKey) -> Option<AtlasSlot>;

    /// Insert a glyph entry and allocate space in the atlas.
    ///
    /// Returns `(dst_x, dst_y, atlas_slot, recorder)` if successful, `None`
    /// if allocation failed (e.g., atlas is full).  The returned recorder
    /// accumulates draw commands for the atlas page the glyph was placed on.
    fn insert(
        &mut self,
        image_cache: &mut ImageCache,
        key: GlyphCacheKey,
        raster_metrics: RasterMetrics,
    ) -> Option<(u16, u16, AtlasSlot, &mut AtlasCommandRecorder)>;

    /// Queue a bitmap pixmap for later processing.
    fn push_pending_upload(
        &mut self,
        image_id: ImageId,
        pixmap: Arc<Pixmap>,
        atlas_slot: AtlasSlot,
    );

    /// Drain all pending bitmap uploads, keeping the allocation for reuse.
    fn drain_pending_uploads(&mut self) -> impl Iterator<Item = PendingBitmapUpload> + '_;

    /// Replay all pending atlas command recorders (one per dirty page).
    ///
    /// The closure receives each non-empty recorder by mutable reference.
    /// After the closure returns, the recorder's commands are cleared but
    /// the allocation is kept for reuse next frame.
    fn replay_pending_atlas_commands(&mut self, f: impl FnMut(&mut AtlasCommandRecorder));

    /// Drain all pending clear rects, keeping the allocation for reuse.
    ///
    /// Each rect describes an atlas region that was freed during
    /// [`maintain`](GlyphCache::maintain) and must be zeroed to transparent.
    /// Drain these **after** calling `maintain`.
    fn drain_pending_clear_rects(&mut self) -> impl Iterator<Item = PendingClearRect> + '_;

    /// Advance the frame counter and potentially evict old entries.
    fn maintain(&mut self, image_cache: &mut ImageCache);

    /// Clear the entire cache.
    fn clear(&mut self);

    /// Get the number of cached glyphs.
    fn len(&self) -> usize;

    /// Returns `true` if the cache contains no entries.
    fn is_empty(&self) -> bool;

    /// Get the number of cache hits since last `clear_stats()`.
    fn cache_hits(&self) -> u64;

    /// Get the number of cache misses since last `clear_stats()`.
    fn cache_misses(&self) -> u64;

    /// Reset cache hit/miss counters without clearing the cache itself.
    fn clear_stats(&mut self);

    /// Returns the cache configuration.
    fn config(&self) -> &GlyphCacheConfig;
}

/// A bitmap glyph pixmap awaiting GPU upload.
///
/// Accumulated during glyph encoding when a bitmap glyph is inserted into the
/// atlas cache. The application must drain these via
/// [`GlyphCache::drain_pending_uploads`] and upload each pixmap to the
/// GPU atlas at the position indicated by `image_id` (look up via
/// `ImageCache::get` to obtain atlas layer and offset).
#[derive(Debug)]
pub struct PendingBitmapUpload {
    /// The image ID allocated in the shared `ImageCache`.
    /// Use `image_cache.get(image_id)` to obtain `atlas_id` and `offset`.
    pub image_id: ImageId,
    /// The bitmap pixel data to upload.
    pub pixmap: Arc<Pixmap>,
    /// The atlas slot information for this glyph (includes dimensions).
    pub atlas_slot: AtlasSlot,
}

/// An atlas region that must be cleared to transparent.
///
/// Accumulated during eviction ([`GlyphAtlas::maintain`]) for every evicted
/// glyph. The application must drain these via
/// [`GlyphCache::drain_pending_clear_rects`] **after** calling `maintain` so
/// that freed atlas regions are zeroed before the slot is reused on a
/// subsequent frame. This prevents stale pixel data from bleeding through
/// when the renderer composites (`SrcOver`) onto the atlas.
#[derive(Clone, Copy, Debug)]
pub struct PendingClearRect {
    /// Which atlas page contains this region.
    pub page_index: u32,
    /// X position of the padded region in the atlas (pixels).
    pub x: u16,
    /// Y position of the padded region in the atlas (pixels).
    pub y: u16,
    /// Width of the padded region (pixels).
    pub width: u16,
    /// Height of the padded region (pixels).
    pub height: u16,
}

/// Core glyph atlas cache data shared by all renderer backends.
///
/// Contains the cache entries, LRU tracking, pending uploads, and statistics.
/// Does **not** own any pixel storage — that responsibility belongs to the
/// concrete wrapper types ([`CpuGlyphAtlas`](crate::renderers::vello_cpu::CpuGlyphAtlas), [`GpuGlyphAtlas`](crate::renderers::vello_hybrid::GpuGlyphAtlas)).
pub struct GlyphAtlas {
    /// Eviction configuration.
    eviction_config: GlyphCacheConfig,
    /// Entries for non-variable fonts.
    static_entries: FixedHashMap<GlyphCacheKey, GlyphCacheEntry>,
    /// Entries for variable fonts, keyed by variation coordinates.
    variable_entries: FixedHashMap<VarKey, FixedHashMap<GlyphCacheKey, GlyphCacheEntry>>,
    /// Current frame serial for LRU tracking.
    serial: u64,
    /// Serial of last eviction pass.
    last_eviction_serial: u64,
    /// Total cached glyph count (across all maps).
    entry_count: usize,
    /// Bitmap glyphs awaiting GPU upload.
    pending_uploads: Vec<PendingBitmapUpload>,
    /// Atlas regions that must be cleared to transparent before compositing.
    pending_clear_rects: Vec<PendingClearRect>,
    /// Outline and COLR glyph commands awaiting replay, indexed by atlas page.
    /// Uses `SmallVec` with inline capacity of 1 because most applications use
    /// a single atlas page; the common case avoids heap allocation entirely.
    pending_atlas_commands: SmallVec<[Option<AtlasCommandRecorder>; 1]>,
    /// Number of cache hits since last `clear_stats()`.
    cache_hits: u64,
    /// Number of cache misses since last `clear_stats()`.
    cache_misses: u64,
}

impl GlyphAtlas {
    /// Creates a new empty core cache with default eviction settings.
    pub fn new() -> Self {
        Self::with_config(GlyphCacheConfig::default())
    }

    /// Creates a new empty core cache with custom eviction settings.
    pub fn with_config(eviction_config: GlyphCacheConfig) -> Self {
        Self {
            eviction_config,
            static_entries: EMPTY_GLYPH_MAP,
            variable_entries: EMPTY_VAR_MAP,
            serial: 0,
            last_eviction_serial: 0,
            entry_count: 0,
            pending_uploads: Vec::new(),
            pending_clear_rects: Vec::new(),
            pending_atlas_commands: SmallVec::new(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Returns a reference to the cache configuration.
    pub fn config(&self) -> &GlyphCacheConfig {
        &self.eviction_config
    }

    /// Look up a cached glyph.
    pub fn get(&mut self, key: &GlyphCacheKey) -> Option<AtlasSlot> {
        let serial = self.serial;
        let entries = if key.var_coords.is_empty() {
            &mut self.static_entries
        } else {
            match self
                .variable_entries
                .raw_entry_mut()
                .from_key(&VarLookupKey(&key.var_coords))
            {
                RawEntryMut::Occupied(e) => e.into_mut(),
                RawEntryMut::Vacant(_) => {
                    self.cache_misses += 1;
                    return None;
                }
            }
        };

        match entries.get_mut(key) {
            Some(entry) => {
                entry.serial = serial;
                self.cache_hits += 1;
                Some(entry.atlas_slot)
            }
            None => {
                self.cache_misses += 1;
                None
            }
        }
    }

    /// Allocate atlas space and insert a cache entry.
    ///
    /// Returns `(page_index, dst_x, dst_y, atlas_slot)` on success. The caller is
    /// responsible for ensuring the atlas page at `page_index` exists in its
    /// storage backend.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "atlas offsets fit in u16 at reasonable atlas sizes"
    )]
    pub fn insert_entry(
        &mut self,
        image_cache: &mut ImageCache,
        key: GlyphCacheKey,
        raster_metrics: RasterMetrics,
    ) -> Option<(usize, u16, u16, AtlasSlot)> {
        let padded_w = u32::from(raster_metrics.width) + u32::from(GLYPH_PADDING) * 2;
        let padded_h = u32::from(raster_metrics.height) + u32::from(GLYPH_PADDING) * 2;

        let image_id = image_cache.allocate(padded_w, padded_h).ok()?;
        let resource = image_cache.get(image_id)?;
        let page_index = resource.atlas_id.as_u32() as usize;

        let x = resource.offset[0] + GLYPH_PADDING;
        let y = resource.offset[1] + GLYPH_PADDING;

        let atlas_slot = AtlasSlot {
            image_id,
            page_index: page_index as u32,
            x,
            y,
            width: raster_metrics.width,
            height: raster_metrics.height,
            bearing_x: raster_metrics.bearing_x,
            bearing_y: raster_metrics.bearing_y,
        };

        let entry = GlyphCacheEntry {
            atlas_slot,
            serial: self.serial,
        };

        let entries = if key.var_coords.is_empty() {
            &mut self.static_entries
        } else {
            match self
                .variable_entries
                .raw_entry_mut()
                .from_key(&VarLookupKey(&key.var_coords))
            {
                RawEntryMut::Occupied(e) => e.into_mut(),
                RawEntryMut::Vacant(e) => e.insert(key.var_coords.clone(), EMPTY_GLYPH_MAP).1,
            }
        };

        entries.insert(key, entry);
        self.entry_count += 1;

        Some((page_index, atlas_slot.x, atlas_slot.y, atlas_slot))
    }

    /// Drain all pending bitmap uploads, keeping the allocation for reuse.
    pub fn drain_pending_uploads(&mut self) -> impl Iterator<Item = PendingBitmapUpload> + '_ {
        self.pending_uploads.drain(..)
    }

    /// Drain all pending clear rects, keeping the allocation for reuse.
    pub fn drain_pending_clear_rects(&mut self) -> impl Iterator<Item = PendingClearRect> + '_ {
        self.pending_clear_rects.drain(..)
    }

    /// Queue a bitmap pixmap for later processing.
    pub fn push_pending_upload(
        &mut self,
        image_id: ImageId,
        pixmap: Arc<Pixmap>,
        atlas_slot: AtlasSlot,
    ) {
        self.pending_uploads.push(PendingBitmapUpload {
            image_id,
            pixmap,
            atlas_slot,
        });
    }

    /// Replay all pending atlas command recorders (one per dirty page).
    ///
    /// The closure receives each non-empty recorder by mutable reference.
    /// After the closure returns, the recorder's commands are cleared but
    /// the allocation is kept for reuse next frame.
    pub fn replay_pending_atlas_commands(&mut self, mut f: impl FnMut(&mut AtlasCommandRecorder)) {
        for slot in &mut self.pending_atlas_commands {
            if let Some(recorder) = slot.as_mut() {
                if !recorder.commands.is_empty() {
                    f(recorder);
                    recorder.commands.clear();
                }
            }
        }
    }

    /// Get (or create) the command recorder for the given atlas page.
    pub fn recorder_for_page(
        &mut self,
        page_index: u32,
        atlas_width: u16,
        atlas_height: u16,
    ) -> &mut AtlasCommandRecorder {
        let idx = page_index as usize;
        if self.pending_atlas_commands.len() <= idx {
            self.pending_atlas_commands.resize_with(idx + 1, || None);
        }
        self.pending_atlas_commands[idx]
            .get_or_insert_with(|| AtlasCommandRecorder::new(page_index, atlas_width, atlas_height))
    }

    /// Advance the frame counter and potentially evict old entries.
    pub fn maintain(&mut self, image_cache: &mut ImageCache) {
        self.tick();
        let frames_since_eviction = self.serial - self.last_eviction_serial;
        if frames_since_eviction < self.eviction_config.eviction_frequency {
            return;
        }

        self.last_eviction_serial = self.serial;
        self.evict_old_entries(image_cache);
    }

    /// Advance the frame counter.
    fn tick(&mut self) {
        self.serial += 1;
    }

    /// Evict entries that haven't been used recently.
    ///
    /// For each evicted entry, queues a [`PendingClearRect`] covering the full
    /// padded atlas region. The application must drain these via
    /// [`drain_pending_clear_rects`](GlyphAtlas::drain_pending_clear_rects) and
    /// zero each region so that stale pixel data doesn't bleed through when
    /// the slot is later reused and composited with `SrcOver`.
    fn evict_old_entries(&mut self, image_cache: &mut ImageCache) {
        let serial = self.serial;
        let max_entry_age = self.eviction_config.max_entry_age;
        let entry_count = &mut self.entry_count;
        let pending_clear_rects = &mut self.pending_clear_rects;

        let mut should_retain = |entry: &GlyphCacheEntry| -> bool {
            let age = serial - entry.serial;
            if age > max_entry_age {
                image_cache.deallocate(entry.atlas_slot.image_id);
                *entry_count = entry_count.saturating_sub(1);
                push_clear_rect_for_slot(pending_clear_rects, &entry.atlas_slot);
                false
            } else {
                true
            }
        };

        self.static_entries.retain(|_, entry| should_retain(entry));

        self.variable_entries.retain(|_, entries| {
            entries.retain(|_, entry| should_retain(entry));
            !entries.is_empty()
        });
    }

    /// Clear all cache entries, pending work queues, and statistics.
    pub fn clear(&mut self) {
        self.static_entries.clear();
        self.variable_entries.clear();
        self.serial = 0;
        self.last_eviction_serial = 0;
        self.entry_count = 0;
        self.pending_uploads.clear();
        self.pending_clear_rects.clear();
        self.pending_atlas_commands.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    /// Get the number of cached glyphs.
    #[inline]
    pub fn len(&self) -> usize {
        self.entry_count
    }

    /// Returns `true` if the cache contains no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Get the number of cache hits since last `clear_stats()`.
    #[inline]
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits
    }

    /// Get the number of cache misses since last `clear_stats()`.
    #[inline]
    pub fn cache_misses(&self) -> u64 {
        self.cache_misses
    }

    /// Reset cache hit/miss counters without clearing the cache itself.
    pub fn clear_stats(&mut self) {
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
}

/// Queue a clear rect covering the full padded region of an evicted atlas slot.
///
/// The slot's `x`/`y` are already inset by [`GLYPH_PADDING`] from the
/// allocation origin, so we subtract it back to get the padded top-left
/// corner and add `2 * GLYPH_PADDING` to each dimension.
fn push_clear_rect_for_slot(pending: &mut Vec<PendingClearRect>, slot: &AtlasSlot) {
    pending.push(PendingClearRect {
        page_index: slot.page_index,
        x: slot.x - GLYPH_PADDING,
        y: slot.y - GLYPH_PADDING,
        width: slot.width + 2 * GLYPH_PADDING,
        height: slot.height + 2 * GLYPH_PADDING,
    });
}

/// Statistics about cached glyphs.
#[cfg(debug_assertions)]
#[derive(Debug)]
pub struct GlyphCacheStats {
    /// Number of glyphs from static (non-variable) fonts.
    pub static_glyphs: usize,
    /// Number of glyphs from variable fonts.
    pub variable_glyphs: usize,
    /// Number of atlas pages currently allocated.
    pub page_count: usize,
    /// Number of unique glyph IDs (same glyph may have multiple entries due to subpixel).
    pub unique_glyph_ids: usize,
    /// Distribution of entries across subpixel buckets.
    pub subpixel_distribution: [usize; SUBPIXEL_BUCKETS as usize],
    /// List of unique font sizes used.
    pub sizes_used: Vec<f32>,
}

#[cfg(debug_assertions)]
impl GlyphCacheStats {
    /// Total number of cached glyph entries (static + variable).
    pub fn total_glyphs(&self) -> usize {
        self.static_glyphs + self.variable_glyphs
    }
}

#[cfg(debug_assertions)]
impl GlyphAtlas {
    /// Get detailed statistics about cached glyphs.
    pub fn stats(&self, page_count: usize) -> GlyphCacheStats {
        let mut unique_ids = HashSet::new();
        let mut subpixel_dist = [0; SUBPIXEL_BUCKETS as usize];
        let mut sizes = HashSet::new();

        for key in self.static_entries.keys() {
            unique_ids.insert(key.glyph_id);
            subpixel_dist[key.subpixel_x as usize] += 1;
            sizes.insert(key.size_bits);
        }

        let variable_count: usize = self.variable_entries.values().map(|m| m.len()).sum();

        for entries in self.variable_entries.values() {
            for key in entries.keys() {
                unique_ids.insert(key.glyph_id);
                subpixel_dist[key.subpixel_x as usize] += 1;
                sizes.insert(key.size_bits);
            }
        }

        GlyphCacheStats {
            static_glyphs: self.static_entries.len(),
            variable_glyphs: variable_count,
            page_count,
            unique_glyph_ids: unique_ids.len(),
            subpixel_distribution: subpixel_dist,
            sizes_used: sizes.into_iter().map(f32::from_bits).collect(),
        }
    }

    /// Log cache hit/miss statistics at debug level.
    pub fn log_hit_miss_stats(&self) {
        let total = self.cache_hits + self.cache_misses;
        let hit_rate = if total > 0 {
            (self.cache_hits as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        log::debug!("=== Cache Hit/Miss Statistics ===");
        log::debug!("Cache hits:   {}", self.cache_hits);
        log::debug!("Cache misses: {}", self.cache_misses);
        log::debug!("Total lookups: {}", total);
        log::debug!("Hit rate:     {:.2}%", hit_rate);
    }

    /// Log detailed atlas statistics at debug level.
    pub fn log_atlas_stats(&self, page_count: usize) {
        let stats = self.stats(page_count);
        log::debug!("=== Glyph Atlas Statistics ===");
        log::debug!("Total cached glyphs: {}", stats.total_glyphs());
        log::debug!("Unique glyph IDs: {}", stats.unique_glyph_ids);
        log::debug!("Atlas pages: {}", stats.page_count);
        log::debug!("Static font glyphs: {}", stats.static_glyphs);
        log::debug!("Variable font glyphs: {}", stats.variable_glyphs);
        log::debug!("Subpixel distribution: {:?}", stats.subpixel_distribution);
        log::debug!("Font sizes: {:?}", stats.sizes_used);

        if stats.unique_glyph_ids > 0 {
            let ratio = stats.total_glyphs() as f32 / stats.unique_glyph_ids as f32;
            log::debug!("Avg entries per unique glyph: {:.2}", ratio);
        }
    }

    /// Returns all cached glyph keys (for debugging).
    pub fn all_keys(&self) -> impl Iterator<Item = &GlyphCacheKey> {
        self.static_entries
            .keys()
            .chain(self.variable_entries.values().flat_map(|e| e.keys()))
    }

    /// Log all cached keys grouped by glyph ID at debug level.
    ///
    /// This is useful for understanding why the same glyph appears multiple
    /// times in the atlas (e.g., different subpixel positions or sizes).
    pub fn log_keys_grouped(&self) {
        let mut by_glyph: HashMap<u32, Vec<(&GlyphCacheKey, &str)>> = HashMap::new();

        for key in self.static_entries.keys() {
            by_glyph
                .entry(key.glyph_id)
                .or_default()
                .push((key, "stat"));
        }
        for entries in self.variable_entries.values() {
            for key in entries.keys() {
                by_glyph
                    .entry(key.glyph_id)
                    .or_default()
                    .push((key, "var "));
            }
        }

        log::debug!(
            "=== Glyph Keys Grouped by ID ({} unique) ===",
            by_glyph.len()
        );

        let mut ids: Vec<_> = by_glyph.keys().copied().collect();
        ids.sort();

        for glyph_id in ids {
            let keys = &by_glyph[&glyph_id];
            let suffix = if keys.len() == 1 { "entry" } else { "entries" };
            log::debug!("glyph_id {:4} ({} {}):", glyph_id, keys.len(), suffix);
            for (k, source) in keys {
                log::debug!(
                    "    [{}] subpx: {}, size: {:.2}, hinted: {}, font_id: {:016x}, font_index: {}",
                    source,
                    k.subpixel_x,
                    f32::from_bits(k.size_bits),
                    k.hinted,
                    k.font_id,
                    k.font_index,
                );
            }
        }
    }
}

impl Default for GlyphAtlas {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for GlyphAtlas {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GlyphAtlas")
            .field("entry_count", &self.entry_count)
            .field("static_entries", &self.static_entries.len())
            .field("variable_fonts", &self.variable_entries.len())
            .field("serial", &self.serial)
            .finish_non_exhaustive()
    }
}

/// Internal cache entry storing atlas slot and access time.
struct GlyphCacheEntry {
    /// Atlas slot information for blitting.
    atlas_slot: AtlasSlot,
    /// Frame serial when last accessed (for LRU eviction).
    serial: u64,
}

/// Key for variable font caches (owned version).
type VarKey = SmallVec<[skrifa::instance::NormalizedCoord; 4]>;

/// Lookup key for variable font caches (borrowed version).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
struct VarLookupKey<'a>(&'a [skrifa::instance::NormalizedCoord]);

impl hashbrown::Equivalent<VarKey> for VarLookupKey<'_> {
    fn equivalent(&self, other: &VarKey) -> bool {
        self.0 == other.as_slice()
    }
}

impl From<VarLookupKey<'_>> for VarKey {
    fn from(key: VarLookupKey<'_>) -> Self {
        Self::from_slice(key.0)
    }
}
