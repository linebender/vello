// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gradient ramp cache for `vello_hybrid` renderer.

use alloc::vec::Vec;
use hashbrown::HashMap;
use vello_common::encode::{EncodedGradient, GradientCacheKey};
use vello_common::fearless_simd::{Level, simd_dispatch};
use vello_common::peniko::color::cache_key::CacheKey;

/// Number of bytes per texel in the gradient texture.
/// Gradient textures use `Rgba8Unorm` format (4 bytes per texel).
/// This constant is used to convert between byte offsets and texel indices.
const BYTES_PER_TEXEL: u32 = 4;

#[derive(Debug)]
pub(crate) struct GradientRampCache {
    /// Current epoch for LRU tracking.
    epoch: u64,
    /// Cache mapping gradient signature to cached ramps and last access time.
    cache: HashMap<CacheKey<GradientCacheKey>, (CachedRamp, u64)>,
    /// Packed gradient luts.
    luts: Vec<u8>,
    /// Whether the packed luts needs to be re-uploaded.
    has_changed: bool,
    /// Maximum number of gradient cache entries to retain.
    retained_count: u32,
    /// SIMD level used for gradient LUT generation.
    level: Level,
}

impl GradientRampCache {
    /// Create a new gradient ramp cache with the specified retained count.
    pub(crate) fn new(retained_count: u32, level: Level) -> Self {
        Self {
            epoch: 0,
            cache: HashMap::new(),
            luts: Vec::new(),
            has_changed: false,
            retained_count,
            level,
        }
    }

    /// Get or generate a gradient ramp, returning its offset in the packed luts.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Conversion from usize to u32 is safe, used for texture coordinates"
    )]
    pub(crate) fn get_or_create_ramp(&mut self, gradient: &EncodedGradient) -> (u32, u32) {
        self.epoch += 1;

        // Check if we already have this gradient cached.
        if let Some((cached_ramp, last_used)) = self.cache.get_mut(&gradient.cache_key) {
            *last_used = self.epoch;
            return (cached_ramp.lut_start, cached_ramp.width);
        }

        // Generate new gradient LUT.
        let lut_start = self.luts.len() as u32 / BYTES_PER_TEXEL;
        generate_gradient_lut_dispatch(self.level, gradient, &mut self.luts);
        let lut_end = self.luts.len() as u32 / BYTES_PER_TEXEL;
        let width = lut_end - lut_start;
        let cached_ramp = CachedRamp { width, lut_start };
        self.has_changed = true;
        self.cache
            .insert(gradient.cache_key.clone(), (cached_ramp, self.epoch));

        (lut_start, width)
    }

    /// Maintain the gradient cache by evicting old entries.
    pub(crate) fn maintain(&mut self) {
        let entries_to_remove_count = self
            .cache
            .len()
            .saturating_sub(self.retained_count as usize);
        self.remove_lru_entries(entries_to_remove_count);
    }

    /// Get the size of the packed luts.
    pub(crate) fn luts_size(&self) -> usize {
        self.luts.len()
    }

    /// Check if the packed luts is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.luts.is_empty()
    }

    /// Check if the luts data has changed.
    pub(crate) fn has_changed(&self) -> bool {
        self.has_changed
    }

    /// Mark the luts as synced.
    pub(crate) fn mark_synced(&mut self) {
        self.has_changed = false;
    }

    /// Take ownership of the luts, leaving an empty vector in its place.
    pub(crate) fn take_luts(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.luts)
    }

    /// Restore the luts. The restored luts should have the same logical content as the original.
    pub(crate) fn restore_luts(&mut self, luts: Vec<u8>) {
        self.luts = luts;
    }

    /// Remove multiple least recently used cache entries and compact the luts vector efficiently.
    fn remove_lru_entries(&mut self, count: usize) {
        if count == 0 {
            return;
        }

        let entries_to_remove = self.get_lru_entries(count);
        if entries_to_remove.is_empty() {
            return;
        }
        for (key, _) in &entries_to_remove {
            self.cache.remove(key);
        }
        self.compact_luts(entries_to_remove);
        self.has_changed = true;
    }

    /// Compact the luts vector by removing multiple cached ramps
    fn compact_luts(&mut self, mut ramps_to_remove: Vec<(CacheKey<GradientCacheKey>, CachedRamp)>) {
        if ramps_to_remove.is_empty() {
            return;
        }

        // Sort by lut_start position (ascending) for efficient processing
        ramps_to_remove.sort_by_key(|(_, ramp)| ramp.lut_start);

        // Convert to byte ranges for easier processing
        let ranges_to_remove: Vec<(usize, usize)> = ramps_to_remove
            .iter()
            .map(|(_, ramp)| {
                let start = (ramp.lut_start * BYTES_PER_TEXEL) as usize;
                let end = start + (ramp.width * BYTES_PER_TEXEL) as usize;
                (start, end)
            })
            .collect();

        // Total bytes removed so far
        let mut write_offset = 0;
        // Current read position
        let mut read_pos = 0;
        // Index into ranges_to_remove
        let mut range_idx = 0;

        while read_pos < self.luts.len() {
            // Check if we're at the start of a range to remove
            if range_idx < ranges_to_remove.len() && read_pos == ranges_to_remove[range_idx].0 {
                let (start, end) = ranges_to_remove[range_idx];
                // Skip over the range to remove
                write_offset += end - start;
                read_pos = end;
                range_idx += 1;
            } else {
                // Copy byte from read position to write position (read_pos - write_offset)
                if write_offset > 0 {
                    self.luts[read_pos - write_offset] = self.luts[read_pos];
                }
                read_pos += 1;
            }
        }

        // Truncate the vector to remove the unused tail
        self.luts.truncate(self.luts.len() - write_offset);

        // Update lut_start values for remaining entries
        // Calculate how much data was removed before each ramp's original position
        for (_, (ramp, _)) in self.cache.iter_mut() {
            let mut removed_before = 0;
            for (_, removed_ramp) in &ramps_to_remove {
                if removed_ramp.lut_start < ramp.lut_start {
                    removed_before += removed_ramp.width;
                }
            }
            ramp.lut_start -= removed_before;
        }
    }

    /// Get multiple least recently used cache entries.
    fn get_lru_entries(&self, count: usize) -> Vec<(CacheKey<GradientCacheKey>, CachedRamp)> {
        if count == 0 || self.cache.is_empty() {
            return Vec::new();
        }

        let mut entries: Vec<_> = self
            .cache
            .iter()
            .map(|(key, (ramp, last_used))| (key.clone(), ramp.clone(), *last_used))
            .collect();

        // Sort by last_used (ascending) to get LRU entries first
        entries.sort_by_key(|(_, _, last_used)| *last_used);

        entries
            .into_iter()
            .take(count)
            .map(|(key, ramp, _)| (key, ramp))
            .collect()
    }
}

/// Cached gradient ramp data with metadata.
#[derive(Debug, Clone)]
pub(crate) struct CachedRamp {
    /// Width of this gradient's LUT.
    pub width: u32,
    /// Offset in luts where this ramp starts.
    pub lut_start: u32,
}

simd_dispatch!(fn generate_gradient_lut_dispatch(
    level,
    gradient: &vello_common::encode::EncodedGradient,
    output: &mut Vec<u8>
) = generate_gradient_lut_impl);

/// Generate the gradient LUT.
// TODO: Consider adding a method that generates LUT data directly into output buffer
// to avoid duplicate allocation when lut() is only used once (e.g., in gradient cache).
// The current approach allocates LUT in OnceCell and then copies to output, keeping
// both allocations alive.
#[inline(always)]
fn generate_gradient_lut_impl<S: vello_common::fearless_simd::Simd>(
    simd: S,
    gradient: &vello_common::encode::EncodedGradient,
    output: &mut Vec<u8>,
) {
    let lut = gradient.u8_lut(simd);
    let bytes: &[u8] = bytemuck::cast_slice(lut.lut());
    output.reserve(bytes.len());
    output.extend_from_slice(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use vello_common::color::{ColorSpaceTag, DynamicColor, HueDirection};
    use vello_common::encode::{EncodeExt, EncodedPaint};
    use vello_common::kurbo::{Affine, Point};
    use vello_common::peniko::{Color, ColorStop, ColorStops, Gradient, GradientKind};

    fn insert_entries(cache: &mut GradientRampCache, count: usize) {
        for i in 0..count {
            let offset = i as f32 / count as f32;

            let gradient: Gradient = create_gradient(offset);
            insert_entry(cache, gradient);
        }
    }

    fn insert_entry(cache: &mut GradientRampCache, gradient: Gradient) {
        let encoded_gradient = create_encoded_gradient(gradient);
        cache.get_or_create_ramp(&encoded_gradient);
    }

    fn create_encoded_gradient(gradient: Gradient) -> EncodedGradient {
        let mut encoded_paints = vec![];
        gradient.encode_into(&mut encoded_paints, Affine::IDENTITY);
        match encoded_paints.into_iter().last().unwrap() {
            EncodedPaint::Gradient(encoded_gradient) => encoded_gradient,
            _ => panic!("Expected a gradient paint"),
        }
    }

    fn create_gradient(offset: f32) -> Gradient {
        Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(100.0, 0.0),
            },
            stops: ColorStops(
                vec![
                    ColorStop {
                        offset: 0.0,
                        color: DynamicColor::from_alpha_color(Color::from_rgb8(255, 0, 0)),
                    },
                    ColorStop {
                        offset,
                        color: DynamicColor::from_alpha_color(Color::from_rgb8(0, 255, 0)),
                    },
                    ColorStop {
                        offset: 1.0,
                        color: DynamicColor::from_alpha_color(Color::from_rgb8(0, 0, 255)),
                    },
                ]
                .into(),
            ),
            interpolation_cs: ColorSpaceTag::Srgb,
            hue_direction: HueDirection::Shorter,
            ..Default::default()
        }
    }

    #[test]
    fn test_cache_empty() {
        let mut cache = GradientRampCache::new(5, Level::fallback());
        cache.maintain();

        assert_eq!(cache.cache.len(), 0);
        assert_eq!(cache.epoch, 0);
        assert!(cache.is_empty());
        assert!(!cache.has_changed());
    }

    #[test]
    fn test_unique_entry_creation() {
        let mut cache = GradientRampCache::new(5, Level::fallback());
        insert_entries(&mut cache, 4);
        cache.maintain();

        assert_eq!(cache.cache.len(), 4);
        assert!(!cache.is_empty());
        assert!(cache.has_changed());
    }

    #[test]
    fn test_no_eviction_under_limit() {
        let mut cache = GradientRampCache::new(5, Level::fallback());
        insert_entries(&mut cache, 4);
        cache.maintain();

        assert_eq!(cache.cache.len(), 4);
    }

    #[test]
    fn test_no_eviction_at_limit() {
        let mut cache = GradientRampCache::new(5, Level::fallback());
        insert_entries(&mut cache, 5);
        cache.maintain();

        assert_eq!(cache.cache.len(), 5);
    }

    #[test]
    fn test_eviction_over_limit() {
        let mut cache = GradientRampCache::new(5, Level::fallback());
        insert_entries(&mut cache, 10);
        cache.maintain();

        // Should trigger eviction since we're over the limit
        assert_eq!(cache.cache.len(), 5);
    }

    #[test]
    fn test_lut_compaction_and_offset_updates() {
        let mut cache = GradientRampCache::new(2, Level::fallback());

        // Start from 1 to keep LUT sizes consistent, making it easier to test LUT size
        // before and after eviction.
        for i in 1..3 {
            let gradient = create_gradient(i as f32 / 10.0);
            insert_entry(&mut cache, gradient);
        }
        let initial_luts_size = cache.luts_size();

        // This should evict the LRU entry and compact the luts
        for i in 3..5 {
            let gradient = create_gradient(i as f32 / 10.0);
            insert_entry(&mut cache, gradient);
        }
        cache.maintain();
        assert_eq!(cache.cache.len(), 2);

        // Verify that luts vector was compacted properly
        assert_eq!(cache.luts_size(), initial_luts_size);

        // Verify that remaining entries have valid, updated offsets
        let mut offsets = Vec::new();
        for (_, (ramp, _)) in &cache.cache {
            offsets.push((ramp.lut_start, ramp.width));
        }
        offsets.sort();

        // The first entry should start at 0 after compaction
        assert_eq!(offsets[0].0, 0);
        // Each subsequent entry should start where the previous one ended
        for i in 1..offsets.len() {
            let expected_start = offsets[i - 1].0 + offsets[i - 1].1;
            assert_eq!(offsets[i].0, expected_start);
        }

        // Total luts size should match sum of all widths * BYTES_PER_TEXEL
        let total_width: u32 = offsets.iter().map(|(_, width)| width).sum();
        assert_eq!(cache.luts.len(), (total_width * BYTES_PER_TEXEL) as usize);
    }

    #[test]
    fn test_correct_lru_eviction() {
        let mut cache = GradientRampCache::new(3, Level::fallback());

        // Insert 3 gradients to fill the cache
        let gradient1 = create_gradient(0.1);
        let gradient2 = create_gradient(0.2);
        let gradient3 = create_gradient(0.3);

        insert_entry(&mut cache, gradient1.clone());
        insert_entry(&mut cache, gradient2.clone());
        insert_entry(&mut cache, gradient3.clone());

        assert_eq!(cache.cache.len(), 3);

        // Access gradient1 and gradient3 to make them more recently used
        // This should make gradient2 the least recently used
        insert_entry(&mut cache, gradient1.clone());
        insert_entry(&mut cache, gradient3.clone());

        // Now insert a new gradient that should evict gradient2
        let gradient4 = create_gradient(0.4);
        insert_entry(&mut cache, gradient4.clone());
        cache.maintain();

        // Cache should still have 3 entries
        assert_eq!(cache.cache.len(), 3);

        // Check gradient1 is still cached
        let encoded_gradient1 = create_encoded_gradient(gradient1);
        assert!(
            cache.cache.contains_key(&encoded_gradient1.cache_key),
            "Gradient1 should still be cached"
        );

        // Check gradient2 was evicted
        let encoded_gradient2 = create_encoded_gradient(gradient2);
        assert!(
            !cache.cache.contains_key(&encoded_gradient2.cache_key),
            "Gradient2 should have been evicted"
        );

        // Check gradient3 is still cached
        let encoded_gradient3 = create_encoded_gradient(gradient3);
        assert!(
            cache.cache.contains_key(&encoded_gradient3.cache_key),
            "Gradient3 should still be cached"
        );

        // Check gradient4 is cached
        let encoded_gradient4 = create_encoded_gradient(gradient4);
        assert!(
            cache.cache.contains_key(&encoded_gradient4.cache_key),
            "Gradient4 should be cached"
        );
    }

    #[test]
    fn test_take_and_restore_luts() {
        let mut cache = GradientRampCache::new(5, Level::fallback());

        let gradient1 = create_gradient(0.1);
        let gradient2 = create_gradient(0.2);
        let gradient3 = create_gradient(0.3);
        insert_entry(&mut cache, gradient1.clone());
        insert_entry(&mut cache, gradient2.clone());
        insert_entry(&mut cache, gradient3.clone());
        cache.maintain();
        let original_size = cache.luts_size();
        let original_cache_size = cache.cache.len();

        let luts = cache.take_luts();
        assert_eq!(luts.len(), original_size);
        assert_eq!(cache.luts_size(), 0);
        assert!(cache.is_empty());

        cache.restore_luts(luts);

        assert_eq!(cache.luts_size(), original_size);
        assert_eq!(cache.cache.len(), original_cache_size);
        assert!(!cache.is_empty());

        let encoded_gradient1 = create_encoded_gradient(gradient1.clone());
        let encoded_gradient2 = create_encoded_gradient(gradient2.clone());
        let encoded_gradient3 = create_encoded_gradient(gradient3.clone());
        assert!(cache.cache.contains_key(&encoded_gradient1.cache_key));
        assert!(cache.cache.contains_key(&encoded_gradient2.cache_key));
        assert!(cache.cache.contains_key(&encoded_gradient3.cache_key));
    }

    #[test]
    fn test_lut_start_invalidation() {
        let mut cache = GradientRampCache::new(2, Level::fallback());

        let gradient_1 = create_encoded_gradient(create_gradient(0.1));
        let gradient_2 = create_encoded_gradient(create_gradient(0.2));
        let gradient_3 = create_encoded_gradient(create_gradient(0.3));

        // Frame 1 start
        let _lut_start_1 = cache.get_or_create_ramp(&gradient_1).0;
        let _lut_start_2 = cache.get_or_create_ramp(&gradient_2).0;
        // Frame 1 end

        // Frame 2 start
        let lut_start_2 = cache.get_or_create_ramp(&gradient_2).0;
        let lut_start_3 = cache.get_or_create_ramp(&gradient_3).0;
        // Frame 2 end

        // Gradient lut shouldn't mutate within a frame
        assert_eq!(
            lut_start_2,
            cache.cache.get(&gradient_2.cache_key).unwrap().0.lut_start
        );
        assert_eq!(
            lut_start_3,
            cache.cache.get(&gradient_3.cache_key).unwrap().0.lut_start
        );
    }
}
