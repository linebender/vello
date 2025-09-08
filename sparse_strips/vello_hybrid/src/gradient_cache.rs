// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gradient ramp cache for `vello_hybrid` renderer.

use alloc::vec::Vec;
use core::hash::{Hash, Hasher};
use hashbrown::HashMap;
use vello_common::color::{ColorSpaceTag, HueDirection};
use vello_common::encode::EncodedGradient;
use vello_common::fearless_simd::{Level, simd_dispatch};
use vello_common::peniko::ColorStops;
use vello_common::peniko::color::cache_key::{BitEq, BitHash, CacheKey};

const RETAINED_COUNT: usize = 64;

#[derive(Debug, Default)]
pub(crate) struct GradientRampCache {
    /// Current epoch for LRU tracking.
    epoch: u64,
    /// Cache mapping gradient signature to cached ramps and last access time.
    cache: HashMap<CacheKey<GradientRampKey>, (CachedRamp, u64)>,
    /// Packed gradient luts.
    luts: Vec<u8>,
    /// Whether the packed luts needs to be re-uploaded.
    has_changed: bool,
}

// TODO: Rebuild the packed luts or allow to rewrite evicted gradient ramps after eviction.
impl GradientRampCache {
    /// Create a new gradient ramp cache.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Get or generate a gradient ramp, returning its offset in the packed luts.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Conversion from usize to u32 is safe, used for texture coordinates"
    )]
    pub(crate) fn get_or_create_ramp(&mut self, gradient: &EncodedGradient) -> (u32, u32) {
        self.epoch += 1;

        // Create cache key from gradient color properties.
        let cache_key = CacheKey(GradientRampKey {
            stops: gradient.stops.clone(),
            interpolation_cs: gradient.interpolation_cs,
            hue_direction: gradient.hue_direction,
        });

        // Check if we already have this gradient cached.
        if let Some((cached_ramp, last_used)) = self.cache.get_mut(&cache_key) {
            *last_used = self.epoch;
            return (cached_ramp.lut_start, cached_ramp.width);
        }

        // Generate new gradient LUT.
        let lut_start = (self.luts.len() / 4) as u32;
        generate_gradient_lut_dispatch(Level::new(), gradient, &mut self.luts);
        let lut_end = (self.luts.len() / 4) as u32;
        let width = lut_end - lut_start;
        let cached_ramp = CachedRamp { width, lut_start };
        self.has_changed = true;

        if self.cache.len() >= RETAINED_COUNT {
            self.evict_old_entries();
        }

        self.cache.insert(cache_key, (cached_ramp, self.epoch));

        (lut_start, width)
    }

    /// Get the packed luts.
    pub(crate) fn luts(&self) -> &[u8] {
        &self.luts
    }

    /// Get the size of the packed luts.
    pub(crate) fn size(&self) -> usize {
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

    /// Periodic maintenance - call occasionally to prevent unbounded growth.
    pub(crate) fn maintain(&mut self) {
        self.epoch += 1;

        if self.cache.len() > RETAINED_COUNT {
            self.evict_old_entries();
        }
    }

    /// Evict old cache entries using LRU policy.
    fn evict_old_entries(&mut self) {
        // Find entries that haven't been used in the last few epochs.
        let cutoff_epoch = self.epoch.saturating_sub(3);

        let entries_to_remove: Vec<_> = self
            .cache
            .iter()
            .filter(|(_, (_, last_used))| *last_used < cutoff_epoch)
            .map(|(key, _)| key.clone())
            .collect();

        // Remove old entries.
        for key in entries_to_remove {
            self.cache.remove(&key);
        }

        // If still too many entries, remove oldest ones.
        if self.cache.len() > RETAINED_COUNT {
            let mut entries: Vec<_> = self
                .cache
                .iter()
                .map(|(k, (_, last_used))| (k.clone(), *last_used))
                .collect();
            entries.sort_by_key(|(_, last_used)| *last_used);

            let to_remove = entries.len() - RETAINED_COUNT;
            for (key, _) in entries.into_iter().take(to_remove) {
                self.cache.remove(&key);
            }
        }
    }
}

/// Cache key for gradient color ramps based on color-affecting properties.
#[derive(Debug, Clone)]
pub(crate) struct GradientRampKey {
    /// The color stops (offsets + colors).
    pub stops: ColorStops,
    /// Color space used for interpolation.
    pub interpolation_cs: ColorSpaceTag,
    /// Hue direction used for interpolation.
    pub hue_direction: HueDirection,
}

impl BitHash for GradientRampKey {
    fn bit_hash<H: Hasher>(&self, state: &mut H) {
        self.stops.bit_hash(state);
        core::mem::discriminant(&self.interpolation_cs).hash(state);
        core::mem::discriminant(&self.hue_direction).hash(state);
    }
}

impl BitEq for GradientRampKey {
    fn bit_eq(&self, other: &Self) -> bool {
        self.stops.bit_eq(&other.stops)
            && self.interpolation_cs == other.interpolation_cs
            && self.hue_direction == other.hue_direction
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
#[inline(always)]
fn generate_gradient_lut_impl<S: vello_common::fearless_simd::Simd>(
    simd: S,
    gradient: &vello_common::encode::EncodedGradient,
    output: &mut Vec<u8>,
) {
    let lut = gradient.u8_lut(simd);
    for color in lut.lut() {
        output.extend_from_slice(&[color[0], color[1], color[2], color[3]]);
    }
}
