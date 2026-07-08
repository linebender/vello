// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Persistent renderer resources shared across frames.

#[cfg(all(feature = "text", any(feature = "webgl", feature = "wgpu")))]
use crate::text::GlyphAtlasResources;
#[cfg(feature = "text")]
use glifo::GlyphPrepCache;
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::AtlasConfig;

/// Persistent resources required by Vello Hybrid for rendering.
#[derive(Debug)]
pub struct Resources {
    #[cfg_attr(
        not(any(feature = "webgl", feature = "wgpu")),
        expect(
            dead_code,
            reason = "this is used by renderer backends, which may be disabled"
        )
    )]
    pub(crate) image_cache: ImageCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_prep_cache: GlyphPrepCache,
    #[cfg(all(feature = "text", any(feature = "webgl", feature = "wgpu")))]
    pub(crate) glyph_resources: Option<GlyphAtlasResources>,
}

impl Resources {
    /// Create a new set of renderer resources.
    pub fn new() -> Self {
        Self {
            image_cache: ImageCache::new_with_config(AtlasConfig::default()),
            #[cfg(feature = "text")]
            glyph_prep_cache: GlyphPrepCache::default(),
            // Will be initialized lazily.
            #[cfg(all(feature = "text", any(feature = "webgl", feature = "wgpu")))]
            glyph_resources: None,
        }
    }
}

impl Default for Resources {
    fn default() -> Self {
        Self::new()
    }
}
