// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Persistent renderer resources shared across frames.

#[cfg(feature = "text")]
use crate::text::GlyphAtlasResources;
#[cfg(feature = "text")]
use glifo::GlyphPrepCache;
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::AtlasConfig;

/// Persistent resources required by Vello Hybrid for rendering.
#[derive(Debug)]
pub struct Resources {
    pub(crate) image_cache: ImageCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_prep_cache: GlyphPrepCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_resources: Option<GlyphAtlasResources>,
}

impl Resources {
    /// Create a new set of renderer resources.
    pub fn new() -> Self {
        Self::new_with_config(AtlasConfig::default())
    }

    /// Returns the CPU-side image cache for debugging.
    #[doc(hidden)]
    pub fn debug_image_cache(&self) -> &ImageCache {
        &self.image_cache
    }

    /// Returns the glyph atlas cache for debugging, if it has been initialized.
    #[cfg(feature = "text")]
    #[doc(hidden)]
    pub fn debug_glyph_atlas(&self) -> Option<&glifo::GlyphAtlas> {
        self.glyph_resources
            .as_ref()
            .map(|resources| &resources.glyph_atlas)
    }

    /// Create a new set of renderer resources with a custom image/glyph atlas configuration.
    ///
    /// This should match the
    /// [`image_atlas_config`](crate::MemorySettings::image_atlas_config) passed to the renderer so
    /// that allocations and the GPU atlas texture agree on the atlas size.
    // TODO: Requiring callers to keep this config in sync with the renderer's
    // `memory_settings.image_atlas_config` by hand is a footgun; tie them together so they can't
    // diverge. Note the renderer also normalizes the config against the device limits, so even
    // matching configs here can still end up disagreeing.
    pub fn new_with_config(image_atlas_config: AtlasConfig) -> Self {
        Self {
            image_cache: ImageCache::new_with_config(image_atlas_config),
            #[cfg(feature = "text")]
            glyph_prep_cache: GlyphPrepCache::default(),
            // Will be initialized lazily.
            #[cfg(feature = "text")]
            glyph_resources: None,
        }
    }
}

impl Default for Resources {
    fn default() -> Self {
        Self::new()
    }
}
