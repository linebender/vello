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
///
/// A set of resources must only be used with the renderer instance associated with it.
#[derive(Debug)]
pub struct Resources {
    pub(crate) image_cache: ImageCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_prep_cache: GlyphPrepCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_resources: Option<GlyphAtlasResources>,
}

impl Resources {
    pub(crate) fn new(image_atlas_config: AtlasConfig) -> Self {
        Self {
            image_cache: ImageCache::new_with_config(image_atlas_config),
            #[cfg(feature = "text")]
            glyph_prep_cache: GlyphPrepCache::default(),
            // Will be initialized lazily.
            #[cfg(feature = "text")]
            glyph_resources: None,
        }
    }

    /// Shared access to the image atlas cache.
    pub fn image_cache(&self) -> &ImageCache {
        &self.image_cache
    }

    /// Exclusive access to the image atlas cache.
    ///
    /// Deallocating directly through this handle frees the CPU-side slot
    /// without clearing the freed region on the GPU; prefer
    /// [`Renderer::destroy_image`](crate::Renderer::destroy_image) unless you
    /// clear the region yourself.
    pub fn image_cache_mut(&mut self) -> &mut ImageCache {
        &mut self.image_cache
    }
}
