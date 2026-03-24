// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared resources for scene building and rendering.

use crate::scene::Scene;
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::AtlasConfig;

/// Shared resources for scene building and rendering.
///
/// Holds the image atlas allocator and (when the `text` feature is enabled)
/// glyph caches. Pass a mutable reference to scene-building methods that need
/// atlas allocation (e.g. [`Scene::glyph_run`](crate::Scene::glyph_run),
/// [`Renderer::upload_image`](crate::Renderer::upload_image)) and to
/// [`Renderer::render`](crate::Renderer::render).
#[derive(Debug)]
pub struct SceneResources {
    /// Atlas allocator shared between scene building and GPU rendering.
    pub image_cache: ImageCache,
    /// Glyph atlas caches.
    #[cfg(feature = "text")]
    pub(crate) glyph_caches: crate::text::GpuGlyphCaches,
    /// Scratch scene for rendering outline/COLR glyphs into atlas pages.
    #[cfg(feature = "text")]
    pub(crate) glyph_renderer: Scene,
}

impl SceneResources {
    /// Creates a new `SceneResources` with default settings.
    pub fn new() -> Self {
        let atlas_config = AtlasConfig::default();
        let (w, h) = atlas_config.atlas_size;
        Self {
            image_cache: ImageCache::new_with_config(atlas_config),
            #[cfg(feature = "text")]
            glyph_caches: Default::default(),
            #[cfg(feature = "text")]
            glyph_renderer: Scene::new(w as u16, h as u16),
        }
    }

    /// Creates a new `SceneResources` with a custom atlas configuration.
    pub fn new_with_config(atlas_config: AtlasConfig) -> Self {
        let (w, h) = atlas_config.atlas_size;
        Self {
            image_cache: ImageCache::new_with_config(atlas_config),
            #[cfg(feature = "text")]
            glyph_caches: Default::default(),
            #[cfg(feature = "text")]
            glyph_renderer: Scene::new(w as u16, h as u16),
        }
    }

    /// Run periodic maintenance (LRU eviction, etc.).
    ///
    /// Call this once per frame, typically before building a new scene.
    #[cfg(feature = "text")]
    pub fn maintain(&mut self) {
        self.glyph_caches.maintain(&mut self.image_cache);
    }
}

impl Default for SceneResources {
    fn default() -> Self {
        Self::new()
    }
}
