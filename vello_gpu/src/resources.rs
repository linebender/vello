// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Persistent renderer resources shared across frames.

#[cfg(feature = "text")]
use crate::text::GlyphAtlasResources;
use alloc::vec::Vec;
#[cfg(feature = "text")]
use glifo::GlyphPrepCache;
use vello_common::geometry::RectU16;
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::{AtlasConfig, AtlasId};
use vello_common::paint::ImageId;

/// Persistent resources required by Vello GPU for rendering.
///
/// A set of resources must only be used with the renderer instance associated with it.
#[derive(Debug)]
pub struct Resources {
    pub(crate) image_cache: ImageCache,
    /// Images passed to [`Resources::destroy_image`] whose atlas regions are not freed yet.
    destroyed_images: Vec<ImageId>,
    #[cfg(feature = "text")]
    pub(crate) glyph_prep_cache: GlyphPrepCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_resources: Option<GlyphAtlasResources>,
}

impl Resources {
    pub(crate) fn new(image_atlas_config: AtlasConfig) -> Self {
        Self {
            image_cache: ImageCache::new_with_config(image_atlas_config),
            destroyed_images: Vec::new(),
            #[cfg(feature = "text")]
            glyph_prep_cache: GlyphPrepCache::default(),
            // Will be initialized lazily.
            #[cfg(feature = "text")]
            glyph_resources: None,
        }
    }

    /// Returns the image cache, which allocates the atlas regions of images and cached glyphs.
    pub fn image_cache(&self) -> &ImageCache {
        &self.image_cache
    }

    /// Returns the image cache mutably, for callers that allocate atlas regions themselves.
    ///
    /// Release such regions with [`destroy_image`](Self::destroy_image).
    /// [`ImageCache::deallocate`] frees a region without clearing it, which is only safe for
    /// regions that were never written.
    pub fn image_cache_mut(&mut self) -> &mut ImageCache {
        &mut self.image_cache
    }

    /// Destroys an image and releases its atlas region. Returns `false` if there is no image
    /// with this ID or it was already destroyed.
    ///
    /// The region is cleared and freed at the start of the renderer's next `render` call. GPU
    /// work that uses the image, including renders already encoded, must be submitted before
    /// that call; later renders must not draw it.
    pub fn destroy_image(&mut self, image_id: ImageId) -> bool {
        let exists =
            self.image_cache.get(image_id).is_some() && !self.destroyed_images.contains(&image_id);
        if exists {
            self.destroyed_images.push(image_id);
        }
        exists
    }

    /// Clears the atlas regions of the images destroyed since the previous render and frees them.
    ///
    /// An image whose clear fails stays queued.
    pub(crate) fn free_destroyed_images<T, E>(
        &mut self,
        backend: &mut T,
        mut clear_region: impl FnMut(&mut T, AtlasId, RectU16) -> Result<(), E>,
    ) -> Result<(), E> {
        while let Some(&image_id) = self.destroyed_images.last() {
            if let Some(image) = self.image_cache.get(image_id) {
                let [x, y] = image.offset;
                let padding = image.padding;
                let region = RectU16::new(
                    x - padding,
                    y - padding,
                    x + image.width + padding,
                    y + image.height + padding,
                );
                clear_region(backend, image.atlas_id, region)?;
                self.image_cache.deallocate(image_id);
            }
            self.destroyed_images.pop();
        }

        Ok(())
    }
}
