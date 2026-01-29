// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::multi_atlas::{AtlasConfig, AtlasError, AtlasId, MultiAtlasManager};
use alloc::vec::Vec;
use guillotiere::AllocId;
use vello_common::paint::ImageId;

/// Represents an image resource for rendering.
#[derive(Debug)]
pub(crate) struct ImageResource {
    /// The width of the image.
    pub(crate) width: u16,
    /// The height of the image.
    pub(crate) height: u16,
    /// The Id of the atlas containing this image.
    pub(crate) atlas_id: AtlasId,
    /// The offset of the image within its atlas.
    pub(crate) offset: [u16; 2],
    /// The atlas allocation ID for deallocation.
    atlas_alloc_id: AllocId,
}

/// Manages image resources for the renderer.
pub(crate) struct ImageCache {
    /// Multi-atlas manager for handling multiple texture atlases.
    atlas_manager: MultiAtlasManager,
    /// Vector of optional image resources (None = free slot).
    slots: Vec<Option<ImageResource>>,
    /// Stack of free indices.
    free_idxs: Vec<usize>,
}

impl core::fmt::Debug for ImageCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let atlas_stats = self.atlas_manager.atlas_stats();

        f.debug_struct("ImageCache")
            .field("slots", &self.slots)
            .field("free_idxs", &self.free_idxs)
            .field("atlas_count", &self.atlas_manager.atlas_count())
            .field("atlas_stats", &atlas_stats)
            .finish()
    }
}

impl ImageCache {
    /// Create a new image cache with custom atlas configuration.
    pub(crate) fn new_with_config(config: AtlasConfig) -> Self {
        Self {
            atlas_manager: MultiAtlasManager::new(config),
            slots: Vec::new(),
            free_idxs: Vec::new(),
        }
    }

    /// Get an image resource by its Id.
    pub(crate) fn get(&self, id: ImageId) -> Option<&ImageResource> {
        self.slots.get(id.as_u32() as usize)?.as_ref()
    }

    /// Allocate an image in the cache.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "u16 is enough for the offset and width/height"
    )]
    pub(crate) fn allocate(&mut self, width: u32, height: u32) -> Result<ImageId, AtlasError> {
        let atlas_alloc = self.atlas_manager.try_allocate(width, height)?;

        let slot_idx = self.free_idxs.pop().unwrap_or_else(|| {
            // No free slots, append to vector
            let index = self.slots.len();
            // Placeholder, will be replaced
            self.slots.push(None);
            index
        });

        let image_id = ImageId::new(slot_idx as u32);
        let image_resource = ImageResource {
            width: width as u16,
            height: height as u16,
            atlas_id: atlas_alloc.atlas_id,
            offset: [
                atlas_alloc.allocation.rectangle.min.x as u16,
                atlas_alloc.allocation.rectangle.min.y as u16,
            ],
            atlas_alloc_id: atlas_alloc.allocation.id,
        };
        self.slots[slot_idx] = Some(image_resource);

        Ok(image_id)
    }

    /// Deallocate an image from the cache, returning the image resource if it existed.
    pub(crate) fn deallocate(&mut self, id: ImageId) -> Option<ImageResource> {
        let index = id.as_u32() as usize;
        if let Some(image_resource) = self.slots.get_mut(index).and_then(Option::take) {
            // Deallocate from the appropriate atlas
            self.atlas_manager
                .deallocate(
                    image_resource.atlas_id,
                    image_resource.atlas_alloc_id,
                    image_resource.width as u32,
                    image_resource.height as u32,
                )
                .unwrap();
            self.free_idxs.push(index);
            Some(image_resource)
        } else {
            None
        }
    }

    /// Get access to the atlas manager.
    pub(crate) fn atlas_manager(&self) -> &MultiAtlasManager {
        &self.atlas_manager
    }

    /// Get the number of atlases.
    pub(crate) fn atlas_count(&self) -> usize {
        self.atlas_manager.atlas_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ATLAS_SIZE: u32 = 1024;

    #[test]
    fn test_insert_single_image() {
        let mut cache = ImageCache::new_with_config(AtlasConfig {
            atlas_size: (ATLAS_SIZE, ATLAS_SIZE),
            ..Default::default()
        });

        let id = cache.allocate(100, 100).unwrap();

        assert_eq!(id.as_u32(), 0);
        let resource = cache.get(id).unwrap();
        assert_eq!(resource.width, 100);
        assert_eq!(resource.height, 100);
        // First image should be at origin
        assert_eq!(resource.offset, [0, 0]);
    }

    #[test]
    fn test_insert_multiple_images() {
        let mut cache = ImageCache::new_with_config(AtlasConfig {
            atlas_size: (ATLAS_SIZE, ATLAS_SIZE),
            ..Default::default()
        });

        let id1 = cache.allocate(50, 50).unwrap();
        let id2 = cache.allocate(75, 75).unwrap();

        assert_eq!(id1.as_u32(), 0);
        assert_eq!(id2.as_u32(), 1);

        let resource1 = cache.get(id1).unwrap();
        let resource2 = cache.get(id2).unwrap();

        assert_eq!(resource1.width, 50);
        assert_eq!(resource2.width, 75);

        // Second image should be placed adjacent to first
        assert_ne!(resource1.offset, resource2.offset);
    }

    #[test]
    fn test_get_nonexistent_image() {
        let cache: ImageCache = ImageCache::new_with_config(AtlasConfig {
            atlas_size: (ATLAS_SIZE, ATLAS_SIZE),
            ..Default::default()
        });

        assert!(cache.get(ImageId::new(0)).is_none());
        assert!(cache.get(ImageId::new(999)).is_none());
    }

    #[test]
    fn test_remove_image() {
        let mut cache = ImageCache::new_with_config(AtlasConfig {
            atlas_size: (ATLAS_SIZE, ATLAS_SIZE),
            ..Default::default()
        });

        let id = cache.allocate(100, 100).unwrap();
        assert!(cache.get(id).is_some());

        cache.deallocate(id);
        assert!(cache.get(id).is_none());
    }

    #[test]
    fn test_remove_nonexistent_image() {
        let mut cache: ImageCache = ImageCache::new_with_config(AtlasConfig {
            atlas_size: (ATLAS_SIZE, ATLAS_SIZE),
            ..Default::default()
        });

        // Should not panic when unregistering non-existent image
        cache.deallocate(ImageId::new(0));
        cache.deallocate(ImageId::new(999));
    }

    #[test]
    fn test_slot_reuse_after_remove() {
        let mut cache = ImageCache::new_with_config(AtlasConfig {
            atlas_size: (ATLAS_SIZE, ATLAS_SIZE),
            ..Default::default()
        });

        // Register three images
        let id1 = cache.allocate(50, 50).unwrap();
        let id2 = cache.allocate(60, 60).unwrap();
        let id3 = cache.allocate(70, 70).unwrap();

        assert_eq!(id1.as_u32(), 0);
        assert_eq!(id2.as_u32(), 1);
        assert_eq!(id3.as_u32(), 2);

        // Unregister the middle one
        cache.deallocate(id2);
        assert!(cache.get(id2).is_none());

        // Register a new image - should reuse slot 1
        let id4 = cache.allocate(80, 80).unwrap();
        // Reused slot 1
        assert_eq!(id4.as_u32(), 1);

        // Verify other images are still there
        assert!(cache.get(id1).is_some());
        assert!(cache.get(id3).is_some());
        assert!(cache.get(id4).is_some());
        assert_eq!(cache.get(id4).unwrap().width, 80);
    }

    #[test]
    fn test_multiple_remove_and_reuse() {
        let mut cache = ImageCache::new_with_config(AtlasConfig {
            atlas_size: (ATLAS_SIZE, ATLAS_SIZE),
            ..Default::default()
        });

        // Register several images
        let ids: Vec<_> = (0..5)
            .map(|i| cache.allocate(100 + i * 10, 100 + i * 10).unwrap())
            .collect();

        // Unregister some in the middle
        cache.deallocate(ids[1]);
        cache.deallocate(ids[3]);

        // Register new images - should reuse the freed slots
        let new_id1 = cache.allocate(200, 200).unwrap();
        let new_id2 = cache.allocate(300, 300).unwrap();

        // Should have reused slots 3 and 1 (in reverse order due to stack behavior)
        assert_eq!(new_id1.as_u32(), 3);
        assert_eq!(new_id2.as_u32(), 1);
        assert_ne!(new_id1.as_u32(), new_id2.as_u32());
    }
}
