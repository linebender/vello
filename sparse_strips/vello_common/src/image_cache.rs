// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image resource caching with multi-atlas allocation.
//!
//! This module provides an [`ImageCache`] that manages image resources across multiple texture
//! atlases, supporting allocation, deallocation, and slot reuse.

use crate::multi_atlas::{
    AllocId, AllocationStrategy, AtlasConfig, AtlasError, AtlasId, MultiAtlasManager,
};
use crate::paint::ImageId;
use alloc::vec::Vec;

/// Represents an image resource for rendering.
#[derive(Debug)]
pub struct ImageResource {
    /// The width of the image.
    pub width: u16,
    /// The height of the image.
    pub height: u16,
    /// The Id of the atlas containing this image.
    pub atlas_id: AtlasId,
    /// The offset of the image within its atlas (does not include padding, i.e. it points to the
    /// position of the first actual top-left pixel).
    pub offset: [u16; 2],
    /// The number of transparent padding pixels around the image in the atlas.
    pub padding: u16,
    /// The atlas allocation ID for deallocation.
    atlas_alloc_id: AllocId,
}

impl ImageResource {
    /// Returns the offset as `[u32; 2]`.
    pub fn offsets(&self) -> [u32; 2] {
        [self.offset[0] as u32, self.offset[1] as u32]
    }

    /// Returns the size as `[u32; 2]`.
    pub fn size(&self) -> [u32; 2] {
        [self.width as u32, self.height as u32]
    }
}

/// Manages image resources for the renderer.
pub struct ImageCache {
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
    pub fn new_with_config(config: AtlasConfig) -> Self {
        Self {
            atlas_manager: MultiAtlasManager::new(config),
            slots: Vec::new(),
            free_idxs: Vec::new(),
        }
    }

    /// Create a new dummy image atlas that is supposed to act as a stub.
    pub fn new_dummy() -> Self {
        Self::new_with_config(AtlasConfig {
            initial_atlas_count: 1,
            max_atlases: 1,
            atlas_size: (1, 1),
            auto_grow: false,
            allocation_strategy: AllocationStrategy::FirstFit,
        })
    }

    /// Get an image resource by its Id.
    pub fn get(&self, id: ImageId) -> Option<&ImageResource> {
        self.slots.get(id.as_u32() as usize)?.as_ref()
    }

    /// Allocate an image in the cache, with optional transparent padding.
    pub fn allocate(
        &mut self,
        width: u32,
        height: u32,
        padding: u16,
    ) -> Result<ImageId, AtlasError> {
        self.allocate_excluding(width, height, padding, None)
    }

    /// Allocate an image in the cache, with optional transparency padding
    /// and optionally excluding a specific atlas.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "u16 is enough for the offset and width/height"
    )]
    pub fn allocate_excluding(
        &mut self,
        width: u32,
        height: u32,
        padding: u16,
        exclude_atlas_id: Option<AtlasId>,
    ) -> Result<ImageId, AtlasError> {
        let padded_width = width + u32::from(padding) * 2;
        let padded_height = height + u32::from(padding) * 2;
        let atlas_alloc = self.atlas_manager.try_allocate_excluding(
            padded_width,
            padded_height,
            exclude_atlas_id,
        )?;

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
                atlas_alloc.allocation.x as u16 + padding,
                atlas_alloc.allocation.y as u16 + padding,
            ],
            padding,
            atlas_alloc_id: atlas_alloc.allocation.id,
        };
        self.slots[slot_idx] = Some(image_resource);

        Ok(image_id)
    }

    /// Deallocate an image from the cache, returning the image resource if it existed.
    pub fn deallocate(&mut self, id: ImageId) -> Option<ImageResource> {
        let index = id.as_u32() as usize;
        if let Some(image_resource) = self.slots.get_mut(index).and_then(Option::take) {
            // Deallocate from the appropriate atlas
            let padded_width = image_resource.width as u32 + u32::from(image_resource.padding) * 2;
            let padded_height =
                image_resource.height as u32 + u32::from(image_resource.padding) * 2;
            self.atlas_manager
                .deallocate(
                    image_resource.atlas_id,
                    image_resource.atlas_alloc_id,
                    padded_width,
                    padded_height,
                )
                .unwrap();
            self.free_idxs.push(index);
            Some(image_resource)
        } else {
            None
        }
    }

    /// Get access to the atlas manager.
    pub fn atlas_manager(&self) -> &MultiAtlasManager {
        &self.atlas_manager
    }

    /// Get the number of atlases.
    pub fn atlas_count(&self) -> usize {
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

        let id = cache.allocate(100, 100, 0).unwrap();

        assert_eq!(id.as_u32(), 0);
        let resource = cache.get(id).unwrap();
        assert_eq!(resource.width, 100);
        assert_eq!(resource.height, 100);
        // First image should be at origin
        assert_eq!(resource.offset, [0, 0]);
    }

    #[test]
    fn test_insert_single_image_with_padding() {
        let mut cache = ImageCache::new_with_config(AtlasConfig {
            atlas_size: (ATLAS_SIZE, ATLAS_SIZE),
            ..Default::default()
        });

        let id = cache.allocate(100, 100, 4).unwrap();

        assert_eq!(id.as_u32(), 0);
        let resource = cache.get(id).unwrap();
        assert_eq!(resource.width, 100);
        assert_eq!(resource.height, 100);
        assert_eq!(resource.padding, 4);
        // Offset should be shifted inward by padding.
        assert_eq!(resource.offset, [4, 4]);
    }

    #[test]
    fn test_insert_multiple_images() {
        let mut cache = ImageCache::new_with_config(AtlasConfig {
            atlas_size: (ATLAS_SIZE, ATLAS_SIZE),
            ..Default::default()
        });

        let id1 = cache.allocate(50, 50, 0).unwrap();
        let id2 = cache.allocate(75, 75, 0).unwrap();

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

        let id = cache.allocate(100, 100, 0).unwrap();
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
        let id1 = cache.allocate(50, 50, 0).unwrap();
        let id2 = cache.allocate(60, 60, 0).unwrap();
        let id3 = cache.allocate(70, 70, 0).unwrap();

        assert_eq!(id1.as_u32(), 0);
        assert_eq!(id2.as_u32(), 1);
        assert_eq!(id3.as_u32(), 2);

        // Unregister the middle one
        cache.deallocate(id2);
        assert!(cache.get(id2).is_none());

        // Register a new image - should reuse slot 1
        let id4 = cache.allocate(80, 80, 0).unwrap();
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
            .map(|i| cache.allocate(100 + i * 10, 100 + i * 10, 0).unwrap())
            .collect();

        // Unregister some in the middle
        cache.deallocate(ids[1]);
        cache.deallocate(ids[3]);

        // Register new images - should reuse the freed slots
        let new_id1 = cache.allocate(200, 200, 0).unwrap();
        let new_id2 = cache.allocate(300, 300, 0).unwrap();

        // Should have reused slots 3 and 1 (in reverse order due to stack behavior)
        assert_eq!(new_id1.as_u32(), 3);
        assert_eq!(new_id2.as_u32(), 1);
        assert_ne!(new_id1.as_u32(), new_id2.as_u32());
    }
}
