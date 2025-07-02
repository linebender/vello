// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(dead_code, reason = "Clippy fails when --no-default-features")]

use alloc::vec::Vec;
use guillotiere::{AllocId, AtlasAllocator, size2};
use vello_common::paint::ImageId;

const DEFAULT_ATLAS_SIZE: u32 = 1024;

/// Represents an image resource for rendering
#[derive(Debug)]
pub(crate) struct ImageResource {
    /// The ID of the image
    pub(crate) id: ImageId,
    /// The width of the image
    pub(crate) width: u16,
    /// The height of the image
    pub(crate) height: u16,
    /// The offset of the image in the atlas
    pub(crate) offset: [u16; 2],
    /// The atlas allocation ID for deallocation
    atlas_alloc_id: AllocId,
}

/// Manages image resources for the renderer
pub(crate) struct ImageCache {
    /// Atlas allocator for the images
    atlas: AtlasAllocator,
    /// Vector of optional image resources (None = free slot)
    slots: Vec<Option<ImageResource>>,
    /// Stack of free indices for O(1) allocation/deallocation
    free_idxs: Vec<usize>,
}

impl core::fmt::Debug for ImageCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Count allocated and free rectangles in the atlas
        let mut allocated_count = 0;
        let mut free_count = 0;

        self.atlas.for_each_allocated_rectangle(|_id, _rect| {
            allocated_count += 1;
        });

        self.atlas.for_each_free_rectangle(|_rect| {
            free_count += 1;
        });

        f.debug_struct("ImageCache")
            .field("slots", &self.slots)
            .field("free_idxs", &self.free_idxs)
            .field("atlas_size", &self.atlas.size())
            .field("atlas_is_empty", &self.atlas.is_empty())
            .field("atlas_allocated_count", &allocated_count)
            .field("atlas_free_count", &free_count)
            .finish()
    }
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new(DEFAULT_ATLAS_SIZE, DEFAULT_ATLAS_SIZE)
    }
}

impl ImageCache {
    /// Create a new image cache
    pub(crate) fn new(width: u32, height: u32) -> Self {
        Self {
            atlas: AtlasAllocator::new(size2(width as i32, height as i32)),
            slots: Vec::new(),
            free_idxs: Vec::new(),
        }
    }

    /// Get an image resource by its Id
    pub(crate) fn get(&self, id: ImageId) -> Option<&ImageResource> {
        self.slots.get(id.as_u32() as usize)?.as_ref()
    }

    /// Allocate an image in the cache
    #[expect(
        clippy::cast_possible_truncation,
        reason = "u16 is enough for the offset and width/height"
    )]
    pub(crate) fn allocate(&mut self, width: u32, height: u32) -> ImageId {
        let alloc = self
            .atlas
            .allocate(size2(width as i32, height as i32))
            .expect("Failed to allocate texture");

        let slot_idx = self.free_idxs.pop().unwrap_or_else(|| {
            // No free slots, append to vector
            let index = self.slots.len();
            // Placeholder, will be replaced
            self.slots.push(None);
            index
        });

        let image_id = ImageId::new(slot_idx as u32);
        self.slots[slot_idx] = Some(ImageResource {
            id: image_id,
            width: width as u16,
            height: height as u16,
            offset: [alloc.rectangle.min.x as u16, alloc.rectangle.min.y as u16],
            atlas_alloc_id: alloc.id,
        });
        image_id
    }

    /// Deallocate an image from the cache, returning true if it existed
    pub(crate) fn deallocate(&mut self, id: ImageId) -> Option<ImageResource> {
        let index = id.as_u32() as usize;
        if let Some(image_resource) = self.slots.get_mut(index).and_then(Option::take) {
            // Deallocate from the atlas using the stored allocation ID
            self.atlas.deallocate(image_resource.atlas_alloc_id);
            self.free_idxs.push(index);
            Some(image_resource)
        } else {
            None
        }
    }

    /// Clear all images from the cache
    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.free_idxs.clear();
        self.atlas.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_single_image() {
        let mut cache = ImageCache::default();

        let id = cache.allocate(100, 100);

        assert_eq!(id.as_u32(), 0);
        let resource = cache.get(id).unwrap();
        assert_eq!(resource.id, id);
        assert_eq!(resource.width, 100);
        assert_eq!(resource.height, 100);
        assert_eq!(resource.offset, [0, 0]); // First image should be at origin
    }

    #[test]
    fn test_insert_multiple_images() {
        let mut cache = ImageCache::default();

        let id1 = cache.allocate(50, 50);
        let id2 = cache.allocate(75, 75);

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
        let cache: ImageCache = ImageCache::default();

        assert!(cache.get(ImageId::new(0)).is_none());
        assert!(cache.get(ImageId::new(999)).is_none());
    }

    #[test]
    fn test_remove_image() {
        let mut cache = ImageCache::default();

        let id = cache.allocate(100, 100);
        assert!(cache.get(id).is_some());

        cache.deallocate(id);
        assert!(cache.get(id).is_none());
    }

    #[test]
    fn test_remove_nonexistent_image() {
        let mut cache: ImageCache = ImageCache::default();

        // Should not panic when unregistering non-existent image
        cache.deallocate(ImageId::new(0));
        cache.deallocate(ImageId::new(999));
    }

    #[test]
    fn test_slot_reuse_after_remove() {
        let mut cache = ImageCache::default();

        // Register three images
        let id1 = cache.allocate(50, 50);
        let id2 = cache.allocate(60, 60);
        let id3 = cache.allocate(70, 70);

        assert_eq!(id1.as_u32(), 0);
        assert_eq!(id2.as_u32(), 1);
        assert_eq!(id3.as_u32(), 2);

        // Unregister the middle one
        cache.deallocate(id2);
        assert!(cache.get(id2).is_none());

        // Register a new image - should reuse slot 1
        let id4 = cache.allocate(80, 80);
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
        let mut cache = ImageCache::default();

        // Register several images
        let ids: Vec<_> = (0..5)
            .map(|i| cache.allocate(100 + i * 10, 100 + i * 10))
            .collect();

        // Unregister some in the middle
        cache.deallocate(ids[1]);
        cache.deallocate(ids[3]);

        // Register new images - should reuse the freed slots
        let new_id1 = cache.allocate(200, 200);
        let new_id2 = cache.allocate(300, 300);

        // Should have reused slots 3 and 1 (in reverse order due to stack behavior)
        assert!(new_id1.as_u32() == 3);
        assert!(new_id2.as_u32() == 1);
        assert_ne!(new_id1.as_u32(), new_id2.as_u32());
    }
}
