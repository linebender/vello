// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::multi_atlas::{AtlasConfig, AtlasError, AtlasId, MultiAtlasManager};
use alloc::sync::Arc;
use alloc::vec::Vec;
use guillotiere::AllocId;
use hashbrown::HashMap;
use vello_common::paint::ImageId;
use vello_common::pixmap::Pixmap;

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

/// An image that has been allocated in the [`ImageCache`] but not yet uploaded to the GPU.
///
/// After [`ImageCache::allocate`], the atlas slot and offset are reserved,
/// but the pixel data hasn't been written yet. The renderer should iterate
/// pending uploads and write each pixmap to the atlas before rendering.
#[derive(Debug)]
pub(crate) struct PendingImageUpload {
    /// The image ID (use [`ImageCache::get`] with this id for `atlas_id` + offset).
    pub image_id: ImageId,
    /// The pixel data to upload.
    pub pixmap: Arc<Pixmap>,
}

/// A registered pixmap entry with its last-used frame serial.
#[derive(Debug, Clone, Copy)]
struct PixmapEntry {
    image_id: ImageId,
    /// Frame serial when this entry was last referenced.
    serial: u32,
}

/// Configuration for [`PixmapRegister`] eviction behaviour.
#[derive(Clone, Debug)]
pub(crate) struct PixmapRegisterConfig {
    /// Maximum age (in frames) before an unused entry is evicted.
    pub max_entry_age: u32,
    /// How often (in frames) to run the eviction pass.
    pub eviction_frequency: u32,
}

impl Default for PixmapRegisterConfig {
    fn default() -> Self {
        Self {
            max_entry_age: 64,
            eviction_frequency: 64,
        }
    }
}

/// Registry that maps `Arc<Pixmap>` pointer addresses to [`ImageId`]s.
///
/// This enables deduplication: the same `Arc<Pixmap>` (by pointer identity)
/// is only allocated and uploaded once. Entries persist across frames so
/// that a pixmap reused in consecutive frames doesn't get re-uploaded.
///
/// Stale entries (not referenced for
/// [`max_entry_age`](PixmapRegisterConfig::max_entry_age) frames) are
/// periodically evicted via [`maintain`](Self::maintain), which also
/// deallocates their atlas space from the [`ImageCache`].
#[derive(Debug)]
pub(crate) struct PixmapRegister {
    /// Maps `Arc<Pixmap>` pointer address to its entry.
    map: HashMap<usize, PixmapEntry>,
    /// Current frame counter, incremented each frame via [`tick`](Self::tick).
    serial: u32,
    /// Serial at which we last ran eviction.
    last_eviction_serial: u32,
    /// Eviction configuration.
    config: PixmapRegisterConfig,
}

impl Default for PixmapRegister {
    fn default() -> Self {
        Self::new(PixmapRegisterConfig::default())
    }
}

impl PixmapRegister {
    /// Create a new register with the given eviction configuration.
    pub(crate) fn new(config: PixmapRegisterConfig) -> Self {
        Self {
            map: HashMap::default(),
            serial: 0,
            last_eviction_serial: 0,
            config,
        }
    }

    /// Look up an `Arc<Pixmap>` by pointer identity.
    ///
    /// Returns the previously allocated [`ImageId`] if this exact `Arc` has
    /// been registered before, or `None` if it hasn't. The entry's serial
    /// is updated to mark it as recently used.
    pub(crate) fn get(&mut self, pixmap: &Arc<Pixmap>) -> Option<ImageId> {
        let ptr_key = Arc::as_ptr(pixmap) as usize;
        if let Some(entry) = self.map.get_mut(&ptr_key) {
            entry.serial = self.serial;
            Some(entry.image_id)
        } else {
            None
        }
    }

    /// Register a mapping from an `Arc<Pixmap>` pointer to an [`ImageId`].
    pub(crate) fn insert(&mut self, pixmap: &Arc<Pixmap>, image_id: ImageId) {
        let ptr_key = Arc::as_ptr(pixmap) as usize;
        self.map.insert(
            ptr_key,
            PixmapEntry {
                image_id,
                serial: self.serial,
            },
        );
    }

    /// Advance the frame counter and potentially evict old entries.
    ///
    /// Should be called once per frame after rendering. Entries that haven't
    /// been referenced for [`max_entry_age`](PixmapRegisterConfig::max_entry_age)
    /// frames are removed and their atlas allocations are freed.
    pub(crate) fn maintain(&mut self, image_cache: &mut ImageCache) {
        self.tick();

        let frames_since_eviction = self.serial.wrapping_sub(self.last_eviction_serial);
        if frames_since_eviction < self.config.eviction_frequency {
            return;
        }

        self.last_eviction_serial = self.serial;
        self.evict_old_entries(image_cache);
    }

    /// Advance the frame counter.
    fn tick(&mut self) {
        self.serial = self.serial.wrapping_add(1);
    }

    /// Evict entries that haven't been used recently.
    fn evict_old_entries(&mut self, image_cache: &mut ImageCache) {
        let serial = self.serial;
        let max_entry_age = self.config.max_entry_age;

        self.map.retain(|_, entry| {
            let age = serial.wrapping_sub(entry.serial);
            if age > max_entry_age {
                image_cache.deallocate(entry.image_id);
                false
            } else {
                true
            }
        });
    }
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
