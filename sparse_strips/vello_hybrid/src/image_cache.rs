// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::vec::Vec;
use guillotiere::{AllocId, AtlasAllocator, size2};
use vello_common::paint::ImageId;

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
use web_sys::WebGlTexture;

const DEFAULT_ATLAS_SIZE: i32 = 1024;

/// WebGL texture wrapper that stores dimensions
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
#[derive(Debug)]
pub struct WebGlTextureWrapper {
    /// The WebGL texture
    pub texture: WebGlTexture,
    /// The width of the texture    
    pub width: u32,
    /// The height of the texture
    pub height: u32,
}

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
impl WebGlTextureWrapper {
    /// Create a new `WebGlTextureWrapper`
    pub fn new(texture: WebGlTexture, width: u32, height: u32) -> Self {
        Self {
            texture,
            width,
            height,
        }
    }
}

/// Trait for types that can be used as textures
pub trait TextureHandle {
    /// Get the width of the texture
    fn width(&self) -> u32;
    /// Get the height of the texture
    fn height(&self) -> u32;
}

#[cfg(feature = "wgpu")]
impl TextureHandle for wgpu::Texture {
    fn width(&self) -> u32 {
        self.width()
    }

    fn height(&self) -> u32 {
        self.height()
    }
}

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
impl TextureHandle for WebGlTextureWrapper {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }
}

/// Represents an image resource for rendering
#[derive(Debug)]
pub struct ImageResource<T: TextureHandle> {
    /// The ID of the image
    pub id: ImageId,
    /// The texture containing the image data
    pub texture: T,
    /// The offset of the image in the atlas
    pub offset: [u32; 2],
    /// The atlas allocation ID for deallocation
    pub(crate) atlas_alloc_id: AllocId,
}

impl<T: TextureHandle> ImageResource<T> {
    /// Get the width of the image
    pub fn width(&self) -> u32 {
        self.texture.width()
    }

    /// Get the height of the image
    pub fn height(&self) -> u32 {
        self.texture.height()
    }
}

/// Manages image resources for the renderer
#[allow(
    missing_debug_implementations,
    reason = "AtlasAllocator doesn't implement Debug"
)]
pub struct ImageCache<T: TextureHandle> {
    /// Atlas allocator for the images
    atlas: AtlasAllocator,
    /// Vector of optional image resources (None = free slot)
    slots: Vec<Option<ImageResource<T>>>,
    /// Stack of free indices for O(1) allocation/deallocation
    free_idxs: Vec<usize>,
}

impl<T: TextureHandle> Default for ImageCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: TextureHandle> ImageCache<T> {
    /// Create a new image cache
    pub fn new() -> Self {
        Self {
            atlas: AtlasAllocator::new(size2(DEFAULT_ATLAS_SIZE, DEFAULT_ATLAS_SIZE)),
            slots: Vec::new(),
            free_idxs: Vec::new(),
        }
    }

    /// Resize the image cache
    pub fn resize(&mut self, width: u32, height: u32) {
        self.atlas = AtlasAllocator::new(size2(width as i32, height as i32));
        self.slots.clear();
        self.free_idxs.clear();
    }

    /// Get an image resource by its Id
    pub fn get(&self, id: ImageId) -> Option<&ImageResource<T>> {
        let index = id.0 as usize;
        if index < self.slots.len() {
            self.slots[index].as_ref()
        } else {
            None
        }
    }

    /// Insert an image into the cache
    pub fn insert(&mut self, texture: T) -> ImageId {
        let alloc = self
            .atlas
            .allocate(size2(texture.width() as i32, texture.height() as i32))
            .expect("Failed to allocate texture");
        let atlas_alloc_id = alloc.id;
        let x = alloc.rectangle.min.x as u32;
        let y = alloc.rectangle.min.y as u32;

        // Try to reuse a free slot first
        let index = if let Some(free_index) = self.free_idxs.pop() {
            free_index
        } else {
            // No free slots, append to vector
            let index = self.slots.len();
            // Placeholder, will be replaced
            self.slots.push(None);
            index
        };

        #[expect(
            clippy::cast_possible_truncation,
            reason = "u32 is enough for the index"
        )]
        let image_id = ImageId(index as u32);
        let image_resource = ImageResource {
            id: image_id,
            texture,
            offset: [x, y],
            atlas_alloc_id,
        };

        self.slots[index] = Some(image_resource);
        image_id
    }

    /// Remove an image from the cache, returning true if it existed
    pub fn remove(&mut self, id: ImageId) -> bool {
        let index = id.0 as usize;
        if index < self.slots.len() {
            if let Some(image_resource) = &self.slots[index] {
                // Deallocate from the atlas using the stored allocation ID
                self.atlas.deallocate(image_resource.atlas_alloc_id);

                // Mark slot as free and add to free list
                self.slots[index] = None;
                self.free_idxs.push(index);
                return true;
            }
        }
        false
    }

    /// Clear all images from the cache
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_idxs.clear();
        self.atlas.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock texture for testing
    #[derive(Debug, Clone)]
    struct MockTexture {
        width: u32,
        height: u32,
    }

    impl MockTexture {
        fn new(width: u32, height: u32) -> Self {
            Self { width, height }
        }
    }

    impl TextureHandle for MockTexture {
        fn width(&self) -> u32 {
            self.width
        }

        fn height(&self) -> u32 {
            self.height
        }
    }

    #[test]
    fn test_insert_single_image() {
        let mut cache = ImageCache::new();
        let texture = MockTexture::new(100, 100);

        let id = cache.insert(texture.clone());

        assert_eq!(id.0, 0);
        let resource = cache.get(id).unwrap();
        assert_eq!(resource.id, id);
        assert_eq!(resource.width(), 100);
        assert_eq!(resource.height(), 100);
        assert_eq!(resource.offset, [0, 0]); // First image should be at origin
    }

    #[test]
    fn test_insert_multiple_images() {
        let mut cache = ImageCache::new();

        let id1 = cache.insert(MockTexture::new(50, 50));
        let id2 = cache.insert(MockTexture::new(75, 75));

        assert_eq!(id1.0, 0);
        assert_eq!(id2.0, 1);

        let resource1 = cache.get(id1).unwrap();
        let resource2 = cache.get(id2).unwrap();

        assert_eq!(resource1.width(), 50);
        assert_eq!(resource2.width(), 75);

        // Second image should be placed adjacent to first
        assert_ne!(resource1.offset, resource2.offset);
    }

    #[test]
    fn test_get_nonexistent_image() {
        let cache: ImageCache<MockTexture> = ImageCache::new();

        assert!(cache.get(ImageId(0)).is_none());
        assert!(cache.get(ImageId(999)).is_none());
    }

    #[test]
    fn test_remove_image() {
        let mut cache = ImageCache::new();
        let texture = MockTexture::new(100, 100);

        let id = cache.insert(texture);
        assert!(cache.get(id).is_some());

        cache.remove(id);
        assert!(cache.get(id).is_none());
    }

    #[test]
    fn test_remove_nonexistent_image() {
        let mut cache: ImageCache<MockTexture> = ImageCache::new();

        // Should not panic when unregistering non-existent image
        cache.remove(ImageId(0));
        cache.remove(ImageId(999));
    }

    #[test]
    fn test_slot_reuse_after_remove() {
        let mut cache = ImageCache::new();

        // Register three images
        let id1 = cache.insert(MockTexture::new(50, 50));
        let id2 = cache.insert(MockTexture::new(60, 60));
        let id3 = cache.insert(MockTexture::new(70, 70));

        assert_eq!(id1.0, 0);
        assert_eq!(id2.0, 1);
        assert_eq!(id3.0, 2);

        // Unregister the middle one
        cache.remove(id2);
        assert!(cache.get(id2).is_none());

        // Register a new image - should reuse slot 1
        let id4 = cache.insert(MockTexture::new(80, 80));
        // Reused slot 1
        assert_eq!(id4.0, 1);

        // Verify other images are still there
        assert!(cache.get(id1).is_some());
        assert!(cache.get(id3).is_some());
        assert!(cache.get(id4).is_some());
        assert_eq!(cache.get(id4).unwrap().width(), 80);
    }

    #[test]
    fn test_multiple_remove_and_reuse() {
        let mut cache = ImageCache::new();

        // Register several images
        let ids: Vec<_> = (0..5)
            .map(|i| cache.insert(MockTexture::new(100 + i * 10, 100 + i * 10)))
            .collect();

        // Unregister some in the middle
        cache.remove(ids[1]);
        cache.remove(ids[3]);

        // Register new images - should reuse the freed slots
        let new_id1 = cache.insert(MockTexture::new(200, 200));
        let new_id2 = cache.insert(MockTexture::new(300, 300));

        // Should have reused slots 3 and 1 (in reverse order due to stack behavior)
        assert!(new_id1.0 == 3);
        assert!(new_id2.0 == 1);
        assert_ne!(new_id1.0, new_id2.0);
    }

    #[test]
    fn test_resize_clears_cache() {
        let mut cache = ImageCache::new();
        let id = cache.insert(MockTexture::new(100, 100));
        assert!(cache.get(id).is_some());
        cache.resize(2048, 2048);
        // After resize, all images should be cleared
        assert!(cache.get(id).is_none());
    }
}
