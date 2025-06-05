extern crate std;
extern crate wgpu;

use std::collections::HashMap;

use guillotiere::{AtlasAllocator, size2};
use vello_common::paint::ImageId;
use wgpu::Texture;

const DEFAULT_ATLAS_SIZE: i32 = 1024;

/// Represents an image resource for rendering
#[derive(Debug)]
pub struct ImageResource {
    /// The ID of the image
    pub id: ImageId,
    /// The texture containing the image data
    pub texture: Texture,
    /// The offset of the image in the atlas
    pub offset: [u32; 2],
}

impl ImageResource {
    pub fn width(&self) -> u32 {
        self.texture.width()
    }
    pub fn height(&self) -> u32 {
        self.texture.height()
    }
}

/// Manages image resources for the renderer
#[allow(
    missing_debug_implementations,
    reason = "AtlasAllocator doesn't implement Debug"
)]
pub struct ImageCache {
    /// Atlas allocator for the images
    atlas: AtlasAllocator,
    /// Map of image IDs to resources
    pub images: HashMap<ImageId, ImageResource>,
}

impl ImageCache {
    /// Create a new image cache
    pub fn new() -> Self {
        Self {
            atlas: AtlasAllocator::new(size2(DEFAULT_ATLAS_SIZE, DEFAULT_ATLAS_SIZE)),
            images: HashMap::new(),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.atlas = AtlasAllocator::new(size2(width as i32, height as i32));
        self.images.clear();
    }

    pub fn register_image(&mut self, texture: Texture) -> &ImageResource {
        let alloc = self
            .atlas
            .allocate(size2(texture.width() as i32, texture.height() as i32))
            .expect("Failed to allocate texture");
        let x = alloc.rectangle.min.x as u32;
        let y = alloc.rectangle.min.y as u32;
        let id = ImageId(self.images.len() as u32);
        let image_resource = ImageResource {
            id,
            texture,
            offset: [x, y],
        };
        self.images.insert(id, image_resource);
        &self.images.get(&id).unwrap()
    }

    pub fn get_image(&self, id: ImageId) -> Option<&ImageResource> {
        self.images.get(&id)
    }
}
