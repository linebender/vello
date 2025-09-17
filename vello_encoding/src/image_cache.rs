// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use guillotiere::{AtlasAllocator, size2};
use peniko::ImageData;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

const DEFAULT_ATLAS_SIZE: i32 = 1024;
const MAX_ATLAS_SIZE: i32 = 8192;

#[derive(Default)]
pub struct Images<'a> {
    pub width: u32,
    pub height: u32,
    pub images: &'a [(ImageData, u32, u32)],
}

pub(crate) struct ImageCache {
    atlas: AtlasAllocator,
    /// Map from image blob id to atlas location.
    map: HashMap<u64, (u32, u32)>,
    /// List of all allocated images with associated atlas location.
    images: Vec<(ImageData, u32, u32)>,
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageCache {
    pub(crate) fn new() -> Self {
        Self {
            atlas: AtlasAllocator::new(size2(DEFAULT_ATLAS_SIZE, DEFAULT_ATLAS_SIZE)),
            map: HashMap::default(),
            images: Vec::default(),
        }
    }

    pub(crate) fn images(&self) -> Images<'_> {
        Images {
            width: self.atlas.size().width as u32,
            height: self.atlas.size().height as u32,
            images: &self.images,
        }
    }

    pub(crate) fn bump_size(&mut self) -> bool {
        let new_size = self.atlas.size().width * 2;
        if new_size > MAX_ATLAS_SIZE {
            return false;
        }
        self.atlas = AtlasAllocator::new(size2(new_size, new_size));
        self.map.clear();
        self.images.clear();
        true
    }

    pub(crate) fn clear(&mut self) {
        self.atlas.clear();
        self.map.clear();
        self.images.clear();
    }

    pub(crate) fn get_or_insert(&mut self, image: &ImageData) -> Option<(u32, u32)> {
        match self.map.entry(image.data.id()) {
            Entry::Occupied(occupied) => Some(*occupied.get()),
            Entry::Vacant(vacant) => {
                let alloc = self
                    .atlas
                    .allocate(size2(image.width as _, image.height as _))?;
                let x = alloc.rectangle.min.x as u32;
                let y = alloc.rectangle.min.y as u32;
                self.images.push((image.clone(), x, y));
                Some(*vacant.insert((x, y)))
            }
        }
    }
}
