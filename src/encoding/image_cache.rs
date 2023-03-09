// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

use guillotiere::{size2, AtlasAllocator};
use peniko::Image;
use std::collections::{hash_map::Entry, HashMap};

const DEFAULT_ATLAS_SIZE: i32 = 1024;
const MAX_ATLAS_SIZE: i32 = 8192;

pub struct Images<'a> {
    pub width: u32,
    pub height: u32,
    pub images: &'a [(Image, u32, u32)],
}

pub struct ImageCache {
    atlas: AtlasAllocator,
    /// Map from image blob id to atlas location.
    map: HashMap<u64, (u32, u32)>,
    /// List of all allocated images with associated atlas location.
    images: Vec<(Image, u32, u32)>,
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageCache {
    pub fn new() -> Self {
        Self {
            atlas: AtlasAllocator::new(size2(DEFAULT_ATLAS_SIZE, DEFAULT_ATLAS_SIZE)),
            map: Default::default(),
            images: Default::default(),
        }
    }

    pub fn images(&self) -> Images {
        Images {
            width: self.atlas.size().width as u32,
            height: self.atlas.size().height as u32,
            images: &self.images,
        }
    }

    pub fn bump_size(&mut self) -> bool {
        let new_size = self.atlas.size().width * 2;
        if new_size > MAX_ATLAS_SIZE {
            return false;
        }
        self.atlas = AtlasAllocator::new(size2(new_size, new_size));
        self.map.clear();
        self.images.clear();
        true
    }

    pub fn clear(&mut self) {
        self.atlas.clear();
        self.map.clear();
        self.images.clear();
    }

    pub fn get_or_insert(&mut self, image: &Image) -> Option<(u32, u32)> {
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
