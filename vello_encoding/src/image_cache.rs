// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use guillotiere::{size2, AtlasAllocator};
use peniko::Image;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

const ATLAS_SIZE: i32 = 2048;
const MAX_ATLAS_LAYERS: u32 = 255;

#[derive(Default)]
pub struct Images<'a> {
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub images: &'a [(Image, u32, u32, u32)],
}

pub(crate) struct ImageCache {
    atlas: AtlasAllocator,
    /// Map from image blob id to atlas location.
    map: HashMap<u64, (u32, u32, u32)>,
    /// List of all allocated images with associated atlas location.
    images: Vec<(Image, u32, u32, u32)>,
    /// The current layer we're resolving for
    layer: u32,
    /// The number of layers we use.
    layers: u32,
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageCache {
    pub(crate) fn new() -> Self {
        Self {
            atlas: AtlasAllocator::new(size2(ATLAS_SIZE, ATLAS_SIZE)),
            layer: 0,
            map: Default::default(),
            images: Default::default(),
            layers: 4,
        }
    }

    pub(crate) fn images(&self) -> Images<'_> {
        Images {
            width: self.atlas.size().width as u32,
            height: self.atlas.size().height as u32,
            images: &self.images,
            layers: self.layers,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.atlas.clear();
        self.map.clear();
        self.images.clear();
        self.layer = 0;
    }

    pub(crate) fn get_or_insert(&mut self, image: &Image) -> Option<(u32, u32, u32)> {
        match self.map.entry(image.data.id()) {
            Entry::Occupied(occupied) => Some(*occupied.get()),
            Entry::Vacant(vacant) => {
                if image.width > ATLAS_SIZE as u32 || image.height > ATLAS_SIZE as u32 {
                    // We currently cannot support images larger than 2048 in any axis.
                    // We should probably still support that, but I think the fallback
                    // might end up being a second "atlas"
                    // We choose not to re-size the atlas in that case, because it
                    // would add a large amount of unused data.
                    return None;
                }
                let alloc = self
                    .atlas
                    .allocate(size2(image.width as _, image.height as _));
                let alloc = match alloc {
                    Some(alloc) => alloc,
                    None => {
                        if self.layer >= MAX_ATLAS_LAYERS {
                            return None;
                        }
                        // We implement a greedy system for layers; if we ever get an image that won't fit.
                        self.layer += 1;
                        if self.layer >= self.layers {
                            self.layers = (self.layers * 2).min(MAX_ATLAS_LAYERS);
                            debug_assert!(self.layer < self.layers);
                        }
                        self.atlas.clear();
                        // This should never fail, as it's a fresh atlas
                        self.atlas
                            .allocate(size2(image.width as _, image.height as _))?
                    }
                };
                let x = alloc.rectangle.min.x as u32;
                let y = alloc.rectangle.min.y as u32;
                self.images.push((image.clone(), x, y, self.layer));
                Some(*vacant.insert((x, y, self.layer)))
            }
        }
    }
}
