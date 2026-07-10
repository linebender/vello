// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Atlas allocation for scheduled layer and scratch texture regions.

use super::{IntermediateTextureSizes, TextureRegion, TextureTarget};
use crate::filter::FILTER_ATLAS_PADDING;
use crate::util::{Int16Size, Int32Size};
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::{AllocId, Atlas, AtlasId};

pub(super) trait Allocator {
    type Request: Copy;
    type Allocation: Copy;

    fn allocate(&mut self, request: Self::Request) -> Option<Self::Allocation>;

    fn release(&mut self, allocation: Self::Allocation);
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Allocation<T> {
    pub(super) allocation: T,
    pub(super) round_idx: usize,
}

#[derive(Debug)]
pub(super) struct Atlases {
    layer_atlases: [Atlas; 2],
    scratch_atlases: [Atlas; 2],
}

#[derive(Debug, Clone, Copy)]
enum AtlasKind {
    Layer,
    Scratch,
}

impl Atlases {
    pub(super) fn new(texture_sizes: IntermediateTextureSizes) -> Self {
        let new_atlas = |target, id| {
            let size = texture_sizes.size(target);
            Atlas::new(
                AtlasId::new(id),
                u32::from(size.width()),
                u32::from(size.height()),
            )
        };

        Self {
            layer_atlases: [
                new_atlas(TextureTarget::Layer0, 0),
                new_atlas(TextureTarget::Layer1, 1),
            ],
            scratch_atlases: [
                new_atlas(TextureTarget::Scratch0, 0),
                new_atlas(TextureTarget::Scratch1, 1),
            ],
        }
    }

    fn allocate_scratch(
        &mut self,
        scratch_index: usize,
        request: LayerAllocationRequest,
        allocation_size: Int32Size,
    ) -> Option<ScratchAllocation> {
        let allocation = self.scratch_atlases[scratch_index]
            .allocate(allocation_size.width(), allocation_size.height())?;

        Some(ScratchAllocation {
            texture: request.allocated_texture_region(scratch_index, (allocation.x, allocation.y)),
            alloc_id: allocation.id,
        })
    }

    fn release_texture(
        &mut self,
        atlas_kind: AtlasKind,
        alloc_id: AllocId,
        texture: AllocatedTextureRegion,
    ) {
        let atlas = match atlas_kind {
            AtlasKind::Layer => &mut self.layer_atlases[texture.region.texture_index],
            AtlasKind::Scratch => &mut self.scratch_atlases[texture.region.texture_index],
        };
        let allocation_size = texture.allocation_size();
        atlas.deallocate(alloc_id, allocation_size.width(), allocation_size.height());
    }
}

impl Allocator for Atlases {
    type Request = LayerAllocationRequest;
    type Allocation = LayerAllocation;

    fn allocate(&mut self, request: Self::Request) -> Option<Self::Allocation> {
        assert!(
            request.scratch_count <= 2,
            "scratch count must be at most 2"
        );

        let allocation_size = request.allocation_size();
        let layer_allocation = self.layer_atlases[request.texture_index]
            .allocate(allocation_size.width(), allocation_size.height())?;
        let layer_texture = request.allocated_texture_region(
            request.texture_index,
            (layer_allocation.x, layer_allocation.y),
        );
        let mut scratch_allocations: [Option<ScratchAllocation>; 2] = [None, None];

        if request.scratch_count > 0 {
            let Some(scratch) = self.allocate_scratch(0, request, allocation_size) else {
                self.release_texture(AtlasKind::Layer, layer_allocation.id, layer_texture);
                return None;
            };
            scratch_allocations[0] = Some(scratch);
        }

        if request.scratch_count > 1 {
            let Some(scratch) = self.allocate_scratch(1, request, allocation_size) else {
                let scratch = scratch_allocations[0].expect("scratch 0 must be allocated");
                self.release_texture(AtlasKind::Scratch, scratch.alloc_id, scratch.texture);
                self.release_texture(AtlasKind::Layer, layer_allocation.id, layer_texture);
                return None;
            };
            scratch_allocations[1] = Some(scratch);
        }

        Some(LayerAllocation {
            filter: (request.scratch_count > 0).then_some(scratch_allocations),
            alloc_id: layer_allocation.id,
            texture: layer_texture,
        })
    }

    fn release(&mut self, allocation: Self::Allocation) {
        self.release_texture(AtlasKind::Layer, allocation.alloc_id, allocation.texture);
        if let Some(filter) = allocation.filter {
            for scratch in filter.into_iter().flatten() {
                self.release_texture(AtlasKind::Scratch, scratch.alloc_id, scratch.texture);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerAllocationRequest {
    texture_index: usize,
    size: Int16Size,
    padding: u16,
    scratch_count: usize,
}

impl LayerAllocationRequest {
    pub(super) fn new(texture_index: usize, size: Int16Size, scratch_count: usize) -> Self {
        let padding = if scratch_count > 0 {
            FILTER_ATLAS_PADDING
        } else {
            0
        };

        Self {
            texture_index,
            size,
            padding,
            scratch_count,
        }
    }

    pub(super) fn fits_textures(self, texture_sizes: IntermediateTextureSizes) -> bool {
        let fits = |target| {
            let size = texture_sizes.size(target);
            self.allocation_width() <= size.width() && self.allocation_height() <= size.height()
        };

        fits(TextureTarget::layer(self.texture_index))
            && (0..self.scratch_count).all(|index| fits(TextureTarget::scratch(index)))
    }

    fn allocation_width(self) -> u16 {
        self.size.width() + self.padding * 2
    }

    fn allocation_height(self) -> u16 {
        self.size.height() + self.padding * 2
    }

    fn allocation_size(self) -> Int32Size {
        Int16Size::new(self.allocation_width(), self.allocation_height()).into()
    }

    fn allocated_texture_region(
        self,
        texture_index: usize,
        allocation_origin: (u32, u32),
    ) -> AllocatedTextureRegion {
        let padding = u32::from(self.padding);
        let x = allocation_origin.0 + padding;
        let y = allocation_origin.1 + padding;
        AllocatedTextureRegion::new(
            TextureRegion {
                texture_index,
                rect: RectU16::new(
                    Self::atlas_coord(x),
                    Self::atlas_coord(y),
                    Self::atlas_coord(x + u32::from(self.size.width())),
                    Self::atlas_coord(y + u32::from(self.size.height())),
                ),
            },
            self.padding,
        )
    }

    fn atlas_coord(value: u32) -> u16 {
        u16::try_from(value).expect("atlas coordinate must fit into u16")
    }
}

/// A layer texture region plus the allocator handle needed to release it.
#[derive(Debug, Clone, Copy)]
pub(super) struct LayerAllocation {
    pub(super) filter: Option<FilterAllocation>,
    alloc_id: AllocId,
    pub(super) texture: AllocatedTextureRegion,
}

type FilterAllocation = [Option<ScratchAllocation>; 2];

#[derive(Debug, Clone, Copy)]
pub(super) struct ScratchAllocation {
    pub(super) texture: AllocatedTextureRegion,
    alloc_id: AllocId,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AllocatedTextureRegion {
    pub(super) region: TextureRegion,
    padding: u16,
}

impl AllocatedTextureRegion {
    fn new(region: TextureRegion, padding: u16) -> Self {
        Self { region, padding }
    }

    pub(super) fn clear_region(self) -> TextureRegion {
        TextureRegion {
            texture_index: self.region.texture_index,
            rect: self.region.rect.expand(RectU16::new(
                self.padding,
                self.padding,
                self.padding,
                self.padding,
            )),
        }
    }

    fn allocation_size(self) -> Int32Size {
        Int32Size([
            u32::from(self.region.rect.width()) + u32::from(self.padding) * 2,
            u32::from(self.region.rect.height()) + u32::from(self.padding) * 2,
        ])
    }
}
