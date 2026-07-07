// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Atlas allocation for scheduled layer and scratch texture regions.

use super::TextureRegion;
use super::cursor::Allocator;
use crate::filter::FILTER_ATLAS_PADDING;
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::{AllocId, Atlas, AtlasId};

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
    pub(super) fn new(texture_size: (u32, u32)) -> Self {
        Self {
            layer_atlases: [
                Atlas::new(AtlasId::new(0), texture_size.0, texture_size.1),
                Atlas::new(AtlasId::new(1), texture_size.0, texture_size.1),
            ],
            scratch_atlases: [
                Atlas::new(AtlasId::new(0), texture_size.0, texture_size.1),
                Atlas::new(AtlasId::new(1), texture_size.0, texture_size.1),
            ],
        }
    }

    fn allocate_scratch(
        &mut self,
        scratch_index: usize,
        request: LayerAllocationRequest,
        padded_allocation_width: u32,
        padded_allocation_height: u32,
    ) -> Option<ScratchAllocation> {
        let padding = u32::from(request.padding);
        let allocation = self.scratch_atlases[scratch_index]
            .allocate(padded_allocation_width, padded_allocation_height)?;

        Some(ScratchAllocation {
            texture: AllocatedTextureRegion::new(
                TextureRegion {
                    texture_index: scratch_index,
                    rect: RectU16::new(
                        atlas_coord(allocation.x + padding),
                        atlas_coord(allocation.y + padding),
                        atlas_coord(allocation.x + padding + u32::from(request.width)),
                        atlas_coord(allocation.y + padding + u32::from(request.height)),
                    ),
                },
                request.padding,
            ),
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
        let (allocation_width, allocation_height) = texture.allocation_size();
        atlas.deallocate(alloc_id, allocation_width, allocation_height);
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

        let padding = u32::from(request.padding);
        let width = u32::from(request.allocation_width());
        let height = u32::from(request.allocation_height());
        let layer_allocation = self.layer_atlases[request.texture_index].allocate(width, height)?;
        let layer_texture = AllocatedTextureRegion::new(
            TextureRegion {
                texture_index: request.texture_index,
                rect: RectU16::new(
                    atlas_coord(layer_allocation.x + padding),
                    atlas_coord(layer_allocation.y + padding),
                    atlas_coord(layer_allocation.x + padding + u32::from(request.width)),
                    atlas_coord(layer_allocation.y + padding + u32::from(request.height)),
                ),
            },
            request.padding,
        );
        let mut scratch_allocations: [Option<ScratchAllocation>; 2] = [None, None];

        if request.scratch_count > 0 {
            let Some(scratch) = self.allocate_scratch(0, request, width, height) else {
                self.release_texture(AtlasKind::Layer, layer_allocation.id, layer_texture);
                return None;
            };
            scratch_allocations[0] = Some(scratch);
        }

        if request.scratch_count > 1 {
            let Some(scratch) = self.allocate_scratch(1, request, width, height) else {
                let scratch = scratch_allocations[0].expect("scratch 0 must be allocated");
                self.release_texture(AtlasKind::Scratch, scratch.alloc_id, scratch.texture);
                self.release_texture(AtlasKind::Layer, layer_allocation.id, layer_texture);
                return None;
            };
            scratch_allocations[1] = Some(scratch);
        }

        Some(LayerAllocation {
            filter: (request.scratch_count > 0).then_some(scratch_allocations),
            round_idx: 0,
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
    width: u16,
    height: u16,
    padding: u16,
    scratch_count: usize,
}

impl LayerAllocationRequest {
    pub(super) fn new(texture_index: usize, size: (u16, u16), scratch_count: usize) -> Self {
        let padding = if scratch_count > 0 {
            FILTER_ATLAS_PADDING
        } else {
            0
        };

        Self {
            texture_index,
            width: size.0,
            height: size.1,
            padding,
            scratch_count,
        }
    }

    pub(super) fn fits_texture(self, layer_texture_size: (u32, u32)) -> bool {
        u32::from(self.allocation_width()) <= layer_texture_size.0
            && u32::from(self.allocation_height()) <= layer_texture_size.1
    }

    fn allocation_width(self) -> u16 {
        self.width + self.padding * 2
    }

    fn allocation_height(self) -> u16 {
        self.height + self.padding * 2
    }
}

/// A layer texture region plus the allocator handle needed to release it.
#[derive(Debug, Clone, Copy)]
pub(super) struct LayerAllocation {
    pub(super) filter: Option<FilterAllocation>,
    pub(super) round_idx: usize,
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

    fn allocation_size(self) -> (u32, u32) {
        (
            u32::from(self.region.rect.width()) + u32::from(self.padding) * 2,
            u32::from(self.region.rect.height()) + u32::from(self.padding) * 2,
        )
    }
}

fn atlas_coord(value: u32) -> u16 {
    u16::try_from(value).expect("atlas coordinate must fit into u16")
}
