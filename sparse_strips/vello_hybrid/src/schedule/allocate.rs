// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Atlas allocation for scheduled layer and scratch texture regions.

use super::timeline::ResourceAllocator;
use super::{LayerTextureRegion, TextureRegion};
use crate::filter::FILTER_ATLAS_PADDING;
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::{AllocId, Atlas, AtlasId};

#[derive(Debug)]
pub(super) struct Atlases {
    layer_atlases: [Atlas; 2],
    scratch_atlases: [Atlas; 2],
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
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerAllocationRequest {
    texture_index: usize,
    bbox: RectU16,
    width: u16,
    height: u16,
    padding: u16,
    scratch_count: usize,
}

impl LayerAllocationRequest {
    pub(super) fn new(texture_index: usize, bbox: RectU16, scratch_count: usize) -> Self {
        let width = bbox.width();
        let height = bbox.height();
        let padding = if scratch_count > 0 {
            FILTER_ATLAS_PADDING
        } else {
            0
        };

        Self {
            texture_index,
            bbox,
            width,
            height,
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

impl ResourceAllocator for Atlases {
    type Request = LayerAllocationRequest;
    type Allocation = LayerAllocation;

    fn allocate(&mut self, request: Self::Request) -> Option<Self::Allocation> {
        let padding = u32::from(request.padding);
        let allocation_width = u32::from(request.allocation_width());
        let allocation_height = u32::from(request.allocation_height());
        let allocation = self.layer_atlases[request.texture_index]
            .allocate(allocation_width, allocation_height)?;
        let mut scratch_allocations: [Option<ScratchAllocation>; 2] = [None, None];

        if request.scratch_count > 0 {
            for (scratch_index, scratch) in scratch_allocations
                .iter_mut()
                .enumerate()
                .take(request.scratch_count)
            {
                let Some(allocation) = self.scratch_atlases[scratch_index]
                    .allocate(allocation_width, allocation_height)
                else {
                    for (allocated_index, allocated_scratch) in
                        scratch_allocations.iter().enumerate()
                    {
                        if let Some(allocated_scratch) = allocated_scratch {
                            let (allocation_width, allocation_height) =
                                allocated_scratch.texture.allocation_size();
                            self.scratch_atlases[allocated_index].deallocate(
                                allocated_scratch.alloc_id,
                                allocation_width,
                                allocation_height,
                            );
                        }
                    }
                    self.layer_atlases[request.texture_index].deallocate(
                        allocation.id,
                        allocation_width,
                        allocation_height,
                    );
                    return None;
                };
                *scratch = Some(ScratchAllocation {
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
                });
            }
        }

        let texture = TextureRegion {
            texture_index: request.texture_index,
            rect: RectU16::new(
                atlas_coord(allocation.x + padding),
                atlas_coord(allocation.y + padding),
                atlas_coord(allocation.x + padding + u32::from(request.width)),
                atlas_coord(allocation.y + padding + u32::from(request.height)),
            ),
        };

        Some(LayerAllocation {
            region: LayerTextureRegion {
                texture,
                scene_bbox: request.bbox,
            },
            filter: (request.scratch_count > 0).then_some(scratch_allocations),
            round_idx: 0,
            alloc_id: allocation.id,
            texture: AllocatedTextureRegion::new(texture, request.padding),
        })
    }

    fn release(&mut self, allocation: Self::Allocation) {
        let (allocation_width, allocation_height) = allocation.texture.allocation_size();
        self.layer_atlases[allocation.region.texture.texture_index].deallocate(
            allocation.alloc_id,
            allocation_width,
            allocation_height,
        );
        if let Some(filter) = allocation.filter {
            for scratch in filter.into_iter().flatten() {
                let (allocation_width, allocation_height) = scratch.texture.allocation_size();
                self.scratch_atlases[scratch.texture.region.texture_index].deallocate(
                    scratch.alloc_id,
                    allocation_width,
                    allocation_height,
                );
            }
        }
    }
}

/// A layer texture region plus the allocator handle needed to release it.
#[derive(Debug, Clone, Copy)]
pub(super) struct LayerAllocation {
    pub(super) region: LayerTextureRegion,
    pub(super) filter: Option<FilterAllocation>,
    pub(super) round_idx: usize,
    alloc_id: AllocId,
    pub(super) texture: AllocatedTextureRegion,
}

pub(super) type FilterAllocation = [Option<ScratchAllocation>; 2];

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
