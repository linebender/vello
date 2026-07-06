// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Atlas allocation for scheduled layer and scratch texture regions.

use super::timeline::ResourceAllocator;
use super::{LayerTextureRegion, ScratchRegion};
use crate::filter::{FILTER_ATLAS_PADDING, GpuFilterData};
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::{AllocId, Atlas, AtlasId};

#[derive(Debug)]
pub(super) struct LayerAtlasResource {
    atlases: [Atlas; 2],
    scratch_atlases: [Atlas; 2],
}

impl LayerAtlasResource {
    pub(super) fn new(layer_texture_size: (u32, u32)) -> Self {
        Self {
            atlases: [
                Atlas::new(AtlasId::new(0), layer_texture_size.0, layer_texture_size.1),
                Atlas::new(AtlasId::new(1), layer_texture_size.0, layer_texture_size.1),
            ],
            scratch_atlases: [
                Atlas::new(AtlasId::new(0), layer_texture_size.0, layer_texture_size.1),
                Atlas::new(AtlasId::new(1), layer_texture_size.0, layer_texture_size.1),
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
    allocation_width: u32,
    allocation_height: u32,
    filter: Option<FilterAllocationRequest>,
}

impl LayerAllocationRequest {
    pub(super) fn new(
        texture_index: usize,
        bbox: RectU16,
        filter: Option<FilterAllocationRequest>,
    ) -> Self {
        let width = bbox.width();
        let height = bbox.height();
        let padding = if filter.is_some() {
            FILTER_ATLAS_PADDING
        } else {
            0
        };
        let allocation_width = u32::from(width).saturating_add(u32::from(padding) * 2);
        let allocation_height = u32::from(height).saturating_add(u32::from(padding) * 2);

        Self {
            texture_index,
            bbox,
            width,
            height,
            padding,
            allocation_width,
            allocation_height,
            filter,
        }
    }

    pub(super) fn fits_texture(self, layer_texture_size: (u32, u32)) -> bool {
        self.allocation_width <= layer_texture_size.0
            && self.allocation_height <= layer_texture_size.1
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FilterAllocationRequest {
    pub(super) scratch_count: usize,
    pub(super) filter_data_offset: u32,
    pub(super) gpu_filter: GpuFilterData,
}

impl ResourceAllocator for LayerAtlasResource {
    type Request = LayerAllocationRequest;
    type Allocation = LayerAllocation;

    fn allocate(&mut self, request: Self::Request) -> Option<Self::Allocation> {
        let padding = u32::from(request.padding);
        let allocation = self.atlases[request.texture_index]
            .allocate(request.allocation_width, request.allocation_height)?;
        let mut scratch_allocations: [Option<ScratchAllocation>; 2] = [None, None];

        if let Some(filter) = request.filter {
            for (scratch_index, scratch) in scratch_allocations
                .iter_mut()
                .enumerate()
                .take(filter.scratch_count)
            {
                let Some(allocation) = self.scratch_atlases[scratch_index]
                    .allocate(request.allocation_width, request.allocation_height)
                else {
                    for (allocated_index, allocated_scratch) in
                        scratch_allocations.iter().enumerate()
                    {
                        if let Some(allocated_scratch) = allocated_scratch {
                            self.scratch_atlases[allocated_index].deallocate(
                                allocated_scratch.alloc_id,
                                allocated_scratch.allocation_width,
                                allocated_scratch.allocation_height,
                            );
                        }
                    }
                    self.atlases[request.texture_index].deallocate(
                        allocation.id,
                        request.allocation_width,
                        request.allocation_height,
                    );
                    return None;
                };
                *scratch = Some(ScratchAllocation {
                    region: ScratchRegion {
                        texture_index: scratch_index,
                        rect: RectU16::new(
                            atlas_coord(allocation.x + padding),
                            atlas_coord(allocation.y + padding),
                            atlas_coord(allocation.x + padding + u32::from(request.width)),
                            atlas_coord(allocation.y + padding + u32::from(request.height)),
                        ),
                    },
                    clear_region: ScratchRegion {
                        texture_index: scratch_index,
                        rect: RectU16::new(
                            atlas_coord(allocation.x),
                            atlas_coord(allocation.y),
                            atlas_coord(allocation.x + request.allocation_width),
                            atlas_coord(allocation.y + request.allocation_height),
                        ),
                    },
                    alloc_id: allocation.id,
                    allocation_width: request.allocation_width,
                    allocation_height: request.allocation_height,
                });
            }
        }

        Some(LayerAllocation {
            region: LayerTextureRegion {
                texture_index: request.texture_index,
                x: atlas_coord(allocation.x + padding),
                y: atlas_coord(allocation.y + padding),
                width: request.width,
                height: request.height,
                scene_bbox: request.bbox,
            },
            clear_region: LayerTextureRegion {
                texture_index: request.texture_index,
                x: atlas_coord(allocation.x),
                y: atlas_coord(allocation.y),
                width: atlas_coord(request.allocation_width),
                height: atlas_coord(request.allocation_height),
                scene_bbox: request.bbox,
            },
            filter: request.filter.map(|filter| FilterAllocation {
                scratches: scratch_allocations,
                filter_data_offset: filter.filter_data_offset,
                gpu_filter: filter.gpu_filter,
            }),
            round_idx: 0,
            alloc_id: allocation.id,
            allocation_width: request.allocation_width,
            allocation_height: request.allocation_height,
        })
    }

    fn release(&mut self, allocation: Self::Allocation) {
        self.atlases[allocation.region.texture_index].deallocate(
            allocation.alloc_id,
            allocation.allocation_width,
            allocation.allocation_height,
        );
        if let Some(filter) = allocation.filter {
            for scratch in filter.scratches.into_iter().flatten() {
                self.scratch_atlases[scratch.region.texture_index].deallocate(
                    scratch.alloc_id,
                    scratch.allocation_width,
                    scratch.allocation_height,
                );
            }
        }
    }
}

/// A layer texture region plus the allocator handle needed to release it.
#[derive(Debug, Clone, Copy)]
pub(super) struct LayerAllocation {
    pub(super) region: LayerTextureRegion,
    pub(super) clear_region: LayerTextureRegion,
    pub(super) filter: Option<FilterAllocation>,
    pub(super) round_idx: usize,
    alloc_id: AllocId,
    allocation_width: u32,
    allocation_height: u32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FilterAllocation {
    pub(super) scratches: [Option<ScratchAllocation>; 2],
    pub(super) filter_data_offset: u32,
    pub(super) gpu_filter: GpuFilterData,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ScratchAllocation {
    pub(super) region: ScratchRegion,
    pub(super) clear_region: ScratchRegion,
    alloc_id: AllocId,
    allocation_width: u32,
    allocation_height: u32,
}

fn atlas_coord(value: u32) -> u16 {
    u16::try_from(value).expect("atlas coordinate must fit into u16")
}
