// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Atlas allocation for scheduled layer and scratch texture regions.

use super::OpenLayer;
use crate::filter::{FILTER_ATLAS_PADDING, PreparedGpuFilter};
use crate::target::{IntermediateTextureSizes, TextureIndex, TextureRegion, TextureTarget};
use crate::util::Int16Size;
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::{AllocId, Atlas, AtlasId};
use vello_common::record::RecordedLayerKind;

pub(super) trait Allocator {
    type Release: Copy;

    fn release(&mut self, allocation: Self::Release);
}

pub(super) trait AllocationRequest<R>: Copy {
    type Allocation: Copy;

    fn allocate(self, resource: &mut R) -> Option<Self::Allocation>;
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Allocation<T> {
    /// The underlying allocation.
    pub(super) allocation: T,
    /// The round starting from which this allocation is available.
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
                new_atlas(TextureTarget::layer(TextureIndex::Even), 0),
                new_atlas(TextureTarget::layer(TextureIndex::Odd), 1),
            ],
            scratch_atlases: [
                new_atlas(TextureTarget::scratch(TextureIndex::Even), 0),
                new_atlas(TextureTarget::scratch(TextureIndex::Odd), 1),
            ],
        }
    }

    fn allocate_region(
        &mut self,
        target: TextureTarget,
        request: RegionAllocationRequest,
    ) -> Option<AllocatedTextureRegion> {
        let padding = u32::from(request.padding);
        let full_padding = padding * 2;
        let width = request.size.width();
        let height = request.size.height();
        let texture_index = target.index();

        let atlas = match target {
            TextureTarget::Layer(texture_index) => {
                &mut self.layer_atlases[texture_index.get_index()]
            }
            TextureTarget::Scratch(texture_index) => {
                &mut self.scratch_atlases[texture_index.get_index()]
            }
        };
        let allocation = atlas.allocate(
            u32::from(width) + full_padding,
            u32::from(height) + full_padding,
        )?;
        let x = u16::try_from(allocation.x + padding).unwrap();
        let y = u16::try_from(allocation.y + padding).unwrap();
        let region = AllocatedTextureRegion::new(
            TextureRegion {
                texture_index,
                rect: RectU16::new(x, y, x + width, y + height),
            },
            request.padding,
            allocation.id,
        );

        Some(region)
    }

    fn deallocate_region(&mut self, atlas_kind: AtlasKind, texture: AllocatedTextureRegion) {
        let atlas = match atlas_kind {
            AtlasKind::Layer => &mut self.layer_atlases[texture.region.texture_index.get_index()],
            AtlasKind::Scratch => {
                &mut self.scratch_atlases[texture.region.texture_index.get_index()]
            }
        };

        let allocation_region = texture.allocation_region();

        atlas.deallocate(
            texture.alloc_id,
            u32::from(allocation_region.width()),
            u32::from(allocation_region.height()),
        );
    }
}

impl Allocator for Atlases {
    type Release = AtlasAllocation;

    fn release(&mut self, allocation: Self::Release) {
        match allocation {
            AtlasAllocation::Layer(texture) => {
                self.deallocate_region(AtlasKind::Layer, texture);
            }
            AtlasAllocation::Scratch(texture) => {
                self.deallocate_region(AtlasKind::Scratch, texture);
            }
        }
    }
}

impl AllocationRequest<Atlases> for LayerAllocationRequest {
    type Allocation = AllocatedTextureRegion;

    fn allocate(self, atlases: &mut Atlases) -> Option<Self::Allocation> {
        atlases.allocate_region(TextureTarget::layer(self.texture_index), self.region)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerAllocationRequest {
    texture_index: TextureIndex,
    region: RegionAllocationRequest,
}

impl LayerAllocationRequest {
    pub(super) fn new(layer: &OpenLayer<'_>) -> Self {
        let padding = match layer.kind {
            RecordedLayerKind::Regular => 0,
            // We not only need the padding for the scratch textures of a filter layer, but also
            // in the main layer atlas. The reason is that when applying the filter, we merge
            // the steps "copy into scratch texture + first filter pass".
            RecordedLayerKind::Filter { .. } => FILTER_ATLAS_PADDING,
        };

        let region = RegionAllocationRequest {
            size: Int16Size::new(layer.bbox.width(), layer.bbox.height()),
            padding,
        };

        Self {
            texture_index: layer.texture_index,
            region,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RegionAllocationRequest {
    size: Int16Size,
    padding: u16,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ScratchAllocationRequest {
    region: RegionAllocationRequest,
    count: u8,
}

impl ScratchAllocationRequest {
    pub(super) fn new(region: TextureRegion, filter: &PreparedGpuFilter) -> Self {
        Self {
            region: RegionAllocationRequest {
                size: Int16Size::new(region.rect.width(), region.rect.height()),
                padding: FILTER_ATLAS_PADDING,
            },
            count: filter.scratch_count(),
        }
    }
}

impl AllocationRequest<Atlases> for ScratchAllocationRequest {
    type Allocation = [Option<AllocatedTextureRegion>; 2];

    fn allocate(self, atlases: &mut Atlases) -> Option<Self::Allocation> {
        debug_assert!(self.count <= 2);

        let mut allocations = [None, None];

        for index in 0..usize::from(self.count) {
            let texture_index = TextureIndex::from_index(index);
            let Some(allocation) =
                atlases.allocate_region(TextureTarget::scratch(texture_index), self.region)
            else {
                // Both allocations must succeed, if the second one doesn't succeed we also need
                // to undo the first one.
                for allocation in allocations.into_iter().flatten() {
                    atlases.deallocate_region(AtlasKind::Scratch, allocation);
                }

                return None;
            };

            allocations[index] = Some(allocation);
        }

        Some(allocations)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum AtlasAllocation {
    Layer(AllocatedTextureRegion),
    Scratch(AllocatedTextureRegion),
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AllocatedTextureRegion {
    pub(super) region: TextureRegion,
    padding: u16,
    alloc_id: AllocId,
}

impl AllocatedTextureRegion {
    fn new(region: TextureRegion, padding: u16, alloc_id: AllocId) -> Self {
        Self {
            region,
            padding,
            alloc_id,
        }
    }

    pub(super) fn clear_region(self) -> TextureRegion {
        // The padding region isn't drawn into, so we also don't need to clear it.
        self.region
    }

    fn allocation_region(self) -> RectU16 {
        RectU16::new(
            self.region.rect.x0 - self.padding,
            self.region.rect.y0 - self.padding,
            self.region.rect.x1 + self.padding,
            self.region.rect.y1 + self.padding,
        )
    }
}
