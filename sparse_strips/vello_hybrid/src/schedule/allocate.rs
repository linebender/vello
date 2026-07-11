// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Atlas allocation for scheduled layer and scratch texture regions.

use crate::filter::FILTER_ATLAS_PADDING;
use crate::target::{IntermediateTextureSizes, TextureIndex, TextureRegion, TextureTarget};
use crate::util::Int16Size;
use vello_common::geometry::RectU16;
use vello_common::multi_atlas::{AllocId, Atlas, AtlasId};
use vello_common::record::RecordedLayerKind;

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
        request: LayerAllocationRequest,
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
    type Request = LayerAllocationRequest;
    type Allocation = LayerAllocations;

    fn allocate(&mut self, request: Self::Request) -> Option<Self::Allocation> {
        // First allocate the main region for the layer in the layer atlas.
        let main_allocation =
            self.allocate_region(TextureTarget::layer(request.texture_index), request)?;

        // Then, depending on how many regions in scratch textures are needed, allocate those.
        // If at least one allocation fails, we need to make sure to undo all other allocations
        // we have done so far.
        let mut scratch_allocations = [None, None];

        if request.scratch_count > 0 {
            let Some(texture) =
                self.allocate_region(TextureTarget::scratch(TextureIndex::Even), request)
            else {
                self.deallocate_region(AtlasKind::Layer, main_allocation);

                return None;
            };

            scratch_allocations[0] = Some(texture);
        }

        if request.scratch_count > 1 {
            let Some(texture) =
                self.allocate_region(TextureTarget::scratch(TextureIndex::Odd), request)
            else {
                let scratch_0 = scratch_allocations[0].expect("scratch 0 must be allocated");
                self.deallocate_region(AtlasKind::Scratch, scratch_0);
                self.deallocate_region(AtlasKind::Layer, main_allocation);

                return None;
            };

            scratch_allocations[1] = Some(texture);
        }

        Some(LayerAllocations {
            scratch_allocations,
            main_allocation,
        })
    }

    fn release(&mut self, allocation: Self::Allocation) {
        self.deallocate_region(AtlasKind::Layer, allocation.main_allocation);

        for scratch in allocation.scratch_allocations.into_iter().flatten() {
            self.deallocate_region(AtlasKind::Scratch, scratch);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerAllocationRequest {
    texture_index: TextureIndex,
    size: Int16Size,
    padding: u16,
    scratch_count: u8,
}

impl LayerAllocationRequest {
    pub(super) fn new(
        texture_index: TextureIndex,
        size: Int16Size,
        kind: &RecordedLayerKind,
        scratch_count: u8,
    ) -> Self {
        let padding = match kind {
            RecordedLayerKind::Regular => 0,
            RecordedLayerKind::Filter { .. } => FILTER_ATLAS_PADDING,
        };

        Self {
            texture_index,
            size,
            padding,
            scratch_count,
        }
    }
}

/// A layer texture region plus the allocator handle needed to release it.
#[derive(Debug, Clone, Copy)]
pub(super) struct LayerAllocations {
    pub(super) scratch_allocations: [Option<AllocatedTextureRegion>; 2],
    pub(super) main_allocation: AllocatedTextureRegion,
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
