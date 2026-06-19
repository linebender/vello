// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Atlas allocation for scheduled layer and scratch texture regions.

use super::OpenLayer;
use crate::filter::{FILTER_ATLAS_PADDING, PreparedGpuFilter};
use crate::scene::LayersConfig;
use crate::target::{IntermediateTextureSizes, LayerTextureId, TextureParity, TextureRegion};
use alloc::vec::Vec;
use vello_common::geometry::{RectU16, SizeU16, SizeU32};
use vello_common::multi_atlas::{AllocId, Atlas, AtlasError, AtlasId};
use vello_common::record::RecordedLayerKind;

#[derive(Debug, Clone, Copy)]
pub(super) struct Allocation<T> {
    /// The underlying allocation.
    pub(super) allocation: T,
    /// The round starting from which this allocation is available.
    pub(super) round_idx: usize,
}

#[derive(Debug)]
pub(super) struct Atlases {
    layer_atlases: [Vec<Atlas>; 2],
    scratch_atlases: [Option<Atlas>; 2],
    remaining_textures: usize,
    texture_sizes: IntermediateTextureSizes,
}

impl Atlases {
    pub(super) fn new(texture_sizes: IntermediateTextureSizes, layer_config: LayersConfig) -> Self {
        Self {
            layer_atlases: core::array::from_fn(|_| Vec::new()),
            scratch_atlases: [None, None],
            remaining_textures: layer_config.max_textures.unwrap_or(usize::MAX),
            texture_sizes,
        }
    }

    pub(super) fn scratch_textures(&self) -> [bool; 2] {
        self.scratch_atlases.each_ref().map(Option::is_some)
    }

    pub(super) fn allocate_layer(
        &mut self,
        request: &LayerAllocationRequest,
    ) -> Option<AllocatedTextureRegion<LayerTextureId>> {
        let parity = request.texture_parity.get_parity();

        // It's unfortunate the runtime of this will grow linearly with each allocated page
        // for each allocation request, but since our scheduler is conservative it is unlikely we
        // will ever have more than 2-3 pages, except for scene graphs that have lots of nested
        // layers with multiple children.
        for page_index in 0..u16::try_from(self.layer_atlases[parity].len()).unwrap() {
            let id = LayerTextureId::new(request.texture_parity, page_index);
            let atlas = &mut self.layer_atlases[parity][usize::from(page_index)];

            if let Some(allocation) = atlas.allocate_region(id, request.region) {
                return Some(allocation);
            }
        }

        None
    }

    pub(super) fn allocate_scratch(
        &mut self,
        request: &ScratchAllocationRequest,
    ) -> Result<Option<[Option<AllocatedTextureRegion<TextureParity>>; 2]>, AtlasError> {
        let mut allocations: [Option<AllocatedTextureRegion<TextureParity>>; 2] = [None, None];

        for index in 0..usize::from(request.count) {
            let texture_parity = TextureParity::from_parity(index);
            self.ensure_scratch_texture(texture_parity)?;

            let atlas = self.scratch_atlases[texture_parity.get_parity()]
                .as_mut()
                .unwrap();
            let Some(allocation) = atlas.allocate_region(texture_parity, request.region) else {
                // Both allocations must succeed, if the second one doesn't succeed we also need
                // to undo the first one.
                for allocation in allocations.into_iter().flatten() {
                    let atlas = self.scratch_atlases[allocation.region.target.get_parity()]
                        .as_mut()
                        .unwrap();

                    atlas.deallocate_region(allocation);
                }

                return Ok(None);
            };

            allocations[index] = Some(allocation);
        }

        Ok(Some(allocations))
    }

    pub(super) fn deallocate(&mut self, allocation: AtlasAllocation) {
        match allocation {
            AtlasAllocation::Layer(texture) => {
                let id = texture.region.target;
                let atlas = &mut self.layer_atlases[id.texture_parity.get_parity()]
                    [usize::from(id.page_index)];

                atlas.deallocate_region(texture);
            }
            AtlasAllocation::Scratch(texture) => {
                let atlas = self.scratch_atlases[texture.region.target.get_parity()]
                    .as_mut()
                    .unwrap();

                atlas.deallocate_region(texture);
            }
        }
    }

    fn ensure_scratch_texture(&mut self, texture_parity: TextureParity) -> Result<(), AtlasError> {
        let index = texture_parity.get_parity();
        if self.scratch_atlases[index].is_some() {
            return Ok(());
        }

        self.request_textures(1)?;
        let size = self.texture_sizes.scratch_size(texture_parity);
        self.scratch_atlases[index] = Some(Atlas::new(
            AtlasId::new(u32::try_from(index).unwrap()),
            u32::from(size.width()),
            u32::from(size.height()),
        ));

        Ok(())
    }

    pub(super) fn add_layer_atlas(
        &mut self,
        texture_parity: TextureParity,
    ) -> Result<(), AtlasError> {
        self.request_textures(1)?;

        let parity = texture_parity.get_parity();
        let page_index = u16::try_from(self.layer_atlases[parity].len()).unwrap();
        let size = self.texture_sizes.layer_size(texture_parity);
        let atlas = Atlas::new(
            AtlasId::new(u32::from(page_index)),
            u32::from(size.width()),
            u32::from(size.height()),
        );

        self.layer_atlases[parity].push(atlas);

        Ok(())
    }

    fn request_textures(&mut self, count: usize) -> Result<(), AtlasError> {
        self.remaining_textures = self
            .remaining_textures
            .checked_sub(count)
            .ok_or(AtlasError::NoSpaceAvailable)?;

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerAllocationRequest {
    pub(super) texture_parity: TextureParity,
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
            size: SizeU16::from_wh(layer.bbox.width(), layer.bbox.height()),
            padding,
        };

        Self {
            texture_parity: layer.texture_parity,
            region,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RegionAllocationRequest {
    size: SizeU16,
    padding: u16,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ScratchAllocationRequest {
    region: RegionAllocationRequest,
    count: u8,
}

impl ScratchAllocationRequest {
    pub(super) fn for_filter(rect: RectU16, filter: &PreparedGpuFilter) -> Self {
        Self {
            region: RegionAllocationRequest {
                size: SizeU16::from_wh(rect.width(), rect.height()),
                padding: FILTER_ATLAS_PADDING,
            },
            count: filter.scratch_count(),
        }
    }

    pub(super) fn for_blend(rect: RectU16) -> Self {
        Self {
            region: RegionAllocationRequest {
                size: SizeU16::from_wh(rect.width(), rect.height()),
                padding: 0,
            },
            count: 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum AtlasAllocation {
    Layer(AllocatedTextureRegion<LayerTextureId>),
    Scratch(AllocatedTextureRegion<TextureParity>),
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AllocatedTextureRegion<T> {
    pub(super) region: TextureRegion<T>,
    padding: u16,
    alloc_id: AllocId,
}

impl<T: Copy> AllocatedTextureRegion<T> {
    fn new(region: TextureRegion<T>, padding: u16, alloc_id: AllocId) -> Self {
        Self {
            region,
            padding,
            alloc_id,
        }
    }

    pub(super) fn clear_region(self) -> TextureRegion<T> {
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

trait AtlasExt {
    fn allocate_region<T: Copy>(
        &mut self,
        target: T,
        request: RegionAllocationRequest,
    ) -> Option<AllocatedTextureRegion<T>>;

    fn deallocate_region<T: Copy>(&mut self, texture: AllocatedTextureRegion<T>);
}

impl AtlasExt for Atlas {
    fn allocate_region<T: Copy>(
        &mut self,
        target: T,
        request: RegionAllocationRequest,
    ) -> Option<AllocatedTextureRegion<T>> {
        let padding = u32::from(request.padding);
        let width = request.size.width();
        let height = request.size.height();
        let allocation_size = SizeU32::from(request.size) + padding * 2;
        let allocation = self.allocate(allocation_size.width(), allocation_size.height())?;
        let x = u16::try_from(allocation.x + padding).unwrap();
        let y = u16::try_from(allocation.y + padding).unwrap();
        let region = AllocatedTextureRegion::new(
            TextureRegion {
                target,
                rect: RectU16::new(x, y, x + width, y + height),
            },
            request.padding,
            allocation.id,
        );

        Some(region)
    }

    fn deallocate_region<T: Copy>(&mut self, texture: AllocatedTextureRegion<T>) {
        let allocation_region = texture.allocation_region();

        self.deallocate(
            texture.alloc_id,
            u32::from(allocation_region.width()),
            u32::from(allocation_region.height()),
        );
    }
}
