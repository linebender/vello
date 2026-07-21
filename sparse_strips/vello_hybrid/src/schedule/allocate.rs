// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Atlas allocation for scheduled layer texture regions.

use crate::filter::FILTER_ATLAS_PADDING;
use crate::scene::LayersConfig;
use crate::target::{LayerTextureId, TextureParity, TextureRegion};
use alloc::vec::Vec;
use vello_common::geometry::{RectU16, SizeU16, SizeU32};
use vello_common::multi_atlas::{AllocId, Atlas, AtlasError, AtlasId};
use vello_common::record::RecordedLayerKind;

/// An allocation and the first round in which it can be used.
#[derive(Debug, Clone, Copy)]
pub(super) struct Allocation {
    /// The allocated texture region.
    pub(super) allocation: AllocatedTextureRegion,
    /// The round starting from which this allocation is available.
    pub(super) round_idx: usize,
}

/// Intermediate texture atlases and their shared texture budget.
#[derive(Debug)]
pub(super) struct Atlases {
    /// The atlases for each texture parity.
    layer_atlases: [Vec<Atlas>; 2],
    /// Whether the texture budget includes the shared scratch texture.
    scratch_texture: bool,
    /// Number of additional intermediate textures allowed by the configured limit.
    remaining_textures: usize,
    /// Dimensions of every atlas page and the scratch texture.
    texture_size: SizeU16,
}

impl Atlases {
    pub(super) fn new(texture_size: SizeU16, layer_config: LayersConfig) -> Self {
        Self {
            layer_atlases: core::array::from_fn(|_| Vec::new()),
            scratch_texture: false,
            remaining_textures: layer_config.max_textures.unwrap_or(usize::MAX),
            texture_size,
        }
    }

    pub(super) fn scratch_texture(&self) -> bool {
        self.scratch_texture
    }

    pub(super) fn allocate_layer(
        &mut self,
        request: &LayerAllocationRequest,
    ) -> Option<AllocatedTextureRegion> {
        let parity = request.texture_parity.get_parity();

        // Search for space in all available pages, always preferring lower-index ones.
        //
        // It's unfortunate the runtime of this will grow linearly with each allocated page
        // for every allocation request. But since our scheduler is as conservative as possible,
        // it is unlikely we will ever have more than 2-3 pages, except for scene graphs that have
        // lots of nested layers with multiple children. The maximum page count corresponds roughly
        // to the maximum layer depth, and this will only be reached if every layer in the chain
        // has more than 1 child.
        for page_index in 0..u16::try_from(self.layer_atlases[parity].len()).unwrap() {
            let id = LayerTextureId::new(request.texture_parity, page_index);
            let atlas = &mut self.layer_atlases[parity][usize::from(page_index)];

            if let Some(allocation) = atlas.allocate_region(id, request.region) {
                return Some(allocation);
            }
        }

        None
    }

    pub(super) fn deallocate(&mut self, texture: AllocatedTextureRegion) {
        let id = texture.region.target;
        let atlas =
            &mut self.layer_atlases[id.texture_parity.get_parity()][usize::from(id.page_index)];

        atlas.deallocate_region(texture);
    }

    pub(super) fn require_scratch_texture(&mut self) -> Result<(), AtlasError> {
        if self.scratch_texture {
            return Ok(());
        }

        self.request_textures(1)?;
        self.scratch_texture = true;

        Ok(())
    }

    pub(super) fn add_layer_atlas(
        &mut self,
        texture_parity: TextureParity,
    ) -> Result<(), AtlasError> {
        self.request_textures(1)?;

        let parity = texture_parity.get_parity();
        let page_index = u16::try_from(self.layer_atlases[parity].len()).unwrap();
        let size = self.texture_size;
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

/// A request for a layer allocation.
#[derive(Debug, Clone, Copy)]
pub(super) struct LayerAllocationRequest {
    /// Texture group from which the region must be allocated.
    pub(super) texture_parity: TextureParity,
    /// Size and padding required within the selected atlas.
    region: RegionProps,
}

impl LayerAllocationRequest {
    pub(super) fn new(
        bbox: RectU16,
        kind: &RecordedLayerKind,
        texture_parity: TextureParity,
    ) -> Self {
        let padding = match kind {
            RecordedLayerKind::Regular => 0,
            // Padding is needed because some filters use bilinear sampling for
            // improved performance. Therefore, we need to ensure there is transparent
            // padding on the outside.
            RecordedLayerKind::Filter { .. } => FILTER_ATLAS_PADDING,
        };

        let region = RegionProps {
            size: SizeU16::from_wh(bbox.width(), bbox.height()),
            padding,
        };

        Self {
            texture_parity,
            region,
        }
    }
}

/// Size and padding of a region allocated from one atlas page.
#[derive(Debug, Clone, Copy)]
struct RegionProps {
    /// Size of the usable region.
    size: SizeU16,
    /// Transparent padding reserved around the usable region.
    padding: u16,
}

/// Texture region and allocator metadata needed to release it.
#[derive(Debug, Clone, Copy)]
pub(super) struct AllocatedTextureRegion {
    /// Usable portion of the allocation.
    pub(super) region: TextureRegion,
    /// Padding surrounding the usable region.
    padding: u16,
    /// Identifier used to return the allocation to its atlas.
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

trait AtlasExt {
    fn allocate_region(
        &mut self,
        target: LayerTextureId,
        props: RegionProps,
    ) -> Option<AllocatedTextureRegion>;

    fn deallocate_region(&mut self, texture: AllocatedTextureRegion);
}

impl AtlasExt for Atlas {
    fn allocate_region(
        &mut self,
        target: LayerTextureId,
        props: RegionProps,
    ) -> Option<AllocatedTextureRegion> {
        let padding = u32::from(props.padding);
        let width = props.size.width();
        let height = props.size.height();
        let allocation_size = SizeU32::from(props.size) + padding * 2;
        let allocation = self.allocate(allocation_size.width(), allocation_size.height())?;
        let x = u16::try_from(allocation.x + padding).unwrap();
        let y = u16::try_from(allocation.y + padding).unwrap();
        let region = AllocatedTextureRegion::new(
            TextureRegion {
                target,
                rect: RectU16::new(x, y, x + width, y + height),
            },
            props.padding,
            allocation.id,
        );

        Some(region)
    }

    fn deallocate_region(&mut self, texture: AllocatedTextureRegion) {
        let allocation_region = texture.allocation_region();

        self.deallocate(
            texture.alloc_id,
            u32::from(allocation_region.width()),
            u32::from(allocation_region.height()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{AtlasExt, Atlases, FILTER_ATLAS_PADDING, LayerAllocationRequest, RegionProps};
    use crate::scene::LayersConfig;
    use crate::target::{LayerTextureId, TextureParity};
    use vello_common::filter::{FilterData, FilterLayerPlacement};
    use vello_common::filter_effects::{Filter, FilterPrimitive};
    use vello_common::geometry::{RectU16, SizeU16};
    use vello_common::kurbo::Affine;
    use vello_common::multi_atlas::{Atlas, AtlasError, AtlasId};
    use vello_common::record::RecordedLayerKind;

    fn config(max_textures: usize) -> LayersConfig {
        LayersConfig {
            max_textures: Some(max_textures),
            ..Default::default()
        }
    }

    fn request(
        texture_parity: TextureParity,
        size: SizeU16,
        padding: u16,
    ) -> LayerAllocationRequest {
        LayerAllocationRequest {
            texture_parity,
            region: RegionProps { size, padding },
        }
    }

    #[test]
    fn layer_requests() {
        let bbox = RectU16::new(4, 8, 20, 32);
        let regular_kind = RecordedLayerKind::Regular;
        let regular = LayerAllocationRequest::new(bbox, &regular_kind, TextureParity::Odd);
        assert_eq!(regular.texture_parity, TextureParity::Odd);
        assert_eq!(regular.region.size, SizeU16::from_wh(16, 24));
        assert_eq!(regular.region.padding, 0);

        let filter_kind = RecordedLayerKind::Filter {
            filter_data: FilterData::new(
                Filter::from_primitive(FilterPrimitive::Offset { dx: 1.0, dy: 2.0 }),
                Affine::IDENTITY,
            ),
            placement: FilterLayerPlacement {
                pixmap_bbox: RectU16::ZERO,
                dest_bbox: RectU16::ZERO,
                src_x: 0,
                src_y: 0,
            },
        };
        let filter = LayerAllocationRequest::new(bbox, &filter_kind, TextureParity::Odd);
        assert_eq!(filter.texture_parity, TextureParity::Odd);
        assert_eq!(filter.region.size, SizeU16::from_wh(16, 24));
        assert_eq!(filter.region.padding, FILTER_ATLAS_PADDING);

        let temporary = LayerAllocationRequest::new(
            RectU16::new(10, 20, 42, 68),
            &filter_kind,
            TextureParity::Even,
        );
        assert_eq!(temporary.texture_parity, TextureParity::Even);
        assert_eq!(temporary.region.size, SizeU16::from_wh(32, 48));
        assert_eq!(temporary.region.padding, FILTER_ATLAS_PADDING);
    }

    #[test]
    fn page_selection() {
        let size = SizeU16::new(8);
        let mut atlases = Atlases::new(size, config(3));
        let even = request(TextureParity::Even, size, 0);
        let odd = request(TextureParity::Odd, size, 0);

        atlases.add_layer_atlas(TextureParity::Even).unwrap();
        let even_0 = atlases.allocate_layer(&even).unwrap();
        assert_eq!(
            even_0.region.target,
            LayerTextureId::new(TextureParity::Even, 0)
        );
        // It's already full.
        assert!(atlases.allocate_layer(&even).is_none());

        atlases.add_layer_atlas(TextureParity::Even).unwrap();
        let even_1 = atlases.allocate_layer(&even).unwrap();
        assert_eq!(
            even_1.region.target,
            LayerTextureId::new(TextureParity::Even, 1)
        );

        atlases.add_layer_atlas(TextureParity::Odd).unwrap();
        let odd_0 = atlases.allocate_layer(&odd).unwrap();
        assert_eq!(
            odd_0.region.target,
            LayerTextureId::new(TextureParity::Odd, 0)
        );

        atlases.deallocate(even_0);
        let reused = atlases.allocate_layer(&even).unwrap();
        assert_eq!(
            reused.region.target,
            LayerTextureId::new(TextureParity::Even, 0)
        );
    }

    #[test]
    fn page_capacity() {
        let mut atlases = Atlases::new(SizeU16::from_wh(16, 8), config(1));
        let request = request(TextureParity::Even, SizeU16::new(8), 0);
        atlases.add_layer_atlas(TextureParity::Even).unwrap();

        let first = atlases.allocate_layer(&request).unwrap();
        let second = atlases.allocate_layer(&request).unwrap();
        assert_eq!(
            first.region.target,
            LayerTextureId::new(TextureParity::Even, 0)
        );
        assert_eq!(
            second.region.target,
            LayerTextureId::new(TextureParity::Even, 0)
        );

        assert!(atlases.allocate_layer(&request).is_none());
        assert!(matches!(
            atlases.add_layer_atlas(TextureParity::Even),
            Err(AtlasError::NoSpaceAvailable)
        ));
    }

    #[test]
    fn padded_reuse() {
        let mut atlas = Atlas::new(AtlasId::new(0), 8, 8);
        let target = LayerTextureId::new(TextureParity::Even, 0);
        let request = RegionProps {
            size: SizeU16::new(6),
            padding: 1,
        };

        let allocation = atlas.allocate_region(target, request).unwrap();
        assert_eq!(allocation.region.rect, RectU16::new(1, 1, 7, 7));
        assert_eq!(allocation.clear_region().rect, RectU16::new(1, 1, 7, 7));
        assert_eq!(allocation.allocation_region(), RectU16::new(0, 0, 8, 8));
        assert!(atlas.allocate_region(target, request).is_none());

        atlas.deallocate_region(allocation);
        let reused = atlas.allocate_region(target, request).unwrap();
        assert_eq!(reused.region.rect, RectU16::new(1, 1, 7, 7));
    }

    #[test]
    fn texture_budget() {
        let mut atlases = Atlases::new(SizeU16::new(8), config(2));

        atlases.add_layer_atlas(TextureParity::Even).unwrap();
        atlases.require_scratch_texture().unwrap();
        atlases.require_scratch_texture().unwrap();

        assert!(atlases.scratch_texture());
        assert!(matches!(
            atlases.add_layer_atlas(TextureParity::Odd),
            Err(AtlasError::NoSpaceAvailable)
        ));
    }

    #[test]
    fn oversized_regions() {
        let mut atlas = Atlas::new(AtlasId::new(0), 8, 8);
        let target = LayerTextureId::new(TextureParity::Even, 0);

        assert!(
            atlas
                .allocate_region(
                    target,
                    RegionProps {
                        size: SizeU16::from_wh(9, 8),
                        padding: 0,
                    },
                )
                .is_none()
        );
        assert!(
            atlas
                .allocate_region(
                    target,
                    RegionProps {
                        size: SizeU16::new(8),
                        padding: 1,
                    },
                )
                .is_none()
        );
    }
}
