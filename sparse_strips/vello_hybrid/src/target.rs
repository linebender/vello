// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render targets and regions used by the hybrid renderer.

use vello_common::geometry::RectU16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootRenderTarget {
    /// The root render target is the user-provided surface.
    UserSurface,
    /// The root render target is an atlas layer.
    AtlasLayer,
}

/// A render target whose layer variant carries target-specific data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderTarget<L> {
    /// Render to the root output target.
    Root(RootRenderTarget),
    /// Render to a layer target.
    Layer(L),
}

impl<L> RenderTarget<L> {
    pub(crate) fn enable_opaque(&self) -> bool {
        matches!(self, Self::Root(RootRenderTarget::UserSurface))
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TextureParity {
    Even = 0,
    Odd = 1,
}

impl TextureParity {
    pub(crate) const fn from_parity(parity: usize) -> Self {
        if parity & 1 == 0 {
            Self::Even
        } else {
            Self::Odd
        }
    }

    pub(crate) const fn get_parity(self) -> usize {
        match self {
            Self::Even => 0,
            Self::Odd => 1,
        }
    }

    pub(crate) const fn opposite(self) -> Self {
        match self {
            Self::Even => Self::Odd,
            Self::Odd => Self::Even,
        }
    }
}

impl From<TextureParity> for u8 {
    fn from(parity: TextureParity) -> Self {
        parity as Self
    }
}

impl From<TextureParity> for u32 {
    fn from(parity: TextureParity) -> Self {
        u8::from(parity).into()
    }
}

/// An identifier for a layer texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LayerTextureId {
    /// The parity of the texture.
    pub(crate) texture_parity: TextureParity,
    /// The page index of the texture within the parity group.
    pub(crate) page_index: u16,
}

impl LayerTextureId {
    pub(crate) const fn new(texture_parity: TextureParity, page_index: u16) -> Self {
        Self {
            texture_parity,
            page_index,
        }
    }
}

/// The page indices of a pair of even and odd parity textures
/// that should be bound during a draw call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LayerTexturePair {
    pub(crate) page_indices: [u16; 2],
}

impl LayerTexturePair {
    pub(crate) const fn layer_id(self, parity: TextureParity) -> LayerTextureId {
        LayerTextureId::new(parity, self.page_indices[parity.get_parity()])
    }
}

/// A pair of two layer textures used for a filter operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FilterTexturePair {
    pair: LayerTexturePair,
    original_parity: TextureParity,
}

impl FilterTexturePair {
    pub(crate) const fn new(pair: LayerTexturePair, original_parity: TextureParity) -> Self {
        Self {
            pair,
            original_parity,
        }
    }

    pub(crate) const fn original(self) -> LayerTextureId {
        self.pair.layer_id(self.original_parity)
    }

    pub(crate) const fn pair(self) -> LayerTexturePair {
        self.pair
    }

    pub(crate) const fn temporary(self) -> LayerTextureId {
        self.pair.layer_id(self.original_parity.opposite())
    }

    pub(crate) const fn input(self, step: usize) -> LayerTextureId {
        if step & 1 == 0 {
            self.original()
        } else {
            self.temporary()
        }
    }

    pub(crate) const fn output(self, step: usize) -> LayerTextureId {
        if step & 1 == 0 {
            self.temporary()
        } else {
            self.original()
        }
    }
}

/// A constraint for layer texture pairs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LayerTexturePairConstraint {
    pages: [Option<u16>; 2],
}

impl LayerTexturePairConstraint {
    /// Create a new layer texture pair constraint with the given layer texture ID.
    pub(crate) const fn new(id: LayerTextureId) -> Self {
        let mut pages = [None; 2];
        pages[id.texture_parity.get_parity()] = Some(id.page_index);
        Self { pages }
    }

    /// Try to merge a layer texture pair constraint with the current one.
    pub(crate) fn merge(mut self, other: Self) -> Option<Self> {
        for (current, required) in self.pages.iter_mut().zip(other.pages) {
            match (*current, required) {
                (Some(current), Some(required)) if current != required => return None,
                (None, Some(required)) => *current = Some(required),
                _ => {}
            }
        }

        Some(self)
    }

    /// Resolve the layer texture pair constraint into a concrete texture pair.
    pub(crate) fn resolve(self) -> LayerTexturePair {
        LayerTexturePair {
            page_indices: self.pages.map(|page| page.unwrap_or(0)),
        }
    }

    /// Return the layer textures required by this constraint.
    pub(crate) fn required_textures(self) -> [Option<LayerTextureId>; 2] {
        core::array::from_fn(|index| {
            self.pages[index].map(|page_index| {
                LayerTextureId::new(TextureParity::from_parity(index), page_index)
            })
        })
    }
}

/// The target of an executable draw pass.
pub(crate) type DrawPassTarget = RenderTarget<LayerTextureId>;

/// A rectangular region in one of the intermediate textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextureRegion<T> {
    /// The texture containing this region.
    pub(crate) target: T,
    /// Region in the texture.
    pub(crate) rect: RectU16,
}

/// A rectangular region in a layer texture.
pub(crate) type LayerRegion = TextureRegion<LayerTextureId>;

/// A layer texture region with its corresponding viewport-space bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerTextureRegion {
    /// Region in the layer texture.
    pub(crate) texture: LayerRegion,
    /// Bounds of this layer in viewport coordinates.
    pub(crate) layer_bbox: RectU16,
}

impl LayerTextureRegion {
    /// Translate a scene-space rectangle within this layer to texture coordinates.
    ///
    /// The given bbox must be fully contained within the layer bbox.
    pub(crate) fn texture_rect(self, bbox: RectU16) -> RectU16 {
        let x0 = self.texture.rect.x0 + (bbox.x0.checked_sub(self.layer_bbox.x0).unwrap());
        let y0 = self.texture.rect.y0 + (bbox.y0.checked_sub(self.layer_bbox.y0).unwrap());

        RectU16::new(x0, y0, x0 + bbox.width(), y0 + bbox.height())
    }
}

pub(crate) trait DrawTarget {
    fn enable_opaque(&self) -> bool;

    fn geometry_shift(&self) -> (i32, i32);
}

impl DrawTarget for RootRenderTarget {
    fn enable_opaque(&self) -> bool {
        matches!(self, Self::UserSurface)
    }

    fn geometry_shift(&self) -> (i32, i32) {
        (0, 0)
    }
}

impl DrawTarget for LayerTextureRegion {
    fn enable_opaque(&self) -> bool {
        false
    }

    fn geometry_shift(&self) -> (i32, i32) {
        (
            self.texture.rect.x0 as i32 - i32::from(self.layer_bbox.x0),
            self.texture.rect.y0 as i32 - i32::from(self.layer_bbox.y0),
        )
    }
}
