// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render targets and regions used by the hybrid renderer.

use crate::util::Int16Size;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextureIndex {
    Even = 0,
    Odd = 1,
}

impl TextureIndex {
    pub(crate) const fn from_parity(value: usize) -> Self {
        if value & 1 == 0 {
            Self::Even
        } else {
            Self::Odd
        }
    }

    pub(crate) const fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Even,
            1 => Self::Odd,
            _ => panic!("texture index must be 0 or 1"),
        }
    }

    pub(crate) const fn is_even(self) -> bool {
        matches!(self, Self::Even)
    }

    pub(crate) const fn get_index(self) -> usize {
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

impl From<TextureIndex> for u8 {
    fn from(index: TextureIndex) -> Self {
        index as Self
    }
}

impl From<TextureIndex> for u32 {
    fn from(index: TextureIndex) -> Self {
        u8::from(index).into()
    }
}

/// The target of an executable draw pass.
pub(crate) type DrawPassTarget = RenderTarget<TextureIndex>;

/// Identifies one of the intermediate textures used by the hybrid renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextureTarget {
    /// A layer atlas texture.
    Layer(TextureIndex),
    /// A scratch texture.
    Scratch(TextureIndex),
}

impl TextureTarget {
    pub(crate) fn layer(index: TextureIndex) -> Self {
        Self::Layer(index)
    }

    pub(crate) fn scratch(index: TextureIndex) -> Self {
        Self::Scratch(index)
    }

    pub(crate) fn index(self) -> TextureIndex {
        match self {
            Self::Layer(index) | Self::Scratch(index) => index,
        }
    }
}

/// Dimensions of the intermediate textures used by the hybrid renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IntermediateTextureSizes {
    layer: [Int16Size; 2],
    scratch: [Int16Size; 2],
}

impl IntermediateTextureSizes {
    pub(crate) fn new(layer: [Int16Size; 2], scratch: [Int16Size; 2]) -> Self {
        Self { layer, scratch }
    }

    pub(crate) fn uniform(size: Int16Size) -> Self {
        Self::new([size; 2], [size; 2])
    }

    pub(crate) fn size(self, target: TextureTarget) -> Int16Size {
        match target {
            TextureTarget::Layer(index) => self.layer[index.get_index()],
            TextureTarget::Scratch(index) => self.scratch[index.get_index()],
        }
    }
}

/// A rectangular region in one of the intermediate textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextureRegion {
    /// Texture index, currently `0` or `1`.
    pub(crate) texture_index: TextureIndex,
    /// Region in the texture.
    pub(crate) rect: RectU16,
}

/// A layer texture region with its corresponding viewport-space bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerTextureRegion {
    /// Region in the layer texture.
    pub(crate) texture: TextureRegion,
    /// Bounds of this layer in viewport coordinates.
    pub(crate) layer_bbox: RectU16,
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

impl LayerTextureRegion {
    pub(crate) fn blend_scratch_clear_rect(self, blend_bbox: RectU16) -> RectU16 {
        let x0 = self.texture.rect.x0 + (blend_bbox.x0 - self.layer_bbox.x0);
        let y0 = self.texture.rect.y0 + (blend_bbox.y0 - self.layer_bbox.y0);
        RectU16::new(x0, y0, x0 + blend_bbox.width(), y0 + blend_bbox.height())
    }
}
