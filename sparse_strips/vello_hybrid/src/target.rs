// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render targets, texture bindings, and coordinate-space mappings.

use vello_common::geometry::RectU16;

/// The root target provided by the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RootTarget {
    /// The root target is the user-provided surface.
    UserSurface,
    /// The root render target is an atlas layer.
    AtlasLayer,
}

/// The target for a draw pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DrawPassTarget {
    /// Draw directly to the root.
    Root(RootTarget),
    /// Draw to an intermediate layer texture.
    Layer(LayerTextureId),
}

impl DrawPassTarget {
    pub(crate) fn enable_opaque(&self) -> bool {
        // We could support other targets, but then we need to pay the cost of a depth buffer.
        matches!(self, Self::Root(RootTarget::UserSurface))
    }
}

/// Even or odd intermediate texture group.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TextureParity {
    /// Texture group used by even layer depths.
    Even = 0,
    /// Texture group used by odd layer depths.
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

/// Identifies a page in an intermediate layer texture group.
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

/// Output and optional child input used by a draw pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DrawPassBindings {
    /// Texture or root surface receiving the draw.
    pub(crate) target: DrawPassTarget,
    /// Child layer sampled by the draw, if any.
    pub(crate) child: Option<LayerTextureId>,
}

impl DrawPassBindings {
    pub(crate) const fn new(target: DrawPassTarget, child: Option<LayerTextureId>) -> Self {
        Self { target, child }
    }
}

/// Target and temporary layer textures used by a filter pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FilterPassBindings {
    /// Texture containing the original input and receiving the final result.
    target: LayerTextureId,
    /// Opposite-parity texture used for filter ping-ponging.
    temporary: LayerTextureId,
}

impl FilterPassBindings {
    pub(crate) const fn new(target: LayerTextureId, temporary: LayerTextureId) -> Self {
        debug_assert!(target.texture_parity as u8 != temporary.texture_parity as u8);

        Self { target, temporary }
    }

    pub(crate) const fn target(self) -> LayerTextureId {
        self.target
    }

    pub(crate) const fn temporary(self) -> LayerTextureId {
        self.temporary
    }

    pub(crate) const fn input(self, step: usize) -> LayerTextureId {
        if step & 1 == 0 {
            self.target()
        } else {
            self.temporary()
        }
    }

    pub(crate) const fn output(self, step: usize) -> LayerTextureId {
        if step & 1 == 0 {
            self.temporary()
        } else {
            self.target()
        }
    }
}

/// Target and child layer textures used by a blend pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BlendPassBindings {
    /// Layer serving as the blend backdrop and destination.
    target: LayerTextureId,
    /// Child layer serving as the blend source.
    child: LayerTextureId,
}

impl BlendPassBindings {
    pub(crate) const fn new(target: LayerTextureId, child: LayerTextureId) -> Self {
        debug_assert!(target.texture_parity as u8 != child.texture_parity as u8);

        Self { target, child }
    }

    pub(crate) const fn target(self) -> LayerTextureId {
        self.target
    }

    pub(crate) const fn layer_id(self, parity: TextureParity) -> LayerTextureId {
        if self.target.texture_parity as u8 == parity as u8 {
            self.target
        } else {
            self.child
        }
    }
}

/// The bound layer textures for a round.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct RoundBindings {
    /// The bound page for each parity.
    pages: [Option<u16>; 2],
}

impl RoundBindings {
    pub(crate) const fn new(id: LayerTextureId) -> Self {
        let mut pages = [None; 2];

        pages[id.texture_parity.get_parity()] = Some(id.page_index);

        Self { pages }
    }

    /// Try to merge this binding with the other one.
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

    /// Return the bound texture for a parity, if present.
    pub(crate) fn layer_id(self, parity: TextureParity) -> Option<LayerTextureId> {
        self.pages[parity.get_parity()].map(|page_index| LayerTextureId::new(parity, page_index))
    }

    /// Return the required textures for this binding.
    pub(crate) fn required_textures(self) -> [Option<LayerTextureId>; 2] {
        core::array::from_fn(|index| {
            self.pages[index].map(|page_index| {
                LayerTextureId::new(TextureParity::from_parity(index), page_index)
            })
        })
    }

    #[cfg(test)]
    pub(crate) const fn page_indices(self) -> [Option<u16>; 2] {
        self.pages
    }
}

/// Rectangular region within a render target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextureRegion {
    /// Render target containing the region.
    pub(crate) target: LayerTextureId,
    /// Region in the layer texture.
    pub(crate) rect: RectU16,
}

/// Intermediate texture region paired with its scene-space layer bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayerTextureRegion {
    /// The texture region of the layer.
    pub(crate) texture: TextureRegion,
    /// Bounds of this layer in **viewport** coordinates.
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
    /// Whether this target supports the depth-buffer optimization.
    fn enable_depth(&self) -> bool;

    /// A positional shift that needs to be applied to all geometry when rendering to this target.
    fn geometry_shift(&self) -> (i32, i32);
}

impl DrawTarget for RootTarget {
    fn enable_depth(&self) -> bool {
        matches!(self, Self::UserSurface)
    }

    fn geometry_shift(&self) -> (i32, i32) {
        (0, 0)
    }
}

impl DrawTarget for LayerTextureRegion {
    fn enable_depth(&self) -> bool {
        false
    }

    fn geometry_shift(&self) -> (i32, i32) {
        // We always render layers such that their bbox starts at (0, 0) in the allocated
        // texture region, to minimize the consumed space.
        (
            self.texture.rect.x0 as i32 - i32::from(self.layer_bbox.x0),
            self.texture.rect.y0 as i32 - i32::from(self.layer_bbox.y0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{LayerTextureId, LayerTextureRegion, RoundBindings, TextureParity, TextureRegion};
    use vello_common::geometry::RectU16;

    #[test]
    fn texture_pair_constraints() {
        let constraint = |pages| RoundBindings { pages };
        let empty = constraint([None, None]);
        let even = constraint([Some(2), None]);
        let odd = constraint([None, Some(5)]);

        assert_eq!(even.merge(empty).unwrap().pages, [Some(2), None]);
        assert_eq!(empty.merge(even).unwrap().pages, [Some(2), None]);
        assert_eq!(even.merge(odd).unwrap().pages, [Some(2), Some(5)]);
        assert_eq!(even.merge(even).unwrap().pages, [Some(2), None]);
        assert!(even.merge(constraint([Some(3), None])).is_none());
        assert!(odd.merge(constraint([None, Some(6)])).is_none());
    }

    #[test]
    fn texture_rect_translation() {
        let layer = LayerTextureRegion {
            texture: TextureRegion {
                target: LayerTextureId::new(TextureParity::Even, 0),
                rect: RectU16::new(100, 200, 150, 250),
            },
            layer_bbox: RectU16::new(10, 20, 60, 70),
        };

        assert_eq!(
            layer.texture_rect(RectU16::new(15, 30, 25, 42)),
            RectU16::new(105, 210, 115, 222)
        );
    }
}
