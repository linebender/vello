// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::LayerClip;
use crate::peniko::BlendMode;
use core::num::NonZeroU32;
use core::ops::Range;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::tile::Tile;

#[derive(Debug)]
pub(crate) enum RenderCmd {
    Fill {
        thread_idx: u8,
        strip_range: Range<usize>,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    },
    PushLayer {
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        clip: Option<LayerClip>,
        bbox: RectU16,
    },
    CompositeFilterLayer {
        id: usize,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
        clip: Option<LayerClip>,
    },
    PopLayer,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum FineCmd {
    Fill(FillCmd),
    PushLayer,
    PopBuf,
    Opacity(f32),
    Mask(u32),
    BlendFill(BlendFillCmd),
    FilterLayer(FilterLayerCmd),
}

impl FineCmd {
    #[inline]
    pub(crate) fn generated_span(self) -> Option<Span> {
        match self {
            Self::Fill(cmd) => Some(cmd.span),
            Self::BlendFill(cmd) => Some(cmd.span),
            Self::FilterLayer(cmd) => Some(cmd.span),
            Self::PushLayer | Self::PopBuf | Self::Opacity(_) | Self::Mask(_) => None,
        }
    }
}

/// A horizontal tile-aligned span.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Span {
    /// The horizontal start position in tile coordinates.
    tile_x: u16,
    /// The horizontal span width in tile coordinates.
    tile_width: u16,
}

impl Span {
    /// Creates a span from tile coordinates.
    pub(crate) fn new(tile_x: u16, tile_width: u16) -> Self {
        Self { tile_x, tile_width }
    }

    /// Returns the horizontal start position in tile coordinates.
    pub(crate) fn tile_x(self) -> u16 {
        self.tile_x
    }

    /// Returns the exclusive horizontal end position in tile coordinates.
    pub(crate) fn tile_end(self) -> u16 {
        self.tile_x.saturating_add(self.tile_width)
    }

    /// Extends this span to include another span.
    pub(crate) fn extend(&mut self, other: Self) {
        let tile_x = self.tile_x.min(other.tile_x);
        let tile_end = self.tile_end().max(other.tile_end());
        *self = Self::new(tile_x, tile_end.saturating_sub(tile_x));
    }

    /// Returns the horizontal start position in pixels.
    pub(crate) fn pixel_x(self) -> u16 {
        self.tile_x * Tile::WIDTH
    }

    /// Returns the horizontal span width in pixels.
    pub(crate) fn pixel_width(self) -> u16 {
        self.tile_width * Tile::WIDTH
    }

    /// Returns the exclusive horizontal end position in pixels.
    pub(crate) fn pixel_end(self) -> u16 {
        self.pixel_x().saturating_add(self.pixel_width())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FillCmd {
    pub(crate) span: Span,
    alpha_idx: Option<AlphaIdx>,
    pub(crate) attrs_idx: u32,
}

impl FillCmd {
    pub(crate) fn new(span: Span, alpha_idx: Option<u32>, attrs_idx: u32) -> Self {
        Self {
            span,
            alpha_idx: alpha_idx.map(AlphaIdx::new),
            attrs_idx,
        }
    }

    pub(crate) fn alpha_idx(self) -> Option<u32> {
        self.alpha_idx.map(AlphaIdx::get)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlendFillCmd {
    pub(crate) span: Span,
    alpha_idx: Option<AlphaIdx>,
    pub(crate) attrs_idx: u32,
}

impl BlendFillCmd {
    pub(crate) fn new(span: Span, alpha_idx: Option<u32>, attrs_idx: u32) -> Self {
        Self {
            span,
            alpha_idx: alpha_idx.map(AlphaIdx::new),
            attrs_idx,
        }
    }

    pub(crate) fn alpha_idx(self) -> Option<u32> {
        self.alpha_idx.map(AlphaIdx::get)
    }
}

#[derive(Debug, Clone, Copy)]
struct AlphaIdx(NonZeroU32);

impl AlphaIdx {
    fn new(alpha_idx: u32) -> Self {
        Self(NonZeroU32::new(alpha_idx.checked_add(1).expect("alpha index overflow")).unwrap())
    }

    fn get(self) -> u32 {
        self.0.get() - 1
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterLayerCmd {
    pub(crate) span: Span,
    pub(crate) attrs_idx: u32,
}

/// Attributes of a fill command.
#[derive(Debug, Clone)]
pub(crate) struct FillAttrs {
    pub(crate) paint: Paint,
    pub(crate) blend_mode: BlendMode,
    pub(crate) mask: Option<Mask>,
    pub(crate) draw_id: u32,
    pub(crate) thread_idx: u8,
    pub(crate) paint_offset: (u16, u16),
}

#[derive(Debug, Clone)]
pub(crate) struct BlendAttrs {
    pub(crate) blend_mode: BlendMode,
    pub(crate) thread_idx: u8,
}

/// Attributes of a filter layer.
#[derive(Debug, Clone)]
pub(crate) struct FilterLayerAttrs {
    pub(crate) id: usize,
    pub(crate) draw_id: u32,
    pub(crate) dst_bbox: RectU16,
    pub(crate) src_origin: (u16, u16),
}

#[cfg(test)]
mod tests {
    use super::FineCmd;
    use core::mem::{needs_drop, size_of};

    #[test]
    fn fine_cmd_assertions() {
        assert_eq!(size_of::<FineCmd>(), 16);
        assert!(!needs_drop::<FineCmd>());
    }
}
