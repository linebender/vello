// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::LayerClip;
use crate::peniko::BlendMode;
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
        content_bbox: RectU16,
    },
    CompositeFilterLayer {
        layer_id: usize,
        bbox: RectU16,
        src_x: u16,
        src_y: u16,
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
    AlphaFill(AlphaFillCmd),
    PushLayer,
    PopBuf,
    Opacity(f32),
    Mask(u32),
    BlendFill(BlendFillCmd),
    FilterLayer(FilterLayerCmd),
    BlendAlphaFill(BlendAlphaFillCmd),
}

impl FineCmd {
    #[inline(always)]
    pub(crate) fn generated_span(&self) -> Option<Span> {
        match self {
            Self::Fill(cmd) => Some(cmd.span),
            Self::AlphaFill(cmd) => Some(cmd.span),
            Self::BlendFill(cmd) => Some(cmd.span),
            Self::FilterLayer(cmd) => Some(cmd.span),
            Self::BlendAlphaFill(cmd) => Some(cmd.span),
            Self::PushLayer | Self::PopBuf | Self::Opacity(_) | Self::Mask(_) => None,
        }
    }

    #[inline(always)]
    pub(crate) fn fill_x(&self) -> u16 {
        match self {
            Self::Fill(cmd) => cmd.span.pixel_x(),
            Self::AlphaFill(cmd) => cmd.span.pixel_x(),
            Self::PushLayer
            | Self::PopBuf
            | Self::Opacity(_)
            | Self::Mask(_)
            | Self::FilterLayer(_)
            | Self::BlendFill(_)
            | Self::BlendAlphaFill(_) => unreachable!(),
        }
    }

    #[inline(always)]
    pub(crate) fn fill_width(&self) -> u16 {
        match self {
            Self::Fill(cmd) => cmd.span.pixel_width(),
            Self::AlphaFill(cmd) => cmd.span.pixel_width(),
            Self::PushLayer
            | Self::PopBuf
            | Self::Opacity(_)
            | Self::Mask(_)
            | Self::FilterLayer(_)
            | Self::BlendFill(_)
            | Self::BlendAlphaFill(_) => unreachable!(),
        }
    }

    #[inline(always)]
    pub(crate) fn fill_attrs_idx(&self) -> u32 {
        match self {
            Self::Fill(cmd) => cmd.attrs_idx,
            Self::AlphaFill(cmd) => cmd.attrs_idx,
            Self::PushLayer
            | Self::PopBuf
            | Self::Opacity(_)
            | Self::Mask(_)
            | Self::FilterLayer(_)
            | Self::BlendFill(_)
            | Self::BlendAlphaFill(_) => unreachable!(),
        }
    }
}

/// A horizontal tile-aligned span stored in tile coordinates.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Span {
    /// The horizontal start position in tile coordinates.
    tile_x: u16,
    /// The horizontal span width in tile coordinates.
    tile_width: u16,
}

impl Span {
    /// Creates a span from tile coordinates.
    #[inline(always)]
    pub(crate) fn new(tile_x: u16, tile_width: u16) -> Self {
        Self {
            tile_x,
            tile_width,
        }
    }

    /// Returns the horizontal start position in tile coordinates.
    #[inline(always)]
    pub(crate) fn tile_x(self) -> u16 {
        self.tile_x
    }

    /// Returns the exclusive horizontal end position in tile coordinates.
    #[inline(always)]
    pub(crate) fn tile_end(self) -> u16 {
        self.tile_x.saturating_add(self.tile_width)
    }

    /// Extends this span to include another span.
    #[inline(always)]
    pub(crate) fn extend(&mut self, other: Self) {
        let tile_x = self.tile_x.min(other.tile_x);
        let tile_end = self.tile_end().max(other.tile_end());
        *self = Self::new(tile_x, tile_end.saturating_sub(tile_x));
    }

    /// Returns the horizontal start position in pixels.
    #[inline(always)]
    pub(crate) fn pixel_x(self) -> u16 {
        self.tile_x * Tile::WIDTH
    }

    /// Returns the horizontal span width in pixels.
    #[inline(always)]
    pub(crate) fn pixel_width(self) -> u16 {
        self.tile_width * Tile::WIDTH
    }

    /// Returns the exclusive horizontal end position in pixels.
    #[inline(always)]
    pub(crate) fn pixel_end(self) -> u16 {
        self.pixel_x().saturating_add(self.pixel_width())
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FillCmd {
    pub(crate) span: Span,
    pub(crate) attrs_idx: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AlphaFillCmd {
    pub(crate) span: Span,
    pub(crate) alpha_idx: u32,
    pub(crate) attrs_idx: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlendFillCmd {
    pub(crate) span: Span,
    pub(crate) attrs_idx: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterLayerCmd {
    pub(crate) span: Span,
    pub(crate) attrs_idx: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlendAlphaFillCmd {
    pub(crate) span: Span,
    pub(crate) alpha_idx: u32,
    pub(crate) attrs_idx: u32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GeneratedFill {
    pub(super) span: Span,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GeneratedAlphaFill {
    pub(super) span: Span,
    pub(super) alpha_idx: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct FillAttrs {
    pub(crate) paint: Paint,
    pub(crate) blend_mode: BlendMode,
    pub(crate) mask: Option<Mask>,
    pub(crate) path_id: u32,
    pub(crate) thread_idx: u8,
    pub(crate) paint_offset: (u16, u16),
}

#[derive(Debug, Clone)]
pub(crate) struct BlendAttrs {
    pub(crate) blend_mode: BlendMode,
    pub(crate) thread_idx: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct FilterLayerAttrs {
    pub(crate) layer_id: usize,
    pub(crate) path_id: u32,
    pub(crate) src_x: u16,
    pub(crate) src_y: u16,
    pub(crate) y0: u16,
    pub(crate) y1: u16,
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
