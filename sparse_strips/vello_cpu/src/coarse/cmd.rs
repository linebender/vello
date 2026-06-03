// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko::BlendMode;
use core::num::NonZeroU32;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::tile::Tile;

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

/// A horizontal span in pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Span {
    /// The horizontal start position in pixels.
    x: u16,
    /// The horizontal span width in pixels.
    width: u16,
}

impl Span {
    /// Creates a span from pixel coordinates.
    pub(crate) fn new(x: u16, width: u16) -> Self {
        Self { x, width }
    }

    /// Creates a span from tile coordinates.
    pub(crate) fn new_tile(tile_x: u16, tile_width: u16) -> Self {
        Self {
            x: tile_x * Tile::WIDTH,
            width: tile_width * Tile::WIDTH,
        }
    }

    /// Returns the horizontal start position in tile coordinates.
    pub(crate) fn tile_x(self) -> u16 {
        self.x / Tile::WIDTH
    }

    /// Returns the exclusive horizontal end position in tile coordinates.
    pub(crate) fn tile_end(self) -> u16 {
        self.pixel_end().div_ceil(Tile::WIDTH)
    }

    /// Extends this span to include another span.
    pub(crate) fn extend(&mut self, other: Self) {
        let x = self.x.min(other.x);
        let end = self.pixel_end().max(other.pixel_end());
        *self = Self::new(x, end.saturating_sub(x));
    }

    /// Returns the intersection of this span with another span.
    pub(crate) fn intersect(self, other: Self) -> Option<Self> {
        let x = self.x.max(other.x);
        let end = self.pixel_end().min(other.pixel_end());
        (x < end).then(|| Self::new(x, end - x))
    }

    /// Returns the horizontal start position in pixels.
    pub(crate) fn pixel_x(self) -> u16 {
        self.x
    }

    /// Returns the horizontal span width in pixels.
    pub(crate) fn pixel_width(self) -> u16 {
        self.width
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
