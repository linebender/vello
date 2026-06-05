// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko::BlendMode;
use crate::util::Span;
use core::num::NonZeroU32;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::Paint;

// TODO: If we wanted to, we could likely reduce the memory footprint from 16 bytes to 8 bytes
// by using 4 bits from a span (since tile size is 4, for `x` and `width` we can always store
// the value divided by 4), and creating a bit-packed representation of all commands.

/// A bucketed render command.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RenderCmd {
    /// See [`PaintFill`].
    PaintFill(PaintFill),
    /// Push a new temporary layer buffer.
    PushBuf,
    /// Pop the last temporary layer buffer.
    PopBuf,
    /// See [`LayerFill`].
    LayerFill(LayerFill),
    /// See [`FilterLayerFill`].
    FilterLayerFill(FilterLayerFill),
}

impl RenderCmd {
    #[inline]
    pub(crate) fn span(self) -> Option<Span> {
        match self {
            Self::PaintFill(cmd) => Some(cmd.span),
            Self::LayerFill(cmd) => Some(cmd.span),
            Self::FilterLayerFill(cmd) => Some(cmd.span),
            Self::PushBuf | Self::PopBuf => None,
        }
    }
}

/// Fill a span with the given paint and optionally some alpha coverage.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PaintFill {
    pub(crate) span: Span,
    alpha_idx: Option<AlphaIdx>,
    pub(crate) attrs_idx: u32,
}

impl PaintFill {
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

/// Composite a span from the current temporary layer buffer into the parent
/// buffer and optionally apply some alpha coverage.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerFill {
    pub(crate) span: Span,
    alpha_idx: Option<AlphaIdx>,
    pub(crate) attrs_idx: u32,
}

impl LayerFill {
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

/// Composite a span from a rendered filter layer pixmap
/// into the current buffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterLayerFill {
    pub(crate) span: Span,
    pub(crate) attrs_idx: u32,
}

// We use `NonZeroU32` so that `Option<AlphaIdx>` still only needs 4 bytes.
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

#[derive(Debug, Clone)]
pub(crate) struct PaintFillAttrs {
    pub(crate) paint: Paint,
    pub(crate) blend_mode: BlendMode,
    pub(crate) mask: Option<Mask>,
    pub(crate) draw_id: u32,
    pub(crate) thread_idx: u8,
    /// See the comment in `CommandBucketer::bucket_commands`.
    pub(crate) pixmap_origin: (u16, u16),
}

#[derive(Debug, Clone)]
pub(crate) struct LayerFillAttrs {
    pub(crate) blend_mode: BlendMode,
    pub(crate) opacity: f32,
    pub(crate) mask: Option<Mask>,
    /// In case there is any alpha associated with the layer command, this stores
    /// the index of the thread that stores the alpha.
    pub(crate) thread_idx: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct FilterLayerFillAttrs {
    /// The ID of the filter layer.
    pub(crate) id: usize,
    pub(crate) draw_id: u32,
    pub(crate) dst_bbox: RectU16,
    pub(crate) src_origin: (u16, u16),
}

#[cfg(test)]
mod tests {
    use super::{AlphaIdx, RenderCmd};
    use core::mem::{needs_drop, size_of};

    #[test]
    fn alpha_idx_size() {
        assert_eq!(size_of::<Option<AlphaIdx>>(), 4);
    }

    #[test]
    fn render_cmd_assertions() {
        assert_eq!(size_of::<RenderCmd>(), 16);
        assert!(!needs_drop::<RenderCmd>());
    }
}
