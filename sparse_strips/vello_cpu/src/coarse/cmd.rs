// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko::BlendMode;
use crate::util::Span;
use core::num::NonZeroU32;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::Paint;

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
    use super::{AlphaIdx, FineCmd};
    use core::mem::{needs_drop, size_of};

    #[test]
    fn alpha_idx_size() {
        assert_eq!(size_of::<Option<AlphaIdx>>(), 4);
    }

    #[test]
    fn fine_cmd_assertions() {
        assert_eq!(size_of::<FineCmd>(), 16);
        assert!(!needs_drop::<FineCmd>());
    }
}
