// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::LayerClip;
use crate::peniko::BlendMode;
use core::ops::Range;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::Paint;

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

#[derive(Debug, Clone)]
pub(crate) enum FineCmd {
    Fill(FillCmd),
    AlphaFill(AlphaFillCmd),
    PushLayer,
    PopBuf,
    Opacity(f32),
    Mask(Mask),
    BlendFill(BlendFillCmd),
    FilterLayer(FilterLayerCmd),
    BlendAlphaFill(BlendAlphaFillCmd),
}

impl FineCmd {
    #[inline(always)]
    pub(super) fn generated_span(&self) -> Option<(u16, u16)> {
        match self {
            Self::Fill(cmd) => Some((cmd.x, cmd.width)),
            Self::AlphaFill(cmd) => Some((cmd.x, cmd.width)),
            Self::BlendFill(cmd) => Some((cmd.x, cmd.width)),
            Self::FilterLayer(cmd) => Some((cmd.x, cmd.width)),
            Self::BlendAlphaFill(cmd) => Some((cmd.x, cmd.width)),
            Self::PushLayer | Self::PopBuf | Self::Opacity(_) | Self::Mask(_) => None,
        }
    }

    #[inline(always)]
    pub(crate) fn fill_x(&self) -> u16 {
        match self {
            Self::Fill(cmd) => cmd.x,
            Self::AlphaFill(cmd) => cmd.x,
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
            Self::Fill(cmd) => cmd.width,
            Self::AlphaFill(cmd) => cmd.width,
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct FillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) attrs_idx: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AlphaFillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) alpha_idx: u32,
    pub(crate) attrs_idx: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BlendFillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) blend_mode: BlendMode,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FilterLayerCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) layer_id: usize,
    pub(crate) path_id: u32,
    pub(crate) src_x: u16,
    pub(crate) src_y: u16,
    pub(crate) dst_y_offset: u8,
    pub(crate) height: u8,
}

#[derive(Debug, Clone)]
pub(crate) struct BlendAlphaFillCmd {
    pub(crate) x: u16,
    pub(crate) width: u16,
    pub(crate) alpha_idx: u32,
    pub(crate) thread_idx: u8,
    pub(crate) blend_mode: BlendMode,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GeneratedFill {
    pub(super) x: u16,
    pub(super) width: u16,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct GeneratedAlphaFill {
    pub(super) x: u16,
    pub(super) width: u16,
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
