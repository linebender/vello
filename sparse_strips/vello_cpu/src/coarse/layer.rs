// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko::BlendMode;
use alloc::vec::Vec;
use core::ops::Range;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;

#[derive(Debug, Clone)]
pub(crate) struct LayerClip {
    pub(crate) strip_range: Range<usize>,
    pub(crate) thread_idx: u8,
    pub(crate) bbox: RectU16,
}

#[derive(Debug, Clone)]
pub(super) struct ActiveLayer {
    pub(super) mask: Option<Mask>,
    pub(super) blend_mode: BlendMode,
    pub(super) opacity: f32,
    pub(super) clip: Option<LayerClip>,
    pub(super) bbox: RectU16,
    pub(super) occupied_rows: Vec<usize>,
}
