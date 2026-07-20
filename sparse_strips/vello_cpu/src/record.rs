// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko::BlendMode;
use core::ops::Range;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::Paint;
use vello_common::record::Drawable;
use vello_common::strip::Strip;
use vello_common::util::strip_bbox;

#[derive(Debug)]
pub(crate) struct RecordedFill {
    pub(crate) thread_idx: u8,
    pub(crate) strip_range: Range<usize>,
    pub(crate) paint: Paint,
    pub(crate) blend_mode: BlendMode,
    pub(crate) mask: Option<Mask>,
}

impl RecordedFill {
    pub(crate) fn new(
        thread_idx: u8,
        strip_range: Range<usize>,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    ) -> Self {
        Self {
            thread_idx,
            strip_range,
            paint,
            blend_mode,
            mask,
        }
    }
}

impl Drawable for RecordedFill {
    fn bbox(&self, strips: &[Strip]) -> RectU16 {
        strip_bbox(strips)
    }
}
