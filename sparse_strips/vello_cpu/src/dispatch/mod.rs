// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(feature = "multithreading")]
pub(crate) mod multi_threaded;
pub(crate) mod single_threaded;

use crate::RasterizerSettings;
use crate::kurbo::{Affine, BezPath, Rect, Stroke};
use crate::peniko::{BlendMode, Fill};
use core::fmt::Debug;
use core::ops::Range;
use vello_common::encode::EncodedPaint;
use vello_common::filter::FilterData;
use vello_common::geometry::RectU16;
use vello_common::mask::Mask;
use vello_common::paint::{ImageResolver, Paint};
use vello_common::pixmap::PixmapMut;
use vello_common::record::Drawable;
use vello_common::strip::Strip;
use vello_common::util::strip_bbox;

pub(crate) type Node = vello_common::record::Node;
pub(crate) type RecordedLayer = vello_common::record::RecordedLayer;

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

pub(crate) trait Dispatcher: Debug + Send {
    fn has_layers(&self) -> bool;
    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    );
    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        blend_mode: BlendMode,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
    );
    /// Fill a pixel-aligned rectangle with the current paint.
    fn fill_rect_fast(
        &mut self,
        rect: &Rect,
        paint: Paint,
        blend_mode: BlendMode,
        mask: Option<Mask>,
    );
    fn push_clip_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
    );
    fn pop_clip_path(&mut self);
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        aliasing_threshold: Option<u8>,
        mask: Option<Mask>,
        filter_data: Option<FilterData>,
    );
    fn pop_layer(&mut self);
    fn reset(&mut self, width: u16, height: u16);
    fn flush(&mut self);
    fn rasterize(
        &self,
        target: PixmapMut<'_>,
        scene_width: u16,
        scene_height: u16,
        settings: RasterizerSettings,
        encoded_paints: &[EncodedPaint],
        image_resolver: &dyn ImageResolver,
    );
    fn is_multi_threaded(&self) -> bool;
}
