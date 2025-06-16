// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(feature = "multithreading")]
pub(crate) mod multi_threaded;
pub(crate) mod single_threaded;

use crate::RenderMode;
use crate::kurbo::{Affine, BezPath, Stroke};
use crate::peniko::{BlendMode, Fill};
use core::fmt::Debug;
use vello_common::coarse::Wide;
use vello_common::encode::EncodedPaint;
use vello_common::mask::Mask;
use vello_common::paint::Paint;

pub(crate) trait Dispatcher: Debug {
    fn wide(&self) -> &Wide;
    fn fill_path(&mut self, path: &BezPath, fill_rule: Fill, transform: Affine, paint: Paint);
    fn stroke_path(&mut self, path: &BezPath, stroke: &Stroke, transform: Affine, paint: Paint);
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        mask: Option<Mask>,
    );
    fn pop_layer(&mut self);
    fn reset(&mut self);
    fn flush(&mut self);
    fn rasterize(
        &self,
        buffer: &mut [u8],
        render_mode: RenderMode,
        width: u16,
        height: u16,
        encoded_paints: &[EncodedPaint],
    );
}
