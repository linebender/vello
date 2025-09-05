// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#[cfg(feature = "multithreading")]
pub(crate) mod multi_threaded;
pub(crate) mod single_threaded;

use crate::RenderMode;
use crate::kurbo::{Affine, BezPath, Stroke};
use crate::peniko::{BlendMode, Fill};
use alloc::vec::Vec;
use core::fmt::Debug;
use vello_common::coarse::Wide;
use vello_common::encode::EncodedPaint;
use vello_common::mask::Mask;
use vello_common::paint::Paint;

pub(crate) trait Dispatcher: Debug + Send + Sync {
    fn wide(&self) -> &Wide;
    fn wide_mut(&mut self) -> &mut Wide;
    fn fill_path(
        &mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    );
    fn stroke_path(
        &mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        paint: Paint,
        aliasing_threshold: Option<u8>,
    );
    fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        fill_rule: Fill,
        clip_transform: Affine,
        blend_mode: BlendMode,
        opacity: f32,
        aliasing_threshold: Option<u8>,
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
    fn alpha_buf(&self) -> &[u8];
    fn extend_alpha_buf(&mut self, alphas: &[u8]);
    fn replace_alpha_buf(&mut self, alphas: Vec<u8>) -> Vec<u8>;
    fn set_alpha_buf(&mut self, alphas: Vec<u8>);
}
