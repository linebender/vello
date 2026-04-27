// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::Scene;
use vello_common::filter_effects::Filter;
use vello_common::kurbo::{Affine, BezPath, Rect};
use vello_common::paint::PaintType;
use vello_common::peniko::BlendMode;

#[cfg(feature = "probe")]
impl vello_common::probe::ProbeRenderer for Scene {
    fn set_transform(&mut self, transform: Affine) {
        Self::set_transform(self, transform);
    }

    fn set_paint(&mut self, paint: PaintType) {
        Self::set_paint(self, paint);
    }

    fn fill_path(&mut self, path: &BezPath) {
        Self::fill_path(self, path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        Self::fill_rect(self, rect);
    }

    fn push_layer(&mut self, blend_mode: Option<BlendMode>, opacity: Option<f32>) {
        Self::push_layer(self, None, blend_mode, opacity, None, None);
    }

    fn push_filter_layer(&mut self, filter: Filter) {
        Self::push_filter_layer(self, filter);
    }

    fn pop_layer(&mut self) {
        Self::pop_layer(self);
    }

    fn set_paint_transform(&mut self, paint_transform: Affine) {
        Self::set_paint_transform(self, paint_transform);
    }

    fn reset_paint_transform(&mut self) {
        Self::reset_paint_transform(self);
    }
}
