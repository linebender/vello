// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple example scene with basic shapes.

use vello_common::kurbo::{Affine, BezPath, Stroke};
use vello_common::peniko::color::palette;
use vello_hybrid::Scene;

use crate::ExampleScene;

/// Simple scene state
#[derive(Debug)]
pub struct SimpleScene {}

impl ExampleScene for SimpleScene {
    fn render(&mut self, ctx: &mut Scene, root_transform: Affine) {
        render(ctx, root_transform);
    }
}

impl SimpleScene {
    /// Create a new `SimpleScene`
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for SimpleScene {
    fn default() -> Self {
        Self::new()
    }
}

/// Draws a simple scene with shapes
pub fn render(ctx: &mut Scene, root_transform: Affine) {
    let mut path = BezPath::new();
    path.move_to((10.0, 10.0));
    path.line_to((180.0, 20.0));
    path.line_to((30.0, 40.0));
    path.close_path();

    // Use a combined transform that includes the root transform
    let scene_transform = Affine::scale(5.0);
    ctx.set_transform(root_transform * scene_transform);

    ctx.set_paint(palette::css::REBECCA_PURPLE);
    ctx.fill_path(&path);
    let stroke = Stroke::new(1.0);
    ctx.set_paint(palette::css::DARK_BLUE);
    ctx.set_stroke(stroke);
    ctx.stroke_path(&path);
}
