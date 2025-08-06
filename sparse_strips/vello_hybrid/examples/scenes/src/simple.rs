// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple example scene with basic shapes.

use parley::Rect;
use vello_common::color::palette::css::{BLUE, WHITE, YELLOW};
use vello_common::kurbo::{Affine, BezPath, Stroke};
use vello_common::peniko::color::palette;
use vello_common::kurbo::Shape;
use vello_common::peniko::{BlendMode, Compose, Mix};
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
pub fn render(ctx: &mut Scene, _root_transform: Affine) {
    let path = Rect::new(0.0, 0.0, 100 as f64, 100 as f64).to_path(0.1);

    ctx.set_paint(WHITE);
    ctx.fill_path(&path);

    ctx.push_layer(
        None,
        Some(BlendMode::new(Mix::Normal, Compose::SrcOver)),
        None,
        None,
    );

    // Draw the destination layer.
    ctx.set_paint(YELLOW.with_alpha(1.0));
    ctx.fill_rect(&Rect::new(10.0, 10.0, 70.0, 70.0));
    // Draw the source layer.
    ctx.push_layer(
        None,
        Some(BlendMode::new(Mix::Normal, Compose::Xor)),
        None,
        None,
    );
    ctx.set_paint(BLUE.with_alpha(1.0));
    ctx.fill_rect(&Rect::new(30.0, 30.0, 90.0, 90.0));
    // Compose.
    ctx.pop_layer();
    ctx.pop_layer();
}
