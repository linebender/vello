// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple example scene with basic shapes.

use vello_common::color::palette::css::RED;
use vello_common::kurbo::{Affine, Rect};
use vello_common::peniko::color::palette;
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
pub fn render(ctx: &mut Scene, root_transform: Affine) {
    let rect = Rect::new(0.0, 0.0, 128.0, 4.0);

    ctx.set_paint(palette::css::BLUE);
    ctx.fill_rect(&rect);
    ctx.push_blend_layer(BlendMode::new(Mix::Multiply, Compose::SrcOver));
    ctx.set_paint(RED);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}
