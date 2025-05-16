// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Clip example showing deeply nested clipping.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

use crate::ExampleScene;
use parley::Rect;
use vello_common::color::palette::css::{
    BLACK, BLUE, DARK_BLUE, DARK_GREEN, GREEN, REBECCA_PURPLE, RED,
};
use vello_common::kurbo::{Affine, BezPath, Circle, Point, Shape, Stroke};
use vello_common::peniko::Color;
use vello_hybrid::Scene;

/// Clip scene state
#[derive(Debug)]
pub struct ClipScene {}

impl ExampleScene for ClipScene {
    fn render(&mut self, ctx: &mut Scene, root_transform: Affine) {
        render(ctx, root_transform);
    }
}

impl ClipScene {
    /// Create a new `ClipScene`
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ClipScene {
    fn default() -> Self {
        Self::new()
    }
}

fn draw_clipping_outline(ctx: &mut Scene, path: &BezPath) {
    let stroke = Stroke::new(1.0);
    ctx.set_paint(DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(path);
}

/// Draws a deeply nested clip of circles.
pub fn render(ctx: &mut Scene, root_transform: Affine) {
    const INITIAL_RADIUS: f64 = 48.0;
    const RADIUS_DECREMENT: f64 = 2.5;
    const INNER_COUNT: usize = 10;
    // `.ceil()` is not constant-evaluatable, so we have to do this at runtime.
    let outer_count: usize =
        (INITIAL_RADIUS / RADIUS_DECREMENT / INNER_COUNT as f64).ceil() as usize;
    const COLORS: [Color; INNER_COUNT] = [
        RED,
        DARK_BLUE,
        DARK_GREEN,
        REBECCA_PURPLE,
        BLACK,
        BLUE,
        GREEN,
        RED,
        DARK_BLUE,
        DARK_GREEN,
    ];

    const COVER_RECT: Rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    const CENTER: Point = Point::new(50.0, 50.0);
    let mut radius = INITIAL_RADIUS;

    ctx.set_transform(root_transform);
    for _ in 0..outer_count {
        for color in COLORS.iter() {
            let clip_circle = Circle::new(CENTER, radius).to_path(0.1);
            draw_clipping_outline(ctx, &clip_circle);
            ctx.push_clip_layer(&clip_circle);

            ctx.set_paint((*color).into());
            ctx.fill_rect(&COVER_RECT);

            radius -= RADIUS_DECREMENT;
        }
    }
    for _ in 0..outer_count {
        for _ in COLORS.iter() {
            ctx.pop_layer();
        }
    }
}
