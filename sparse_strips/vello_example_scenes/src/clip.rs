// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Clip example showing deeply nested clipping.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

use crate::{ExampleScene, RenderingContext};
use vello_common::color::palette::css::{
    BLACK, BLUE, DARK_BLUE, DARK_GREEN, GREEN, REBECCA_PURPLE, RED,
};
use vello_common::kurbo::{Affine, BezPath, Circle, Point, Rect, Shape, Stroke};
use vello_common::peniko::Color;

/// Clip scene state
#[derive(Debug)]
pub struct ClipScene {
    use_clip_path: bool,
    num_circles: usize,
}

impl ExampleScene for ClipScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        render(ctx, root_transform, self.use_clip_path, self.num_circles);
    }

    fn handle_key(&mut self, key: &str) -> bool {
        match key {
            "c" | "C" => {
                self.toggle_clip();
                true
            }
            "m" | "M" => {
                self.add_circle();
                true
            }
            _ => false,
        }
    }
}

impl ClipScene {
    /// Create a new `ClipScene`
    pub fn new() -> Self {
        Self {
            use_clip_path: false,
            num_circles: 1,
        }
    }

    /// Toggle using clip path
    pub fn toggle_clip(&mut self) {
        self.use_clip_path = !self.use_clip_path;
        println!("Use clip path: {}", self.use_clip_path);
    }

    /// Add another circle to the scene
    pub fn add_circle(&mut self) {
        self.num_circles += 1;
        println!("Number of circles: {}", self.num_circles);
    }
}

impl Default for ClipScene {
    fn default() -> Self {
        Self::new()
    }
}

fn draw_clipping_outline(ctx: &mut impl RenderingContext, path: &BezPath) {
    let stroke = Stroke::new(1.0);
    ctx.set_paint(DARK_BLUE);
    ctx.set_stroke(stroke);
    ctx.stroke_path(path);
}

/// Draws a deeply nested clip of circles.
pub fn render(
    ctx: &mut impl RenderingContext,
    root_transform: Affine,
    use_clip_path: bool,
    num_circles: usize,
) {
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

    const SPACING: f64 = 120.0;
    const BASE_X: f64 = 50.0;
    const BASE_Y: f64 = 50.0;

    ctx.set_transform(root_transform);

    // Draw multiple circles in a checkerboard pattern
    for circle_idx in 0..num_circles {
        // Calculate checkerboard position
        // Create a grid pattern where circles are placed in a checkerboard layout
        let row = circle_idx / 4;
        let col = circle_idx % 4;

        // Create checkerboard offset pattern
        let offset_x = if (row + col) % 2 == 0 {
            0.0
        } else {
            SPACING / 2.0
        };
        let x = BASE_X + col as f64 * SPACING + offset_x;
        let y = BASE_Y + row as f64 * SPACING;

        let center = Point::new(x, y);
        let cover_rect = Rect::new(x - 50.0, y - 50.0, x + 50.0, y + 50.0);
        let mut radius = INITIAL_RADIUS;

        for _ in 0..outer_count {
            for color in COLORS.iter() {
                let clip_circle = Circle::new(center, radius).to_path(0.1);
                draw_clipping_outline(ctx, &clip_circle);
                if use_clip_path {
                    ctx.push_clip_path(&clip_circle);
                } else {
                    ctx.push_clip_layer(&clip_circle);
                }

                ctx.set_paint(*color);
                ctx.fill_rect(&cover_rect);

                radius -= RADIUS_DECREMENT;
            }
        }

        for _ in 0..outer_count {
            for _ in COLORS.iter() {
                if !use_clip_path {
                    ctx.pop_layer();
                } else {
                    ctx.pop_clip_path();
                }
            }
        }
    }
}
