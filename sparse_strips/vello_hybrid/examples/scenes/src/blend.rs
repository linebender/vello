// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Clip example showing deeply nested clipping.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

use crate::ExampleScene;
use vello_common::kurbo::{Affine, Shape};
use vello_hybrid::Scene;

/// Clip scene state
#[derive(Debug)]
pub struct BlendScene {}

impl ExampleScene for BlendScene {
    fn render(&mut self, ctx: &mut Scene, root_transform: Affine) {
        render(ctx, root_transform);
    }
}

impl BlendScene {
    /// Create a new `ClipScene`
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for BlendScene {
    fn default() -> Self {
        Self::new()
    }
}

/// Draws a deeply nested clip of circles.
pub fn render(ctx: &mut Scene, root_transform: Affine) {
    use vello_common::color::palette::css::{BLUE, GREEN, PURPLE, RED, YELLOW};
    use vello_common::kurbo::{Circle, Point, Rect};
    use vello_common::peniko::{BlendMode, Color, Compose, Mix};

    ctx.set_transform(root_transform);
    ctx.push_layer(
        None,
        Some(BlendMode {
            mix: Mix::Normal,
            compose: Compose::SrcOver,
        }),
        None,
        None,
    );

    ctx.set_paint(Color::from_rgb8(240, 240, 240));
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));

    // Base layer circles to shine through
    ctx.set_paint(YELLOW);
    ctx.fill_path(&Circle::new(Point::new(30.0, 30.0), 25.0).to_path(0.1));
    ctx.set_paint(BLUE);
    ctx.fill_path(&Circle::new(Point::new(70.0, 30.0), 25.0).to_path(0.1));
    ctx.set_paint(RED);
    ctx.fill_path(&Circle::new(Point::new(30.0, 70.0), 25.0).to_path(0.1));
    ctx.set_paint(GREEN);
    ctx.fill_path(&Circle::new(Point::new(70.0, 70.0), 25.0).to_path(0.1));

    // Layer 1: Clip to center area
    let clip1 = Circle::new(Point::new(50.0, 50.0), 40.0).to_path(0.1);
    ctx.push_clip_layer(&clip1);
    {
        // Layer 2: Semi-transparent purple overlay
        ctx.push_layer(None, None, Some(0.7), None);
        ctx.set_paint(PURPLE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
        ctx.pop_layer();

        // Layer 3: XOR blend mode with smaller clip
        let clip2 = Rect::new(25.0, 25.0, 75.0, 75.0).to_path(0.1);
        ctx.push_clip_layer(&clip2);
        ctx.push_layer(
            None,
            Some(BlendMode::new(Mix::Normal, Compose::Xor)),
            None,
            None,
        );

        // Draw overlapping circles with XOR
        ctx.set_paint(RED);
        ctx.fill_path(&Circle::new(Point::new(40.0, 40.0), 20.0).to_path(0.1));
        ctx.set_paint(BLUE);
        ctx.fill_path(&Circle::new(Point::new(60.0, 40.0), 20.0).to_path(0.1));

        ctx.pop_layer();

        // Layer 4: Nested opacity with Plus blend
        ctx.push_layer(None, None, Some(0.5), None);
        ctx.push_layer(
            None,
            Some(BlendMode::new(Mix::Normal, Compose::Plus)),
            None,
            None,
        );
        {
            ctx.set_paint(YELLOW);
            ctx.fill_path(&Circle::new(Point::new(50.0, 60.0), 15.0).to_path(0.1));

            // Layer 5: Another clip with SrcIn
            let clip3 = Circle::new(Point::new(50.0, 50.0), 30.0).to_path(0.1);
            ctx.push_clip_layer(&clip3);
            ctx.push_layer(
                None,
                Some(BlendMode::new(Mix::Normal, Compose::SrcIn)),
                None,
                None,
            );
            {
                ctx.set_paint(GREEN);
                ctx.fill_rect(&Rect::new(35.0, 35.0, 65.0, 65.0));
            }
            ctx.pop_layer();
            ctx.pop_layer();
        }
        ctx.pop_layer();
        ctx.pop_layer();

        ctx.pop_layer();

        // Layer 6: Final overlay with DestOut to create a hole
        ctx.push_layer(
            None,
            Some(BlendMode::new(Mix::Normal, Compose::DestOut)),
            None,
            None,
        );
        ctx.set_paint(Color::BLACK);
        ctx.fill_path(&Circle::new(Point::new(50.0, 50.0), 10.0).to_path(0.1));
        ctx.pop_layer();
    }
    ctx.pop_layer();

    ctx.pop_layer();
}
