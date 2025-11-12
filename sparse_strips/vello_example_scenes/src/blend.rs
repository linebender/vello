// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Example compositing an image using blend layers.

use crate::{ExampleScene, RenderingContext};
use vello_common::color::palette::css::{BLUE, GREEN, PURPLE, RED, YELLOW};
use vello_common::kurbo::{Affine, Circle, Point, Rect, Shape};
use vello_common::peniko::{BlendMode, Color, Compose, Mix};

/// Blend scene state
#[derive(Debug)]
pub struct BlendScene {}

impl ExampleScene for BlendScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        render(ctx, root_transform);
    }
}

impl BlendScene {
    /// Create a new `BlendScene`
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for BlendScene {
    fn default() -> Self {
        Self::new()
    }
}

/// Demonstrates a complex compositing scene.
pub fn render(ctx: &mut impl RenderingContext, root_transform: Affine) {
    ctx.set_transform(root_transform);
    ctx.push_layer(
        None,
        Some(BlendMode {
            mix: Mix::Normal,
            compose: Compose::SrcOver,
        }),
        None,
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

    // Layer 1: Simulate clip to center area using compositing
    ctx.push_layer(None, None, None, None, None);
    {
        // Draw the "clip" shape first
        ctx.set_paint(Color::WHITE);
        ctx.fill_path(&Circle::new(Point::new(50.0, 50.0), 40.0).to_path(0.1));

        // Now use SrcIn to clip content to that shape
        ctx.push_layer(
            None,
            Some(BlendMode::new(Mix::Normal, Compose::SrcIn)),
            None,
            None,
            None,
        );
        {
            // Layer 2: Semi-transparent purple overlay
            ctx.push_layer(None, None, Some(0.7), None, None);
            ctx.set_paint(PURPLE);
            ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
            ctx.pop_layer();

            // Layer 3: XOR blend mode with smaller clip simulation
            ctx.push_layer(None, None, None, None, None);
            {
                // Draw the "clip" rectangle
                ctx.set_paint(Color::WHITE);
                ctx.fill_rect(&Rect::new(25.0, 25.0, 75.0, 75.0));

                // Use SrcIn to clip the XOR content
                ctx.push_layer(
                    None,
                    Some(BlendMode::new(Mix::Normal, Compose::SrcIn)),
                    None,
                    None,
                    None,
                );
                {
                    ctx.push_layer(
                        None,
                        Some(BlendMode::new(Mix::Normal, Compose::Xor)),
                        None,
                        None,
                        None,
                    );

                    // Draw overlapping circles with XOR
                    ctx.set_paint(RED);
                    ctx.fill_path(&Circle::new(Point::new(40.0, 40.0), 20.0).to_path(0.1));
                    ctx.set_paint(BLUE);
                    ctx.fill_path(&Circle::new(Point::new(60.0, 40.0), 20.0).to_path(0.1));

                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();

            // Layer 4: Nested opacity with Plus blend
            ctx.push_layer(None, None, Some(0.5), None, None);
            ctx.push_layer(
                None,
                Some(BlendMode::new(Mix::Normal, Compose::Plus)),
                None,
                None,
                None,
            );
            {
                ctx.set_paint(YELLOW);
                ctx.fill_path(&Circle::new(Point::new(50.0, 60.0), 15.0).to_path(0.1));

                // Layer 5: "Clip" simulation with SrcIn
                ctx.push_layer(None, None, None, None, None);
                {
                    ctx.set_paint(Color::WHITE);
                    ctx.fill_path(&Circle::new(Point::new(50.0, 50.0), 30.0).to_path(0.1));

                    // Use SrcIn twice (once for the clip, once for the original SrcIn)
                    ctx.push_layer(
                        None,
                        Some(BlendMode::new(Mix::Normal, Compose::SrcIn)),
                        None,
                        None,
                        None,
                    );
                    {
                        ctx.push_layer(
                            None,
                            Some(BlendMode::new(Mix::Normal, Compose::SrcIn)),
                            None,
                            None,
                            None,
                        );
                        {
                            ctx.set_paint(GREEN);
                            ctx.fill_rect(&Rect::new(35.0, 35.0, 65.0, 65.0));
                        }
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
            ctx.pop_layer();

            // Layer 6: Final overlay with DestOut to create a hole
            ctx.push_layer(
                None,
                Some(BlendMode::new(Mix::Normal, Compose::DestOut)),
                None,
                None,
                None,
            );
            ctx.set_paint(Color::BLACK);
            ctx.fill_path(&Circle::new(Point::new(50.0, 50.0), 10.0).to_path(0.1));
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }
    ctx.pop_layer();

    ctx.pop_layer();
}
