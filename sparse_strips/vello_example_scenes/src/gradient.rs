// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gradient rendering example scenes.
//! Scenes demonstrating gradient rendering with different extend modes. Taken from Vello Classic
//! test scenes:
//! - `GradientExtendScene`:
//!     - `gradient_extend` method from `https://github.com/linebender/vello/blob/0f3ef03a823eb10b0d7a60164e286cde77ffa222/examples/scenes/src/test_scenes.rs#L815`
//! - `RadialScene`:
//!     - `two_point_radial` method from `https://github.com/linebender/vello/blob/0f3ef03a823eb10b0d7a60164e286cde77ffa222/examples/scenes/src/test_scenes.rs#L882`

use crate::{ExampleScene, RenderingContext};
use smallvec::smallvec;
use vello_common::color::palette::css::{BLACK, BLUE, LIME, RED, WHITE, YELLOW};
use vello_common::kurbo::{Affine, Ellipse, Point, Rect, Shape, Stroke};
use vello_common::peniko::{Color, ColorStop, ColorStops, Extend, Gradient, color::DynamicColor};
use vello_common::peniko::{LinearGradientPosition, RadialGradientPosition, SweepGradientPosition};

/// Gradient scene state
#[derive(Debug, Default)]
pub struct GradientExtendScene {}

impl GradientExtendScene {
    /// Create a new gradient extend scene
    pub fn new() -> Self {
        Self {}
    }
}

impl ExampleScene for GradientExtendScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        enum Kind {
            Linear,
            Radial,
            Sweep,
        }

        /// Helper function to create color stops
        fn create_color_stops(colors: &[Color]) -> ColorStops {
            ColorStops(smallvec![
                ColorStop {
                    offset: 0.0,
                    color: DynamicColor::from_alpha_color(colors[0]),
                },
                ColorStop {
                    offset: 0.5,
                    color: DynamicColor::from_alpha_color(colors[1]),
                },
                ColorStop {
                    offset: 1.0,
                    color: DynamicColor::from_alpha_color(colors[2]),
                },
            ])
        }

        /// Helper function to create a square with a specific gradient type and extend mode
        fn square(ctx: &mut impl RenderingContext, kind: Kind, transform: Affine, extend: Extend) {
            let colors = [RED, LIME, BLUE];
            let width = 300.0;
            let height = 300.0;

            let gradient = match kind {
                Kind::Linear => {
                    let start_x = width * 0.35;
                    let start_y = height * 0.5;
                    let end_x = width * 0.65;
                    let end_y = height * 0.5;

                    Gradient {
                        kind: LinearGradientPosition {
                            start: Point::new(start_x, start_y),
                            end: Point::new(end_x, end_y),
                        }
                        .into(),
                        stops: create_color_stops(&colors),
                        extend,
                        ..Default::default()
                    }
                }
                Kind::Radial => {
                    let center_x = width * 0.5;
                    let center_y = height * 0.5;
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "Width is always positive and bounded"
                    )]
                    let radius = (width * 0.25) as f32;

                    Gradient {
                        kind: RadialGradientPosition {
                            start_center: Point::new(center_x, center_y),
                            start_radius: radius * 0.25,
                            end_center: Point::new(center_x, center_y),
                            end_radius: radius,
                        }
                        .into(),
                        stops: create_color_stops(&colors),
                        extend,
                        ..Default::default()
                    }
                }
                Kind::Sweep => {
                    let center_x = width * 0.5;
                    let center_y = height * 0.5;

                    Gradient {
                        kind: SweepGradientPosition {
                            center: Point::new(center_x, center_y),
                            start_angle: 30.0_f32.to_radians(),
                            end_angle: 150.0_f32.to_radians(),
                        }
                        .into(),
                        stops: create_color_stops(&colors),
                        extend,
                        ..Default::default()
                    }
                }
            };

            ctx.set_transform(transform);
            ctx.set_paint(gradient);
            ctx.fill_rect(&Rect::new(0.0, 0.0, width, height));
        }

        let extend_modes = [Extend::Pad, Extend::Repeat, Extend::Reflect];
        for (x, extend) in extend_modes.iter().enumerate() {
            for (y, kind) in [Kind::Linear, Kind::Radial, Kind::Sweep]
                .into_iter()
                .enumerate()
            {
                let transform = root_transform
                    * Affine::translate((x as f64 * 350.0 + 50.0, y as f64 * 350.0 + 100.0));
                square(ctx, kind, transform, *extend);
            }
        }
    }
}

/// Two-point radial gradient scene
#[derive(Debug, Default)]
pub struct RadialScene;

impl RadialScene {
    /// Create a new radial scene
    pub fn new() -> Self {
        Self {}
    }
}

impl ExampleScene for RadialScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        /// Helper function to create color stops
        fn create_color_stops(colors: &[Color]) -> ColorStops {
            ColorStops(smallvec![
                ColorStop {
                    offset: 0.0,
                    color: DynamicColor::from_alpha_color(colors[0]),
                },
                ColorStop {
                    offset: 0.5,
                    color: DynamicColor::from_alpha_color(colors[1]),
                },
                ColorStop {
                    offset: 1.0,
                    color: DynamicColor::from_alpha_color(colors[2]),
                },
            ])
        }

        /// Helper function to create a two-point radial gradient rectangle
        fn make(
            ctx: &mut impl RenderingContext,
            x0: f64,
            y0: f64,
            r0: f32,
            x1: f64,
            y1: f64,
            r1: f32,
            transform: Affine,
            extend: Extend,
        ) {
            let colors = [RED, YELLOW, Color::from_rgb8(6, 85, 186)];
            let width = 400.0;
            let height = 200.0;

            ctx.set_transform(transform);
            ctx.set_paint(WHITE);
            ctx.fill_rect(&Rect::new(0.0, 0.0, width, height));

            let gradient = Gradient {
                kind: RadialGradientPosition {
                    start_center: Point::new(x0, y0),
                    start_radius: r0,
                    end_center: Point::new(x1, y1),
                    end_radius: r1,
                }
                .into(),
                stops: create_color_stops(&colors),
                extend,
                ..Default::default()
            };

            ctx.set_paint(gradient);
            ctx.fill_rect(&Rect::new(0.0, 0.0, width, height));

            // Draw stroke circles showing the gradient extents
            let r0 = r0 as f64 - 1.0;
            let r1 = r1 as f64 - 1.0;
            ctx.set_paint(BLACK);
            ctx.set_stroke(Stroke::new(1.0));
            ctx.stroke_path(&Ellipse::new((x0, y0), (r0, r0), 0.0).to_path(0.1));
            ctx.stroke_path(&Ellipse::new((x1, y1), (r1, r1), 0.0).to_path(0.1));
        }

        // These demonstrate radial gradient patterns similar to the examples shown
        // at <https://learn.microsoft.com/en-us/typography/opentype/spec/colr#radial-gradients>

        // Row 1: Basic two-point radial gradient
        for (i, mode) in [Extend::Pad, Extend::Repeat, Extend::Reflect]
            .iter()
            .enumerate()
        {
            let y = 100.0;
            let x0 = 140.0;
            let x1 = x0 + 140.0;
            let r0 = 20.0;
            let r1 = 50.0;
            make(
                ctx,
                x0,
                y,
                r0,
                x1,
                y,
                r1,
                root_transform * Affine::translate((i as f64 * 420.0 + 20.0, 20.0)),
                *mode,
            );
        }

        // Row 2: Reversed two-point radial gradient
        for (i, mode) in [Extend::Pad, Extend::Repeat, Extend::Reflect]
            .iter()
            .enumerate()
        {
            let y = 100.0;
            let x0: f64 = 140.0;
            let x1 = x0 + 140.0;
            let r0 = 20.0;
            let r1 = 50.0;
            make(
                ctx,
                x1,
                y,
                r1,
                x0,
                y,
                r0,
                root_transform * Affine::translate((i as f64 * 420.0 + 20.0, 240.0)),
                *mode,
            );
        }

        // Row 3: Equal radii gradient
        for (i, mode) in [Extend::Pad, Extend::Repeat, Extend::Reflect]
            .iter()
            .enumerate()
        {
            let y = 100.0;
            let x0 = 140.0;
            let x1 = x0 + 140.0;
            let r0 = 50.0;
            let r1 = 50.0;
            make(
                ctx,
                x0,
                y,
                r0,
                x1,
                y,
                r1,
                root_transform * Affine::translate((i as f64 * 420.0 + 20.0, 460.0)),
                *mode,
            );
        }

        // Row 4: Overlapping circles
        for (i, mode) in [Extend::Pad, Extend::Repeat, Extend::Reflect]
            .iter()
            .enumerate()
        {
            let x0 = 140.0;
            let y0 = 125.0;
            let r0 = 20.0;
            let x1 = 190.0;
            let y1 = 100.0;
            let r1 = 95.0;
            make(
                ctx,
                x0,
                y0,
                r0,
                x1,
                y1,
                r1,
                root_transform * Affine::translate((i as f64 * 420.0 + 20.0, 680.0)),
                *mode,
            );
        }

        // Row 5: Touching circles
        for (i, mode) in [Extend::Pad, Extend::Repeat, Extend::Reflect]
            .iter()
            .enumerate()
        {
            let x0 = 140.0;
            let y0 = 125.0;
            let r0: f32 = 20.0;
            let x1 = 190.0;
            let y1 = 100.0;
            let r1: f32 = 96.0;
            // Shift p0 so the outer edges of both circles touch
            let direction = Point::new(x0, y0) - Point::new(x1, y1);
            let normalized_direction = direction.normalize();
            let p0 = Point::new(x1, y1) + (normalized_direction * (r1 - r0) as f64);
            make(
                ctx,
                p0.x,
                p0.y,
                r0,
                x1,
                y1,
                r1,
                root_transform * Affine::translate((i as f64 * 420.0 + 20.0, 900.0)),
                *mode,
            );
        }
    }
}
