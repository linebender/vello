// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Path rendering example scenes.
//! Scenes demonstrating various path rendering techniques including strokes, fills, and tricky paths.
//! Adapted from Vello Classic test scenes:
//! - `stroke_styles`, `stroke_styles_non_uniform`, `stroke_styles_skew` methods
//! - `funky_paths` method
//! - `tricky_strokes` method  
//! - `fill_types` method
//! - `robust_paths` method

use crate::{ExampleScene, RenderingContext};
use vello_common::color::palette::css::{AQUA, BLUE, GRAY, LIME, YELLOW};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Point, Rect, Shape, Stroke};
use vello_common::peniko::{Color, Fill};

/// Stroke styles scene state
#[derive(Debug, Default)]
pub struct StrokeStylesScene {
    transform: Affine,
}

impl StrokeStylesScene {
    /// Create a new stroke styles scene
    pub fn new() -> Self {
        Self {
            transform: Affine::IDENTITY,
        }
    }

    /// Create a new stroke styles scene with non-uniform scale
    pub fn new_non_uniform() -> Self {
        Self {
            transform: Affine::scale_non_uniform(1.2, 0.7),
        }
    }

    /// Create a new stroke styles scene with skew
    pub fn new_skew() -> Self {
        Self {
            transform: Affine::skew(1.0, 0.0),
        }
    }
}

const Y_OFFSET: f64 = 180.0;
const X_OFFSET: f64 = 450.0;

impl ExampleScene for StrokeStylesScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        let colors = [
            Color::from_rgb8(140, 181, 236),
            Color::from_rgb8(246, 236, 202),
            Color::from_rgb8(201, 147, 206),
            Color::from_rgb8(150, 195, 160),
        ];

        // Create path data using BezPath
        let mut simple_stroke = BezPath::new();
        simple_stroke.move_to((0.0, 0.0));
        simple_stroke.line_to((100.0, 0.0));

        let mut join_stroke = BezPath::new();
        join_stroke.move_to((0.0, 0.0));
        join_stroke.curve_to((20.0, 0.0), (42.5, 5.0), (50.0, 25.0));
        join_stroke.curve_to((57.5, 5.0), (80.0, 0.0), (100.0, 0.0));

        let mut miter_stroke = BezPath::new();
        miter_stroke.move_to((0.0, 0.0));
        miter_stroke.line_to((90.0, 16.0));
        miter_stroke.line_to((0.0, 31.0));
        miter_stroke.line_to((90.0, 46.0));

        let mut closed_strokes = BezPath::new();
        closed_strokes.move_to((0.0, 0.0));
        closed_strokes.line_to((90.0, 21.0));
        closed_strokes.line_to((0.0, 42.0));
        closed_strokes.close_path();
        closed_strokes.move_to((200.0, 0.0));
        closed_strokes.curve_to((100.0, 72.0), (300.0, 72.0), (200.0, 0.0));
        closed_strokes.close_path();
        closed_strokes.move_to((290.0, 0.0));
        closed_strokes.curve_to((200.0, 72.0), (400.0, 72.0), (310.0, 0.0));
        closed_strokes.close_path();

        let cap_styles = [Cap::Butt, Cap::Square, Cap::Round];
        let join_styles = [Join::Bevel, Join::Miter, Join::Round];
        let miter_limits = [4.0, 6.0, 0.1, 10.0];

        // Simple strokes with cap combinations
        let t = Affine::translate((60.0, 40.0)) * Affine::scale(2.0);
        let mut y = 0.0;
        let mut color_idx = 0;
        for start in cap_styles {
            for end in cap_styles {
                let stroke = Stroke::new(20.0).with_start_cap(start).with_end_cap(end);
                ctx.set_transform(
                    root_transform * Affine::translate((0.0, y + 30.0)) * t * self.transform,
                );
                ctx.set_paint(colors[color_idx]);
                ctx.set_stroke(stroke);
                ctx.stroke_path(&simple_stroke);
                y += Y_OFFSET;
                color_idx = (color_idx + 1) % colors.len();
            }
        }

        // Dashed strokes with cap combinations
        let t = Affine::translate((X_OFFSET, 0.0)) * t;
        y = 0.0;
        for start in cap_styles {
            for end in cap_styles {
                let stroke = Stroke::new(20.0)
                    .with_start_cap(start)
                    .with_end_cap(end)
                    .with_dashes(0.0, [10.0, 21.0]);
                ctx.set_transform(
                    root_transform * Affine::translate((0.0, y + 30.0)) * t * self.transform,
                );
                ctx.set_paint(colors[color_idx]);
                ctx.set_stroke(stroke);
                ctx.stroke_path(&simple_stroke);
                y += Y_OFFSET;
                color_idx = (color_idx + 1) % colors.len();
            }
        }

        // Cap and join combinations
        let t = Affine::translate((X_OFFSET, 0.0)) * t;
        y = 0.0;
        for cap in cap_styles {
            for join in join_styles {
                let stroke = Stroke::new(20.0).with_caps(cap).with_join(join);
                ctx.set_transform(
                    root_transform * Affine::translate((0.0, y + 30.0)) * t * self.transform,
                );
                ctx.set_paint(colors[color_idx]);
                ctx.set_stroke(stroke);
                ctx.stroke_path(&join_stroke);
                y += Y_OFFSET;
                color_idx = (color_idx + 1) % colors.len();
            }
        }

        // Miter limit
        let t = Affine::translate((X_OFFSET, 0.0)) * t;
        y = 0.0;
        for ml in miter_limits {
            let stroke = Stroke::new(10.0)
                .with_caps(Cap::Butt)
                .with_join(Join::Miter)
                .with_miter_limit(ml);
            ctx.set_transform(
                root_transform * Affine::translate((0.0, y + 30.0)) * t * self.transform,
            );
            ctx.set_paint(colors[color_idx]);
            ctx.set_stroke(stroke);
            ctx.stroke_path(&miter_stroke);
            y += Y_OFFSET;
            color_idx = (color_idx + 1) % colors.len();
        }

        // Closed paths
        for (i, join) in join_styles.iter().enumerate() {
            // The cap style is not important since a closed path shouldn't have any caps.
            let stroke = Stroke::new(10.0)
                .with_caps(cap_styles[i])
                .with_join(*join)
                .with_miter_limit(5.0);
            ctx.set_transform(
                root_transform * Affine::translate((0.0, y + 30.0)) * t * self.transform,
            );
            ctx.set_paint(colors[color_idx]);
            ctx.set_stroke(stroke);
            ctx.stroke_path(&closed_strokes);
            y += Y_OFFSET;
            color_idx = (color_idx + 1) % colors.len();
        }
    }
}

/// Funky paths scene state
#[derive(Debug, Default)]
pub struct FunkyPathsScene {}

impl FunkyPathsScene {
    /// Create a new funky paths scene
    pub fn new() -> Self {
        Self {}
    }
}

// TODO: fix issue https://github.com/linebender/vello/issues/1240
impl ExampleScene for FunkyPathsScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        // Missing movetos path
        let mut missing_movetos = BezPath::new();
        missing_movetos.move_to((0.0, 0.0));
        missing_movetos.line_to((100.0, 100.0));
        missing_movetos.line_to((100.0, 200.0));
        missing_movetos.close_path();
        missing_movetos.line_to((0.0, 400.0));
        missing_movetos.line_to((100.0, 400.0));

        // Only movetos path
        let mut only_movetos = BezPath::new();
        only_movetos.move_to((0.0, 0.0));
        only_movetos.move_to((100.0, 100.0));

        // Empty path
        let empty = BezPath::new();

        ctx.set_transform(root_transform * Affine::translate((100.0, 100.0)));
        ctx.set_paint(BLUE);
        ctx.fill_path(&missing_movetos);

        ctx.set_transform(root_transform);
        ctx.set_paint(BLUE);
        ctx.fill_path(&empty);

        ctx.set_transform(root_transform);
        ctx.set_paint(BLUE);
        ctx.fill_path(&only_movetos);

        ctx.set_transform(root_transform * Affine::translate((100.0, 100.0)));
        ctx.set_paint(AQUA);
        ctx.set_stroke(Stroke::new(8.0));
        ctx.stroke_path(&missing_movetos);
    }
}

/// Tricky strokes scene state
#[derive(Debug, Default)]
pub struct TrickyStrokesScene {}

impl TrickyStrokesScene {
    /// Create a new tricky strokes scene
    pub fn new() -> Self {
        Self {}
    }
}

impl ExampleScene for TrickyStrokesScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        let colors = [
            Color::from_rgb8(140, 181, 236),
            Color::from_rgb8(246, 236, 202),
            Color::from_rgb8(201, 147, 206),
            Color::from_rgb8(150, 195, 160),
        ];

        const CELL_SIZE: f64 = 200.0;
        const STROKE_WIDTH: f64 = 30.0;
        const NUM_COLS: usize = 5;

        fn stroke_bounds(pts: &[(f64, f64); 4]) -> Rect {
            use vello_common::kurbo::CubicBez;
            CubicBez::new(pts[0], pts[1], pts[2], pts[3])
                .bounding_box()
                .inflate(STROKE_WIDTH, STROKE_WIDTH)
        }

        fn map_rect_to_rect(src: &Rect, dst: &Rect) -> (Affine, f64) {
            let (scale, x_larger) = {
                let sx = dst.width() / src.width();
                let sy = dst.height() / src.height();
                (sx.min(sy), sx > sy)
            };
            let tx = dst.x0 - src.x0 * scale;
            let ty = dst.y0 - src.y0 * scale;
            let (tx, ty) = if x_larger {
                (tx + 0.5 * (dst.width() - src.width() * scale), ty)
            } else {
                (tx, ty + 0.5 * (dst.height() - src.height() * scale))
            };
            (Affine::new([scale, 0.0, 0.0, scale, tx, ty]), scale)
        }

        let tricky_cubics = [
            [(122., 737.), (348., 553.), (403., 761.), (400., 760.)],
            [(244., 520.), (244., 518.), (1141., 634.), (394., 688.)],
            [(550., 194.), (138., 130.), (1035., 246.), (288., 300.)],
            [(226., 733.), (556., 779.), (-43., 471.), (348., 683.)],
            [(268., 204.), (492., 304.), (352., 23.), (433., 412.)],
            [(172., 480.), (396., 580.), (256., 299.), (338., 677.)],
            [(731., 340.), (318., 252.), (1026., -64.), (367., 265.)],
            [(475., 708.), (62., 620.), (770., 304.), (220., 659.)],
            [(0., 0.), (128., 128.), (128., 0.), (0., 128.)], // Perfect cusp
            [(0., 0.01), (128., 127.999), (128., 0.01), (0., 127.99)], // Near-cusp
            [(0., -0.01), (128., 128.001), (128., -0.01), (0., 128.001)], // Near-cusp
            [(0., 0.), (0., -10.), (0., -10.), (0., 10.)],    // Flat line with 180
            [(10., 0.), (0., 0.), (20., 0.), (10., 0.)],      // Flat line with 2 180s
            [(39., -39.), (40., -40.), (40., -40.), (0., 0.)], // Flat diagonal with 180
            [(40., 40.), (0., 0.), (200., 200.), (0., 0.)],   // Diag w/ an internal 180
            [(0., 0.), (1e-2, 0.), (-1e-2, 0.), (0., 0.)],    // Circle
            // Flat line with no turns:
            [
                (400.75, 100.05),
                (400.75, 100.05),
                (100.05, 300.95),
                (100.05, 300.95),
            ],
            [(0.5, 0.), (0., 0.), (20., 0.), (10., 0.)], // Flat line with 2 180s
            [(10., 0.), (0., 0.), (10., 0.), (10., 0.)], // Flat line with a 180
        ];

        let mut idx = 0;
        let mut color_idx = 0;
        for (i, cubic) in tricky_cubics.into_iter().enumerate() {
            idx += 1;
            let x = (i % NUM_COLS) as f64 * CELL_SIZE;
            let y = (i / NUM_COLS) as f64 * CELL_SIZE;
            let cell = Rect::new(x, y, x + CELL_SIZE, y + CELL_SIZE);
            let bounds = stroke_bounds(&cubic);
            let (t, s) = map_rect_to_rect(&bounds, &cell);

            let mut path = BezPath::new();
            path.move_to(cubic[0]);
            path.curve_to(cubic[1], cubic[2], cubic[3]);

            ctx.set_transform(root_transform * t);
            ctx.set_paint(colors[color_idx]);
            ctx.set_stroke(
                Stroke::new(STROKE_WIDTH / s)
                    .with_caps(Cap::Butt)
                    .with_join(Join::Miter),
            );
            ctx.stroke_path(&path);
            color_idx = (color_idx + 1) % colors.len();
        }

        // Add some flat curves as well
        let flat_quad = [[(2., 1.), (1., 1.)]];
        let flat_curves = [flat_quad.as_slice()];

        for quads in flat_curves.iter() {
            let mut path = BezPath::new();
            path.move_to((1.0, 1.0));
            for quad in quads.iter() {
                path.quad_to(quad[0], quad[1]);
            }
            let x = (idx % NUM_COLS) as f64 * CELL_SIZE;
            let y = (idx / NUM_COLS) as f64 * CELL_SIZE;
            let cell = Rect::new(x, y, x + CELL_SIZE, y + CELL_SIZE);
            let bounds = path.bounding_box().inflate(STROKE_WIDTH, STROKE_WIDTH);
            let (t, s) = map_rect_to_rect(&bounds, &cell);

            ctx.set_transform(root_transform * t);
            ctx.set_paint(colors[color_idx]);
            ctx.set_stroke(
                Stroke::new(STROKE_WIDTH / s)
                    .with_caps(Cap::Butt)
                    .with_join(Join::Miter),
            );
            ctx.stroke_path(&path);
            color_idx = (color_idx + 1) % colors.len();
            idx += 1;
        }
    }
}

/// Fill types scene state
#[derive(Debug, Default)]
pub struct FillTypesScene {}

impl FillTypesScene {
    /// Create a new fill types scene
    pub fn new() -> Self {
        Self {}
    }
}

impl ExampleScene for FillTypesScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        let rect = Rect::from_origin_size(Point::new(0.0, 0.0), (500.0, 500.0));

        // Create star path
        let mut star = BezPath::new();
        star.move_to((250.0, 0.0));
        star.line_to((105.0, 450.0));
        star.line_to((490.0, 175.0));
        star.line_to((10.0, 175.0));
        star.line_to((395.0, 450.0));
        star.close_path();

        // Create arcs path
        let mut arcs = BezPath::new();
        arcs.move_to((0.0, 480.0));
        arcs.curve_to((500.0, 480.0), (500.0, -10.0), (0.0, -10.0));
        arcs.close_path();
        arcs.move_to((500.0, -10.0));
        arcs.curve_to((0.0, -10.0), (0.0, 480.0), (500.0, 480.0));
        arcs.close_path();

        let scale = Affine::scale(0.6);
        let t = Affine::translate((10.0, 25.0));
        let rules = [
            (Fill::NonZero, "Non-Zero", &star),
            (Fill::EvenOdd, "Even-Odd", &star),
            (Fill::NonZero, "Non-Zero", &arcs),
            (Fill::EvenOdd, "Even-Odd", &arcs),
        ];

        // Draw basic fills
        for (i, rule) in rules.iter().enumerate() {
            let t = Affine::translate(((i % 2) as f64 * 306.0, (i / 2) as f64 * 340.0)) * t;

            let t = Affine::translate((0.0, 5.0)) * t * scale;

            // Gray background
            ctx.set_transform(root_transform * t);
            ctx.set_paint(GRAY);
            ctx.fill_rect(&rect);

            // Fill with rule
            ctx.set_transform(root_transform * Affine::translate((0.0, 10.0)) * t);
            ctx.set_paint(YELLOW);
            ctx.set_fill_rule(rule.0);
            ctx.fill_path(rule.2);
        }

        // Draw blends
        let t = Affine::translate((700.0, 0.0)) * t;
        for (i, rule) in rules.iter().enumerate() {
            let t = Affine::translate(((i % 2) as f64 * 306.0, (i / 2) as f64 * 340.0)) * t;

            let t = Affine::translate((0.0, 5.0)) * t * scale;

            // Gray background
            ctx.set_transform(root_transform * t);
            ctx.set_paint(GRAY);
            ctx.fill_rect(&rect);

            // First fill
            ctx.set_transform(root_transform * Affine::translate((0.0, 10.0)) * t);
            ctx.set_paint(YELLOW);
            ctx.set_fill_rule(rule.0);
            ctx.fill_path(rule.2);

            // Second fill with rotation
            ctx.set_transform(
                root_transform * Affine::translate((0.0, 10.0)) * t * Affine::rotate(0.06),
            );
            ctx.set_paint(Color::new([0.0, 1.0, 0.7, 0.6]));
            ctx.set_fill_rule(rule.0);
            ctx.fill_path(rule.2);

            // Third fill with opposite rotation
            ctx.set_transform(
                root_transform * Affine::translate((0.0, 10.0)) * t * Affine::rotate(-0.06),
            );
            ctx.set_paint(Color::new([0.9, 0.7, 0.5, 0.6]));
            ctx.set_fill_rule(rule.0);
            ctx.fill_path(rule.2);
        }
    }
}

/// Robust paths scene state
#[derive(Debug, Default)]
pub struct RobustPathsScene {}

impl RobustPathsScene {
    /// Create a new robust paths scene
    pub fn new() -> Self {
        Self {}
    }
}

impl ExampleScene for RobustPathsScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        let mut path = BezPath::new();
        path.move_to((16.0, 16.0));
        path.line_to((32.0, 16.0));
        path.line_to((32.0, 32.0));
        path.line_to((16.0, 32.0));
        path.close_path();
        path.move_to((48.0, 18.0));
        path.line_to((64.0, 23.0));
        path.line_to((64.0, 33.0));
        path.line_to((48.0, 38.0));
        path.close_path();
        path.move_to((80.0, 18.0));
        path.line_to((82.0, 16.0));
        path.line_to((94.0, 16.0));
        path.line_to((96.0, 18.0));
        path.line_to((96.0, 30.0));
        path.line_to((94.0, 32.0));
        path.line_to((82.0, 32.0));
        path.line_to((80.0, 30.0));
        path.close_path();
        path.move_to((112.0, 16.0));
        path.line_to((128.0, 16.0));
        path.line_to((128.0, 32.0));
        path.close_path();
        path.move_to((144.0, 16.0));
        path.line_to((160.0, 32.0));
        path.line_to((144.0, 32.0));
        path.close_path();
        path.move_to((168.0, 8.0));
        path.line_to((184.0, 8.0));
        path.line_to((184.0, 24.0));
        path.close_path();
        path.move_to((200.0, 8.0));
        path.line_to((216.0, 24.0));
        path.line_to((200.0, 24.0));
        path.close_path();
        path.move_to((241.0, 17.5));
        path.line_to((255.0, 17.5));
        path.line_to((255.0, 19.5));
        path.line_to((241.0, 19.5));
        path.close_path();
        path.move_to((241.0, 22.5));
        path.line_to((256.0, 22.5));
        path.line_to((256.0, 24.5));
        path.line_to((241.0, 24.5));
        path.close_path();

        // Fill with NonZero rule
        ctx.set_transform(root_transform);
        ctx.set_paint(YELLOW);
        ctx.fill_path(&path);

        // Fill with EvenOdd rule
        ctx.set_transform(root_transform * Affine::translate((300.0, 0.0)));
        ctx.set_paint(LIME);
        ctx.set_fill_rule(Fill::EvenOdd);
        ctx.fill_path(&path);

        // Add another path section
        path.move_to((8.0, 4.0));
        path.line_to((8.0, 40.0));
        path.line_to((260.0, 40.0));
        path.line_to((260.0, 4.0));
        path.close_path();

        // Fill with NonZero rule
        ctx.set_transform(root_transform * Affine::translate((0.0, 100.0)));
        ctx.set_paint(YELLOW);
        ctx.fill_path(&path);

        // Fill with EvenOdd rule
        ctx.set_transform(root_transform * Affine::translate((300.0, 100.0)));
        ctx.set_paint(LIME);
        ctx.set_fill_rule(Fill::EvenOdd);
        ctx.fill_path(&path);
    }
}
