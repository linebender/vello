// Copyright 2022 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{ExampleScene, SceneConfig, SceneParams, SceneSet};
use vello::kurbo::{Affine, BezPath, Cap, Ellipse, Join, PathEl, Point, Rect, Stroke};
use vello::peniko::*;
use vello::*;

const FLOWER_IMAGE: &[u8] = include_bytes!("../../assets/splash-flower.jpg");

macro_rules! scene {
    ($name: ident) => {
        scene!($name: false)
    };
    ($name: ident: animated) => {
        scene!($name: true)
    };
    ($name: ident: $animated: literal) => {
        scene!($name, stringify!($name), $animated)
    };
    ($func:expr, $name: expr, $animated: literal) => {
        ExampleScene {
            config: SceneConfig {
                animated: $animated,
                name: $name.to_owned(),
            },
            function: Box::new($func),
        }
    };
}

pub fn test_scenes() -> SceneSet {
    let scenes = vec![
        scene!(splash_with_tiger(), "splash_with_tiger", false),
        scene!(funky_paths),
        scene!(stroke_styles),
        scene!(tricky_strokes),
        scene!(fill_types),
        scene!(cardioid_and_friends),
        scene!(animated_text: animated),
        scene!(gradient_extend),
        scene!(two_point_radial),
        scene!(brush_transform: animated),
        scene!(blend_grid),
        scene!(conflation_artifacts),
        scene!(labyrinth),
        scene!(robust_paths),
        scene!(base_color_test: animated),
        scene!(clip_test: animated),
        scene!(longpathdash(Cap::Butt), "longpathdash (butt caps)", false),
        scene!(longpathdash(Cap::Round), "longpathdash (round caps)", false),
        scene!(crate::mmark::MMark::new(80_000), "mmark", false),
    ];

    SceneSet { scenes }
}

// Scenes

fn funky_paths(sb: &mut SceneBuilder, _: &mut SceneParams) {
    use PathEl::*;
    let missing_movetos = [
        LineTo((100.0, 100.0).into()),
        LineTo((100.0, 200.0).into()),
        ClosePath,
        LineTo((0.0, 400.0).into()),
        LineTo((100.0, 400.0).into()),
    ];
    let only_movetos = [MoveTo((0.0, 0.0).into()), MoveTo((100.0, 100.0).into())];
    let empty: [PathEl; 0] = [];
    sb.fill(
        Fill::NonZero,
        Affine::translate((100.0, 100.0)),
        Color::rgb8(0, 0, 255),
        None,
        &missing_movetos,
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgb8(0, 0, 255),
        None,
        &empty,
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgb8(0, 0, 255),
        None,
        &only_movetos,
    );
    sb.stroke(
        &Stroke::new(8.0),
        Affine::translate((100.0, 100.0)),
        Color::rgb8(0, 255, 255),
        None,
        &missing_movetos,
    );
}

fn stroke_styles(sb: &mut SceneBuilder, params: &mut SceneParams) {
    use PathEl::*;
    let colors = [
        Color::rgb8(140, 181, 236),
        Color::rgb8(246, 236, 202),
        Color::rgb8(201, 147, 206),
        Color::rgb8(150, 195, 160),
    ];
    let simple_stroke = [MoveTo((0., 0.).into()), LineTo((100., 0.).into())];
    let join_stroke = [
        MoveTo((0., 0.).into()),
        CurveTo((20., 0.).into(), (42.5, 5.).into(), (50., 25.).into()),
        CurveTo((57.5, 5.).into(), (80., 0.).into(), (100., 0.).into()),
    ];
    let miter_stroke = [
        MoveTo((0., 0.).into()),
        LineTo((90., 21.).into()),
        LineTo((0., 42.).into()),
    ];
    let closed_strokes = [
        MoveTo((0., 0.).into()),
        LineTo((90., 21.).into()),
        LineTo((0., 42.).into()),
        ClosePath,
        MoveTo((200., 0.).into()),
        CurveTo((100., 42.).into(), (300., 42.).into(), (200., 0.).into()),
        ClosePath,
        MoveTo((290., 0.).into()),
        CurveTo((200., 42.).into(), (400., 42.).into(), (310., 0.).into()),
        ClosePath,
    ];
    let cap_styles = [Cap::Butt, Cap::Square, Cap::Round];
    let join_styles = [Join::Bevel, Join::Miter, Join::Round];
    let miter_limits = [4., 5., 0.1, 10.];

    // Simple strokes with cap combinations
    let t = Affine::translate((60., 40.)) * Affine::scale(2.);
    let mut y = 0.;
    let mut color_idx = 0;
    for start in cap_styles {
        for end in cap_styles {
            params.text.add(
                sb,
                None,
                12.,
                None,
                Affine::translate((0., y)) * t,
                &format!("Start cap: {:?}, End cap: {:?}", start, end),
            );
            sb.stroke(
                &Stroke::new(20.).with_start_cap(start).with_end_cap(end),
                Affine::translate((0., y + 30.)) * t,
                colors[color_idx],
                None,
                &simple_stroke,
            );
            y += 180.;
            color_idx = (color_idx + 1) % colors.len();
        }
    }

    // Cap and join combinations
    let t = Affine::translate((500., 0.)) * t;
    y = 0.;
    for cap in cap_styles {
        for join in join_styles {
            params.text.add(
                sb,
                None,
                12.,
                None,
                Affine::translate((0., y)) * t,
                &format!("Caps: {:?}, Joins: {:?}", cap, join),
            );
            sb.stroke(
                &Stroke::new(20.).with_caps(cap).with_join(join),
                Affine::translate((0., y + 30.)) * t,
                colors[color_idx],
                None,
                &join_stroke,
            );
            y += 185.;
            color_idx = (color_idx + 1) % colors.len();
        }
    }

    // Miter limit
    let t = Affine::translate((500., 0.)) * t;
    y = 0.;
    for ml in miter_limits {
        params.text.add(
            sb,
            None,
            12.,
            None,
            Affine::translate((0., y)) * t,
            &format!("Miter limit: {}", ml),
        );
        sb.stroke(
            &Stroke::new(10.)
                .with_caps(Cap::Butt)
                .with_join(Join::Miter)
                .with_miter_limit(ml),
            Affine::translate((0., y + 30.)) * t,
            colors[color_idx],
            None,
            &miter_stroke,
        );
        y += 180.;
        color_idx = (color_idx + 1) % colors.len();
    }

    // Closed paths
    for (i, join) in join_styles.iter().enumerate() {
        params.text.add(
            sb,
            None,
            12.,
            None,
            Affine::translate((0., y)) * t,
            &format!("Closed path with join: {:?}", join),
        );
        // The cap style is not important since a closed path shouldn't have any caps.
        sb.stroke(
            &Stroke::new(10.)
                .with_caps(cap_styles[i])
                .with_join(*join)
                .with_miter_limit(5.),
            Affine::translate((0., y + 30.)) * t,
            colors[color_idx],
            None,
            &closed_strokes,
        );
        y += 180.;
        color_idx = (color_idx + 1) % colors.len();
    }
}

// This test has been adapted from Skia's "trickycubicstrokes" GM slide which can be found at
// `github.com/google/skia/blob/0d4d11451c4f4e184305cbdbd67f6b3edfa4b0e3/gm/trickycubicstrokes.cpp`
fn tricky_strokes(sb: &mut SceneBuilder, _: &mut SceneParams) {
    use PathEl::*;
    let colors = [
        Color::rgb8(140, 181, 236),
        Color::rgb8(246, 236, 202),
        Color::rgb8(201, 147, 206),
        Color::rgb8(150, 195, 160),
    ];

    const CELL_SIZE: f64 = 200.;
    const STROKE_WIDTH: f64 = 30.;
    const NUM_COLS: usize = 5;

    fn stroke_bounds(pts: &[(f64, f64); 4]) -> Rect {
        use kurbo::{CubicBez, Shape};
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
        // The following cases used to be `_broken_cubics` but now work.
        [(0., -0.01), (128., 128.001), (128., -0.01), (0., 128.001)], // Near-cusp
        [(0., 0.), (0., -10.), (0., -10.), (0., 10.)],                // Flat line with 180
        [(10., 0.), (0., 0.), (20., 0.), (10., 0.)],                  // Flat line with 2 180s
        [(39., -39.), (40., -40.), (40., -40.), (0., 0.)],            // Flat diagonal with 180
        [(40., 40.), (0., 0.), (200., 200.), (0., 0.)],               // Diag w/ an internal 180
        [(0., 0.), (1e-2, 0.), (-1e-2, 0.), (0., 0.)],                // Circle
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

    let mut color_idx = 0;
    for (i, cubic) in tricky_cubics.into_iter().enumerate() {
        let x = (i % NUM_COLS) as f64 * CELL_SIZE;
        let y = (i / NUM_COLS) as f64 * CELL_SIZE;
        let cell = Rect::new(x, y, x + CELL_SIZE, y + CELL_SIZE);
        let bounds = stroke_bounds(&cubic);
        let (t, s) = map_rect_to_rect(&bounds, &cell);
        sb.stroke(
            &Stroke::new(STROKE_WIDTH / s)
                .with_caps(Cap::Butt)
                .with_join(Join::Miter),
            t,
            colors[color_idx],
            None,
            &[
                MoveTo(cubic[0].into()),
                CurveTo(cubic[1].into(), cubic[2].into(), cubic[3].into()),
            ],
        );
        color_idx = (color_idx + 1) % colors.len();
    }
}

fn fill_types(sb: &mut SceneBuilder, params: &mut SceneParams) {
    use PathEl::*;
    let rect = Rect::from_origin_size(Point::new(0., 0.), (500., 500.));
    let star = [
        MoveTo((250., 0.).into()),
        LineTo((105., 450.).into()),
        LineTo((490., 175.).into()),
        LineTo((10., 175.).into()),
        LineTo((395., 450.).into()),
        ClosePath,
    ];
    let arcs = [
        MoveTo((0., 480.).into()),
        CurveTo((500., 480.).into(), (500., -10.).into(), (0., -10.).into()),
        ClosePath,
        MoveTo((500., -10.).into()),
        CurveTo((0., -10.).into(), (0., 480.).into(), (500., 480.).into()),
        ClosePath,
    ];
    let scale = Affine::scale(0.6);
    let t = Affine::translate((10., 25.));
    let rules = [
        (Fill::NonZero, "Non-Zero", star.as_slice()),
        (Fill::EvenOdd, "Even-Odd", &star),
        (Fill::NonZero, "Non-Zero", &arcs),
        (Fill::EvenOdd, "Even-Odd", &arcs),
    ];
    for (i, rule) in rules.iter().enumerate() {
        let t = Affine::translate(((i % 2) as f64 * 306., (i / 2) as f64 * 340.)) * t;
        params.text.add(sb, None, 24., None, t, rule.1);
        let t = Affine::translate((0., 5.)) * t * scale;
        sb.fill(
            Fill::NonZero,
            t,
            &Brush::Solid(Color::rgb8(128, 128, 128)),
            None,
            &rect,
        );
        sb.fill(
            rule.0,
            Affine::translate((0., 10.)) * t,
            Color::YELLOW,
            None,
            &rule.2,
        );
    }

    // Draw blends
    let t = Affine::translate((700., 0.)) * t;
    for (i, rule) in rules.iter().enumerate() {
        let t = Affine::translate(((i % 2) as f64 * 306., (i / 2) as f64 * 340.)) * t;
        params.text.add(sb, None, 24., None, t, rule.1);
        let t = Affine::translate((0., 5.)) * t * scale;
        sb.fill(
            Fill::NonZero,
            t,
            &Brush::Solid(Color::rgb8(128, 128, 128)),
            None,
            &rect,
        );
        sb.fill(
            rule.0,
            Affine::translate((0., 10.)) * t,
            Color::YELLOW,
            None,
            &rule.2,
        );
        sb.fill(
            rule.0,
            Affine::translate((0., 10.)) * t * Affine::rotate(0.06),
            Color::rgba(0., 1., 0.7, 0.6),
            None,
            &rule.2,
        );
        sb.fill(
            rule.0,
            Affine::translate((0., 10.)) * t * Affine::rotate(-0.06),
            Color::rgba(0.9, 0.7, 0.5, 0.6),
            None,
            &rule.2,
        );
    }
}

fn cardioid_and_friends(sb: &mut SceneBuilder, _: &mut SceneParams) {
    render_cardioid(sb);
    render_clip_test(sb);
    render_alpha_test(sb);
    //render_tiger(sb, false);
}

fn longpathdash(cap: Cap) -> impl FnMut(&mut SceneBuilder, &mut SceneParams) {
    use std::f64::consts::PI;
    use PathEl::*;
    move |sb, _| {
        let mut path = BezPath::new();
        let mut x = 32;
        while x < 256 {
            let mut a: f64 = 0.0;
            while a < PI * 2.0 {
                let pts = [
                    (256.0 + a.sin() * x as f64, 256.0 + a.cos() * x as f64),
                    (
                        256.0 + (a + PI / 3.0).sin() * (x + 64) as f64,
                        256.0 + (a + PI / 3.0).cos() * (x + 64) as f64,
                    ),
                ];
                path.push(MoveTo(pts[0].into()));
                let mut i: f64 = 0.0;
                while i < 1.0 {
                    path.push(LineTo(
                        (
                            pts[0].0 * (1.0 - i) + pts[1].0 * i,
                            pts[0].1 * (1.0 - i) + pts[1].1 * i,
                        )
                            .into(),
                    ));
                    i += 0.05;
                }
                a += PI * 0.01;
            }
            x += 16;
        }
        sb.stroke(
            &Stroke::new(1.0).with_caps(cap).with_dashes(0.0, [1.0, 1.0]),
            Affine::translate((50.0, 50.0)),
            Color::YELLOW,
            None,
            &path,
        );
    }
}

fn animated_text(sb: &mut SceneBuilder, params: &mut SceneParams) {
    // Uses the static array address as a cache key for expedience. Real code
    // should use a better strategy.
    let piet_logo = params
        .images
        .from_bytes(FLOWER_IMAGE.as_ptr() as usize, FLOWER_IMAGE)
        .unwrap();

    use PathEl::*;
    let rect = Rect::from_origin_size(Point::new(0.0, 0.0), (1000.0, 1000.0));
    let star = [
        MoveTo((50.0, 0.0).into()),
        LineTo((21.0, 90.0).into()),
        LineTo((98.0, 35.0).into()),
        LineTo((2.0, 35.0).into()),
        LineTo((79.0, 90.0).into()),
        ClosePath,
    ];
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(128, 128, 128)),
        None,
        &rect,
    );
    let text_size = 60.0 + 40.0 * (params.time as f32).sin();
    let s = "\u{1f600}hello vello text!";
    params.text.add(
        sb,
        None,
        text_size,
        None,
        Affine::translate((110.0, 600.0)),
        s,
    );
    params.text.add_run(
        sb,
        None,
        text_size,
        Color::WHITE,
        Affine::translate((110.0, 700.0)),
        // Add a skew to simulate an oblique font.
        Some(Affine::skew(20f64.to_radians().tan(), 0.0)),
        &Stroke::new(1.0),
        s,
    );
    let t = ((params.time).sin() * 0.5 + 0.5) as f32;
    let weight = t * 700.0 + 200.0;
    let width = t * 150.0 + 50.0;
    params.text.add_var_run(
        sb,
        None,
        72.0,
        &[("wght", weight), ("wdth", width)],
        Color::WHITE,
        Affine::translate((110.0, 800.0)),
        // Add a skew to simulate an oblique font.
        None,
        Fill::NonZero,
        "And some vello\ntext with a newline",
    );
    let th = params.time;
    let center = Point::new(500.0, 500.0);
    let mut p1 = center;
    p1.x += 400.0 * th.cos();
    p1.y += 400.0 * th.sin();
    sb.stroke(
        &Stroke::new(5.0),
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(128, 0, 0)),
        None,
        &[PathEl::MoveTo(center), PathEl::LineTo(p1)],
    );
    sb.fill(
        Fill::NonZero,
        Affine::translate((150.0, 150.0)) * Affine::scale(0.2),
        Color::RED,
        None,
        &rect,
    );
    let alpha = params.time.sin() as f32 * 0.5 + 0.5;
    sb.push_layer(Mix::Normal, alpha, Affine::IDENTITY, &rect);
    sb.fill(
        Fill::NonZero,
        Affine::translate((100.0, 100.0)) * Affine::scale(0.2),
        Color::BLUE,
        None,
        &rect,
    );
    sb.fill(
        Fill::NonZero,
        Affine::translate((200.0, 200.0)) * Affine::scale(0.2),
        Color::GREEN,
        None,
        &rect,
    );
    sb.pop_layer();
    sb.fill(
        Fill::NonZero,
        Affine::translate((400.0, 100.0)),
        Color::PURPLE,
        None,
        &star,
    );
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((500.0, 100.0)),
        Color::PURPLE,
        None,
        &star,
    );
    sb.draw_image(
        &piet_logo,
        Affine::translate((800.0, 50.0)) * Affine::rotate(20f64.to_radians()),
    );
}

fn brush_transform(sb: &mut SceneBuilder, params: &mut SceneParams) {
    let th = params.time;
    let linear = Gradient::new_linear((0.0, 0.0), (0.0, 200.0)).with_stops([
        Color::RED,
        Color::GREEN,
        Color::BLUE,
    ]);
    sb.fill(
        Fill::NonZero,
        Affine::rotate(25f64.to_radians()) * Affine::scale_non_uniform(2.0, 1.0),
        &Gradient::new_radial((200.0, 200.0), 80.0).with_stops([
            Color::RED,
            Color::GREEN,
            Color::BLUE,
        ]),
        None,
        &Rect::from_origin_size((100.0, 100.0), (200.0, 200.0)),
    );
    sb.fill(
        Fill::NonZero,
        Affine::translate((200.0, 600.0)),
        &linear,
        Some(around_center(Affine::rotate(th), Point::new(200.0, 100.0))),
        &Rect::from_origin_size(Point::default(), (400.0, 200.0)),
    );
    sb.stroke(
        &Stroke::new(40.0),
        Affine::translate((800.0, 600.0)),
        &linear,
        Some(around_center(Affine::rotate(th), Point::new(200.0, 100.0))),
        &Rect::from_origin_size(Point::default(), (400.0, 200.0)),
    );
}

fn gradient_extend(sb: &mut SceneBuilder, params: &mut SceneParams) {
    fn square(sb: &mut SceneBuilder, is_radial: bool, transform: Affine, extend: Extend) {
        let colors = [Color::RED, Color::rgb8(0, 255, 0), Color::BLUE];
        let width = 300f64;
        let height = 300f64;
        let gradient: Brush = if is_radial {
            let center = (width * 0.5, height * 0.5);
            let radius = (width * 0.25) as f32;
            Gradient::new_two_point_radial(center, radius * 0.25, center, radius)
                .with_stops(colors)
                .with_extend(extend)
                .into()
        } else {
            Gradient::new_linear((width * 0.35, height * 0.5), (width * 0.65, height * 0.5))
                .with_stops(colors)
                .with_extend(extend)
                .into()
        };
        sb.fill(
            Fill::NonZero,
            transform,
            &gradient,
            None,
            &Rect::new(0.0, 0.0, width, height),
        );
    }
    let extend_modes = [Extend::Pad, Extend::Repeat, Extend::Reflect];
    for (x, extend) in extend_modes.iter().enumerate() {
        for y in 0..2 {
            let is_radial = y & 1 != 0;
            let transform = Affine::translate((x as f64 * 350.0 + 50.0, y as f64 * 350.0 + 100.0));
            square(sb, is_radial, transform, *extend);
        }
    }
    for (i, label) in ["Pad", "Repeat", "Reflect"].iter().enumerate() {
        let x = i as f64 * 350.0 + 50.0;
        params.text.add(
            sb,
            None,
            32.0,
            Some(&Color::WHITE.into()),
            Affine::translate((x, 70.0)),
            label,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn two_point_radial(sb: &mut SceneBuilder, _params: &mut SceneParams) {
    fn make(
        sb: &mut SceneBuilder,
        x0: f64,
        y0: f64,
        r0: f32,
        x1: f64,
        y1: f64,
        r1: f32,
        transform: Affine,
        extend: Extend,
    ) {
        let colors = [Color::RED, Color::YELLOW, Color::rgb8(6, 85, 186)];
        let width = 400f64;
        let height = 200f64;
        let rect = Rect::new(0.0, 0.0, width, height);
        sb.fill(Fill::NonZero, transform, Color::WHITE, None, &rect);
        sb.fill(
            Fill::NonZero,
            transform,
            &Gradient::new_two_point_radial((x0, y0), r0, (x1, y1), r1)
                .with_stops(colors)
                .with_extend(extend),
            None,
            &Rect::new(0.0, 0.0, width, height),
        );
        let r0 = r0 as f64 - 1.0;
        let r1 = r1 as f64 - 1.0;
        let stroke_width = 1.0;
        sb.stroke(
            &Stroke::new(stroke_width),
            transform,
            Color::BLACK,
            None,
            &Ellipse::new((x0, y0), (r0, r0), 0.0),
        );
        sb.stroke(
            &Stroke::new(stroke_width),
            transform,
            Color::BLACK,
            None,
            &Ellipse::new((x1, y1), (r1, r1), 0.0),
        );
    }

    // These demonstrate radial gradient patterns similar to the examples shown
    // at <https://learn.microsoft.com/en-us/typography/opentype/spec/colr#radial-gradients>

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
            sb,
            x0,
            y,
            r0,
            x1,
            y,
            r1,
            Affine::translate((i as f64 * 420.0 + 20.0, 20.0)),
            *mode,
        );
    }

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
            sb,
            x1,
            y,
            r1,
            x0,
            y,
            r0,
            Affine::translate((i as f64 * 420.0 + 20.0, 240.0)),
            *mode,
        );
    }

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
            sb,
            x0,
            y,
            r0,
            x1,
            y,
            r1,
            Affine::translate((i as f64 * 420.0 + 20.0, 460.0)),
            *mode,
        );
    }

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
            sb,
            x0,
            y0,
            r0,
            x1,
            y1,
            r1,
            Affine::translate((i as f64 * 420.0 + 20.0, 680.0)),
            *mode,
        );
    }

    for (i, mode) in [Extend::Pad, Extend::Repeat, Extend::Reflect]
        .iter()
        .enumerate()
    {
        let x0 = 140.0;
        let y0 = 125.0;
        let r0 = 20.0;
        let x1 = 190.0;
        let y1 = 100.0;
        let r1 = 96.0;
        // Shift p0 so the outer edges of both circles touch
        let p0 = Point::new(x1, y1)
            + ((Point::new(x0, y0) - Point::new(x1, y1)).normalize() * (r1 - r0));
        make(
            sb,
            p0.x,
            p0.y,
            r0 as f32,
            x1,
            y1,
            r1 as f32,
            Affine::translate((i as f64 * 420.0 + 20.0, 900.0)),
            *mode,
        );
    }
}

fn blend_grid(sb: &mut SceneBuilder, _: &mut SceneParams) {
    const BLEND_MODES: &[Mix] = &[
        Mix::Normal,
        Mix::Multiply,
        Mix::Darken,
        Mix::Screen,
        Mix::Lighten,
        Mix::Overlay,
        Mix::ColorDodge,
        Mix::ColorBurn,
        Mix::HardLight,
        Mix::SoftLight,
        Mix::Difference,
        Mix::Exclusion,
        Mix::Hue,
        Mix::Saturation,
        Mix::Color,
        Mix::Luminosity,
    ];
    for (ix, &blend) in BLEND_MODES.iter().enumerate() {
        let i = ix % 4;
        let j = ix / 4;
        let transform = Affine::translate((i as f64 * 225., j as f64 * 225.));
        let square = blend_square(blend.into());
        sb.append(&square, Some(transform));
    }
}

// Support functions

fn render_cardioid(sb: &mut SceneBuilder) {
    let n = 601;
    let dth = std::f64::consts::PI * 2.0 / (n as f64);
    let center = Point::new(1024.0, 768.0);
    let r = 750.0;
    let mut path = BezPath::new();
    for i in 1..n {
        let mut p0 = center;
        let a0 = i as f64 * dth;
        p0.x += a0.cos() * r;
        p0.y += a0.sin() * r;
        let mut p1 = center;
        let a1 = ((i * 2) % n) as f64 * dth;
        p1.x += a1.cos() * r;
        p1.y += a1.sin() * r;
        path.push(PathEl::MoveTo(p0));
        path.push(PathEl::LineTo(p1));
    }
    sb.stroke(
        &Stroke::new(2.0),
        Affine::IDENTITY,
        Color::rgb8(0, 0, 255),
        None,
        &path,
    );
}

fn render_clip_test(sb: &mut SceneBuilder) {
    const N: usize = 16;
    const X0: f64 = 50.0;
    const Y0: f64 = 450.0;
    // Note: if it gets much larger, it will exceed the 1MB scratch buffer.
    // But this is a pretty demanding test.
    const X1: f64 = 550.0;
    const Y1: f64 = 950.0;
    let step = 1.0 / ((N + 1) as f64);
    for i in 0..N {
        let t = ((i + 1) as f64) * step;
        let path = [
            PathEl::MoveTo((X0, Y0).into()),
            PathEl::LineTo((X1, Y0).into()),
            PathEl::LineTo((X1, Y0 + t * (Y1 - Y0)).into()),
            PathEl::LineTo((X1 + t * (X0 - X1), Y1).into()),
            PathEl::LineTo((X0, Y1).into()),
            PathEl::ClosePath,
        ];
        sb.push_layer(Mix::Clip, 1.0, Affine::IDENTITY, &path);
    }
    let rect = Rect::new(X0, Y0, X1, Y1);
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(Color::rgb8(0, 255, 0)),
        None,
        &rect,
    );
    for _ in 0..N {
        sb.pop_layer();
    }
}

fn render_alpha_test(sb: &mut SceneBuilder) {
    // Alpha compositing tests.
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgb8(255, 0, 0),
        None,
        &make_diamond(1024.0, 100.0),
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgba8(0, 255, 0, 0x80),
        None,
        &make_diamond(1024.0, 125.0),
    );
    sb.push_layer(
        Mix::Clip,
        1.0,
        Affine::IDENTITY,
        &make_diamond(1024.0, 150.0),
    );
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgba8(0, 0, 255, 0x80),
        None,
        &make_diamond(1024.0, 175.0),
    );
    sb.pop_layer();
}

fn render_blend_square(sb: &mut SceneBuilder, blend: BlendMode, transform: Affine) {
    // Inspired by https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
    let rect = Rect::from_origin_size(Point::new(0., 0.), (200., 200.));
    let linear =
        Gradient::new_linear((0.0, 0.0), (200.0, 0.0)).with_stops([Color::BLACK, Color::WHITE]);
    sb.fill(Fill::NonZero, transform, &linear, None, &rect);
    const GRADIENTS: &[(f64, f64, Color)] = &[
        (150., 0., Color::rgb8(255, 240, 64)),
        (175., 100., Color::rgb8(255, 96, 240)),
        (125., 200., Color::rgb8(64, 192, 255)),
    ];
    for (x, y, c) in GRADIENTS {
        let mut color2 = *c;
        color2.a = 0;
        let radial = Gradient::new_radial((*x, *y), 100.0).with_stops([*c, color2]);
        sb.fill(Fill::NonZero, transform, &radial, None, &rect);
    }
    const COLORS: &[Color] = &[
        Color::rgb8(255, 0, 0),
        Color::rgb8(0, 255, 0),
        Color::rgb8(0, 0, 255),
    ];
    sb.push_layer(Mix::Normal, 1.0, transform, &rect);
    for (i, c) in COLORS.iter().enumerate() {
        let linear = Gradient::new_linear((0.0, 0.0), (0.0, 200.0)).with_stops([Color::WHITE, *c]);
        sb.push_layer(blend, 1.0, transform, &rect);
        // squash the ellipse
        let a = transform
            * Affine::translate((100., 100.))
            * Affine::rotate(std::f64::consts::FRAC_PI_3 * (i * 2 + 1) as f64)
            * Affine::scale_non_uniform(1.0, 0.357)
            * Affine::translate((-100., -100.));
        sb.fill(
            Fill::NonZero,
            a,
            &linear,
            None,
            &Ellipse::new((100., 100.), (90., 90.), 0.),
        );
        sb.pop_layer();
    }
    sb.pop_layer();
}

fn blend_square(blend: BlendMode) -> SceneFragment {
    let mut fragment = SceneFragment::default();
    let mut sb = SceneBuilder::for_fragment(&mut fragment);
    render_blend_square(&mut sb, blend, Affine::IDENTITY);
    fragment
}

fn conflation_artifacts(sb: &mut SceneBuilder, _: &mut SceneParams) {
    use PathEl::*;
    const N: f64 = 50.0;
    const S: f64 = 4.0;

    let scale = Affine::scale(S);
    let x = N + 0.5; // Fractional pixel offset reveals the problem on axis-aligned edges.
    let mut y = N;

    let bg_color = Color::rgb8(255, 194, 19);
    let fg_color = Color::rgb8(12, 165, 255);

    // Two adjacent triangles touching at diagonal edge with opposing winding numbers
    sb.fill(
        Fill::NonZero,
        Affine::translate((x, y)) * scale,
        fg_color,
        None,
        &[
            // triangle 1
            MoveTo((0.0, 0.0).into()),
            LineTo((N, N).into()),
            LineTo((0.0, N).into()),
            LineTo((0.0, 0.0).into()),
            // triangle 2
            MoveTo((0.0, 0.0).into()),
            LineTo((N, N).into()),
            LineTo((N, 0.0).into()),
            LineTo((0.0, 0.0).into()),
        ],
    );

    // Adjacent rects, opposite winding
    y += S * N + 10.0;
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((x, y)) * scale,
        bg_color,
        None,
        &Rect::new(0.0, 0.0, N, N),
    );
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((x, y)) * scale,
        fg_color,
        None,
        &[
            // left rect
            MoveTo((0.0, 0.0).into()),
            LineTo((0.0, N).into()),
            LineTo((N * 0.5, N).into()),
            LineTo((N * 0.5, 0.0).into()),
            // right rect
            MoveTo((N * 0.5, 0.0).into()),
            LineTo((N, 0.0).into()),
            LineTo((N, N).into()),
            LineTo((N * 0.5, N).into()),
        ],
    );

    // Adjacent rects, same winding
    y += S * N + 10.0;
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((x, y)) * scale,
        bg_color,
        None,
        &Rect::new(0.0, 0.0, N, N),
    );
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((x, y)) * scale,
        fg_color,
        None,
        &[
            // left rect
            MoveTo((0.0, 0.0).into()),
            LineTo((0.0, N).into()),
            LineTo((N * 0.5, N).into()),
            LineTo((N * 0.5, 0.0).into()),
            // right rect
            MoveTo((N * 0.5, 0.0).into()),
            LineTo((N * 0.5, N).into()),
            LineTo((N, N).into()),
            LineTo((N, 0.0).into()),
        ],
    );
}

fn labyrinth(sb: &mut SceneBuilder, _: &mut SceneParams) {
    use PathEl::*;

    let rows: &[[u8; 12]] = &[
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ];
    let cols: &[[u8; 10]] = &[
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    ];
    let mut path = BezPath::new();
    for (y, row) in rows.iter().enumerate() {
        for (x, flag) in row.iter().enumerate() {
            let x = x as f64;
            let y = y as f64;
            if *flag == 1 {
                path.push(MoveTo((x - 0.1, y + 0.1).into()));
                path.push(LineTo((x + 1.1, y + 0.1).into()));
                path.push(LineTo((x + 1.1, y - 0.1).into()));
                path.push(LineTo((x - 0.1, y - 0.1).into()));

                // The above is equivalent to the following stroke with width 0.2 and square
                // caps.
                //path.push(MoveTo((x, y).into()));
                //path.push(LineTo((x + 1.0, y).into()));
            }
        }
    }
    for (x, col) in cols.iter().enumerate() {
        for (y, flag) in col.iter().enumerate() {
            let x = x as f64;
            let y = y as f64;
            if *flag == 1 {
                path.push(MoveTo((x - 0.1, y - 0.1).into()));
                path.push(LineTo((x - 0.1, y + 1.1).into()));
                path.push(LineTo((x + 0.1, y + 1.1).into()));
                path.push(LineTo((x + 0.1, y - 0.1).into()));
                // The above is equivalent to the following stroke with width 0.2 and square
                // caps.
                //path.push(MoveTo((x, y).into()));
                //path.push(LineTo((x, y + 1.0).into()));
            }
        }
    }

    // Note the artifacts are clearly visible at a fractional pixel offset/translation. They
    // disappear if the translation amount below is a whole number.
    sb.fill(
        Fill::NonZero,
        Affine::translate((20.5, 20.5)) * Affine::scale(80.0),
        Color::rgba8(0x70, 0x80, 0x80, 0xff),
        None,
        &path,
    );
}

fn robust_paths(sb: &mut SceneBuilder, _: &mut SceneParams) {
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
    sb.fill(Fill::NonZero, Affine::IDENTITY, Color::YELLOW, None, &path);
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((300.0, 0.0)),
        Color::LIME,
        None,
        &path,
    );

    path.move_to((8.0, 4.0));
    path.line_to((8.0, 40.0));
    path.line_to((260.0, 40.0));
    path.line_to((260.0, 4.0));
    path.close_path();
    sb.fill(
        Fill::NonZero,
        Affine::translate((0.0, 100.0)),
        Color::YELLOW,
        None,
        &path,
    );
    sb.fill(
        Fill::EvenOdd,
        Affine::translate((300.0, 100.0)),
        Color::LIME,
        None,
        &path,
    );
}

fn base_color_test(sb: &mut SceneBuilder, params: &mut SceneParams) {
    // Cycle through the hue value every 5 seconds (t % 5) * 360/5
    let color = Color::hlc((params.time % 5.0) * 72.0, 80.0, 80.0);
    params.base_color = Some(color);

    // Blend a white square over it.
    sb.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::rgba8(255, 255, 255, 128),
        None,
        &Rect::new(50.0, 50.0, 500.0, 500.0),
    );
}

fn clip_test(sb: &mut SceneBuilder, params: &mut SceneParams) {
    let clip = {
        const X0: f64 = 50.0;
        const Y0: f64 = 0.0;
        const X1: f64 = 200.0;
        const Y1: f64 = 500.0;
        [
            PathEl::MoveTo((X0, Y0).into()),
            PathEl::LineTo((X1, Y0).into()),
            PathEl::LineTo((X1, Y0 + (Y1 - Y0)).into()),
            PathEl::LineTo((X1 + (X0 - X1), Y1).into()),
            PathEl::LineTo((X0, Y1).into()),
            PathEl::ClosePath,
        ]
    };
    sb.push_layer(Mix::Clip, 1.0, Affine::IDENTITY, &clip);
    {
        let text_size = 60.0 + 40.0 * (params.time as f32).sin();
        let s = "Some clipped text!";
        params.text.add(
            sb,
            None,
            text_size,
            None,
            Affine::translate((110.0, 100.0)),
            s,
        );
    }
    sb.pop_layer();

    let large_background_rect = kurbo::Rect::new(-1000.0, -1000.0, 2000.0, 2000.0);
    let inside_clip_rect = kurbo::Rect::new(11.0, 13.399999999999999, 59.0, 56.6);
    let outside_clip_rect = kurbo::Rect::new(
        12.599999999999998,
        12.599999999999998,
        57.400000000000006,
        57.400000000000006,
    );
    let clip_rect = kurbo::Rect::new(0.0, 0.0, 74.4, 339.20000000000005);
    let scale = 2.0;

    sb.push_layer(
        BlendMode {
            mix: peniko::Mix::Normal,
            compose: peniko::Compose::SrcOver,
        },
        1.0,
        Affine::new([scale, 0.0, 0.0, scale, 27.07470703125, 176.40660533027858]),
        &clip_rect,
    );

    sb.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::new([scale, 0.0, 0.0, scale, 27.07470703125, 176.40660533027858]),
        peniko::Color::rgb8(0, 0, 255),
        None,
        &large_background_rect,
    );
    sb.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::new([
            scale,
            0.0,
            0.0,
            scale,
            29.027636718750003,
            182.9755506427786,
        ]),
        peniko::Color::rgb8(0, 255, 0),
        None,
        &inside_clip_rect,
    );
    sb.fill(
        peniko::Fill::NonZero,
        kurbo::Affine::new([
            scale,
            0.0,
            0.0,
            scale,
            29.027636718750003,
            scale * 559.3583631427786,
        ]),
        peniko::Color::rgb8(255, 0, 0),
        None,
        &outside_clip_rect,
    );

    sb.pop_layer();
}

fn around_center(xform: Affine, center: Point) -> Affine {
    Affine::translate(center.to_vec2()) * xform * Affine::translate(-center.to_vec2())
}

fn make_diamond(cx: f64, cy: f64) -> [PathEl; 5] {
    const SIZE: f64 = 50.0;
    [
        PathEl::MoveTo(Point::new(cx, cy - SIZE)),
        PathEl::LineTo(Point::new(cx + SIZE, cy)),
        PathEl::LineTo(Point::new(cx, cy + SIZE)),
        PathEl::LineTo(Point::new(cx - SIZE, cy)),
        PathEl::ClosePath,
    ]
}

fn splash_screen(sb: &mut SceneBuilder, params: &mut SceneParams) {
    let strings = [
        "Vello test",
        "  Arrow keys: switch scenes",
        "  Space: reset transform",
        "  S: toggle stats",
        "  V: toggle vsync",
        "  M: cycle AA method",
        "  Q, E: rotate",
    ];
    // Tweak to make it fit with tiger
    let a = Affine::scale(0.11) * Affine::translate((-90.0, -50.0));
    for (i, s) in strings.iter().enumerate() {
        let text_size = if i == 0 { 60.0 } else { 40.0 };
        params.text.add(
            sb,
            None,
            text_size,
            None,
            a * Affine::translate((100.0, 100.0 + 60.0 * i as f64)),
            s,
        );
    }
}

fn splash_with_tiger() -> impl FnMut(&mut SceneBuilder, &mut SceneParams) {
    let contents = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../assets/Ghostscript_Tiger.svg"
    ));
    let mut tiger = crate::svg::svg_function_of("Ghostscript Tiger".to_string(), move || contents);
    move |sb, params| {
        tiger(sb, params);
        splash_screen(sb, params);
    }
}
