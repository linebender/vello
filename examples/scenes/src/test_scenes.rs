// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{ExampleScene, SceneConfig, SceneSet};
use vello::kurbo::{Affine, Cap};

/// All of the test scenes supported by Vello.
pub fn test_scenes() -> SceneSet {
    test_scenes_inner()
}

/// A macro which exports each passed scene indivudally
///
/// This is used to avoid having to repeatedly define a
macro_rules! export_scenes {
    ($(
        $(#[cfg($feature:meta)])?  // Optional feature gate
        $scene_name:ident($($scene:tt)+)
    ),*$(,)?) => {
        pub fn test_scenes_inner() -> SceneSet {
            let mut scenes = Vec::new();
            $(
                $(#[cfg($feature)])?
                {
                    scenes.push($scene_name());
                }
            )*
            SceneSet { scenes }
        }

        $(
            $(#[cfg($feature)])?
            pub fn $scene_name() -> ExampleScene {
                scene!($($scene)+)
            }
        )*
    };
}

/// A helper to create a shorthand name for a single scene.
/// Used in `export_scenes`.
macro_rules! scene {
    ($name: ident) => {
        scene!($name: false)
    };
    ($name: ident: animated) => {
        scene!($name: true)
    };
    ($name: ident: $animated: literal) => {
        scene!(impls::$name, stringify!($name), $animated)
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

export_scenes!(
    splash_with_tiger(impls::splash_with_tiger(), "splash_with_tiger", false),
    funky_paths(funky_paths),
    stroke_styles(impls::stroke_styles(Affine::IDENTITY), "stroke_styles", false),
    stroke_styles_non_uniform(impls::stroke_styles(Affine::scale_non_uniform(1.2, 0.7)), "stroke_styles (non-uniform scale)", false),
    stroke_styles_skew(impls::stroke_styles(Affine::skew(1., 0.)), "stroke_styles (skew)", false),
    emoji(emoji),
    tricky_strokes(tricky_strokes),
    fill_types(fill_types),
    cardioid_and_friends(cardioid_and_friends),
    animated_text(animated_text: animated),
    gradient_extend(gradient_extend),
    two_point_radial(two_point_radial),
    brush_transform(brush_transform: animated),
    blend_grid(blend_grid),
    deep_blend(deep_blend),
    many_clips(many_clips),
    conflation_artifacts(conflation_artifacts),
    labyrinth(labyrinth),
    robust_paths(robust_paths),
    base_color_test(base_color_test: animated),
    clip_test(clip_test: animated),
    longpathdash_butt(impls::longpathdash(Cap::Butt), "longpathdash (butt caps)", false),
    longpathdash_round(impls::longpathdash(Cap::Round), "longpathdash (round caps)", false),
    mmark(crate::mmark::MMark::new(80_000), "mmark", false),
    many_draw_objects(many_draw_objects),
    blurred_rounded_rect(blurred_rounded_rect),
    image_sampling(image_sampling),
    image_extend_modes(image_extend_modes),
    #[cfg(feature = "cosmic_text")] cosmic_text_scene(crate::cosmic_text_scene::CosmicTextScene::default(), "cosmic_text", false)
);

/// Implementations for the test scenes.
/// In a module because the exported [`ExampleScene`] creation functions use the same names.
mod impls {
    use std::f64::consts::{FRAC_1_SQRT_2, PI};
    use std::sync::Arc;

    use crate::SceneParams;
    use kurbo::RoundedRect;
    use rand::Rng;
    use rand::{rngs::StdRng, SeedableRng};
    use vello::kurbo::{
        Affine, BezPath, Cap, Circle, Ellipse, Join, PathEl, Point, Rect, Shape, Stroke, Vec2,
    };
    use vello::peniko::color::{palette, AlphaColor, Lch};
    use vello::peniko::*;
    use vello::*;

    const FLOWER_IMAGE: &[u8] = include_bytes!("../../assets/splash-flower.jpg");

    pub(super) fn emoji(scene: &mut Scene, params: &mut SceneParams) {
        let text_size = 120. + 20. * (params.time * 2.).sin() as f32;
        let s = "🎉🤠✅";
        params.text.add_colr_emoji_run(
            scene,
            text_size,
            Affine::translate(Vec2::new(100., 250.)),
            None,
            Fill::NonZero,
            s,
        );
        params.text.add_bitmap_emoji_run(
            scene,
            text_size,
            Affine::translate(Vec2::new(100., 500.)),
            None,
            Fill::NonZero,
            s,
        );
    }

    pub(super) fn funky_paths(scene: &mut Scene, _: &mut SceneParams) {
        use PathEl::*;
        let missing_movetos = [
            MoveTo((0., 0.).into()),
            LineTo((100.0, 100.0).into()),
            LineTo((100.0, 200.0).into()),
            ClosePath,
            LineTo((0.0, 400.0).into()),
            LineTo((100.0, 400.0).into()),
        ];
        let only_movetos = [MoveTo((0.0, 0.0).into()), MoveTo((100.0, 100.0).into())];
        let empty: [PathEl; 0] = [];
        scene.fill(
            Fill::NonZero,
            Affine::translate((100.0, 100.0)),
            palette::css::BLUE,
            None,
            &missing_movetos,
        );
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            palette::css::BLUE,
            None,
            &empty,
        );
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            palette::css::BLUE,
            None,
            &only_movetos,
        );
        scene.stroke(
            &Stroke::new(8.0),
            Affine::translate((100.0, 100.0)),
            palette::css::AQUA,
            None,
            &missing_movetos,
        );
    }

    pub(super) fn stroke_styles(transform: Affine) -> impl FnMut(&mut Scene, &mut SceneParams) {
        use PathEl::*;
        move |scene, params| {
            let colors = [
                Color::from_rgba8(140, 181, 236, 255),
                Color::from_rgba8(246, 236, 202, 255),
                Color::from_rgba8(201, 147, 206, 255),
                Color::from_rgba8(150, 195, 160, 255),
            ];
            let simple_stroke = [MoveTo((0., 0.).into()), LineTo((100., 0.).into())];
            let join_stroke = [
                MoveTo((0., 0.).into()),
                CurveTo((20., 0.).into(), (42.5, 5.).into(), (50., 25.).into()),
                CurveTo((57.5, 5.).into(), (80., 0.).into(), (100., 0.).into()),
            ];
            let miter_stroke = [
                MoveTo((0., 0.).into()),
                LineTo((90., 16.).into()),
                LineTo((0., 31.).into()),
                LineTo((90., 46.).into()),
            ];
            let closed_strokes = [
                MoveTo((0., 0.).into()),
                LineTo((90., 21.).into()),
                LineTo((0., 42.).into()),
                ClosePath,
                MoveTo((200., 0.).into()),
                CurveTo((100., 72.).into(), (300., 72.).into(), (200., 0.).into()),
                ClosePath,
                MoveTo((290., 0.).into()),
                CurveTo((200., 72.).into(), (400., 72.).into(), (310., 0.).into()),
                ClosePath,
            ];
            let cap_styles = [Cap::Butt, Cap::Square, Cap::Round];
            let join_styles = [Join::Bevel, Join::Miter, Join::Round];
            let miter_limits = [4., 6., 0.1, 10.];

            // Simple strokes with cap combinations
            let t = Affine::translate((60., 40.)) * Affine::scale(2.);
            let mut y = 0.;
            let mut color_idx = 0;
            for start in cap_styles {
                for end in cap_styles {
                    params.text.add(
                        scene,
                        None,
                        12.,
                        None,
                        Affine::translate((0., y)) * t,
                        &format!("Start cap: {:?}, End cap: {:?}", start, end),
                    );
                    scene.stroke(
                        &Stroke::new(20.).with_start_cap(start).with_end_cap(end),
                        Affine::translate((0., y + 30.)) * t * transform,
                        colors[color_idx],
                        None,
                        &simple_stroke,
                    );
                    y += 180.;
                    color_idx = (color_idx + 1) % colors.len();
                }
            }
            // Dashed strokes with cap combinations
            let t = Affine::translate((450., 0.)) * t;
            let mut y_max = y;
            y = 0.;
            for start in cap_styles {
                for end in cap_styles {
                    params.text.add(
                        scene,
                        None,
                        12.,
                        None,
                        Affine::translate((0., y)) * t,
                        &format!("Dashing - Start cap: {:?}, End cap: {:?}", start, end),
                    );
                    scene.stroke(
                        &Stroke::new(20.)
                            .with_start_cap(start)
                            .with_end_cap(end)
                            .with_dashes(0., [10.0, 21.0]),
                        Affine::translate((0., y + 30.)) * t * transform,
                        colors[color_idx],
                        None,
                        &simple_stroke,
                    );
                    y += 180.;
                    color_idx = (color_idx + 1) % colors.len();
                }
            }

            // Cap and join combinations
            let t = Affine::translate((550., 0.)) * t;
            y_max = y_max.max(y);
            y = 0.;
            for cap in cap_styles {
                for join in join_styles {
                    params.text.add(
                        scene,
                        None,
                        12.,
                        None,
                        Affine::translate((0., y)) * t,
                        &format!("Caps: {:?}, Joins: {:?}", cap, join),
                    );
                    scene.stroke(
                        &Stroke::new(20.).with_caps(cap).with_join(join),
                        Affine::translate((0., y + 30.)) * t * transform,
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
            y_max = y_max.max(y);
            y = 0.;
            for ml in miter_limits {
                params.text.add(
                    scene,
                    None,
                    12.,
                    None,
                    Affine::translate((0., y)) * t,
                    &format!("Miter limit: {}", ml),
                );
                scene.stroke(
                    &Stroke::new(10.)
                        .with_caps(Cap::Butt)
                        .with_join(Join::Miter)
                        .with_miter_limit(ml),
                    Affine::translate((0., y + 30.)) * t * transform,
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
                    scene,
                    None,
                    12.,
                    None,
                    Affine::translate((0., y)) * t,
                    &format!("Closed path with join: {:?}", join),
                );
                // The cap style is not important since a closed path shouldn't have any caps.
                scene.stroke(
                    &Stroke::new(10.)
                        .with_caps(cap_styles[i])
                        .with_join(*join)
                        .with_miter_limit(5.),
                    Affine::translate((0., y + 30.)) * t * transform,
                    colors[color_idx],
                    None,
                    &closed_strokes,
                );
                y += 180.;
                color_idx = (color_idx + 1) % colors.len();
            }
            y_max = y_max.max(y);
            // The closed_strokes has a maximum x of 400, `t` has a scale of `2.`
            // Give 50px of padding to account for `transform`
            let x_max = t.translation().x + 400. * 2. + 50.;
            params.resolution = Some((x_max, y_max).into());
        }
    }

    // This test has been adapted from Skia's "trickycubicstrokes" GM slide which can be found at
    // `github.com/google/skia/blob/0d4d11451c4f4e184305cbdbd67f6b3edfa4b0e3/gm/trickycubicstrokes.cpp`
    pub(super) fn tricky_strokes(scene: &mut Scene, params: &mut SceneParams) {
        use PathEl::*;
        let colors = [
            Color::from_rgba8(140, 181, 236, 255),
            Color::from_rgba8(246, 236, 202, 255),
            Color::from_rgba8(201, 147, 206, 255),
            Color::from_rgba8(150, 195, 160, 255),
        ];

        const CELL_SIZE: f64 = 200.;
        const STROKE_WIDTH: f64 = 30.;
        const NUM_COLS: usize = 5;

        pub(super) fn stroke_bounds(pts: &[(f64, f64); 4]) -> Rect {
            use kurbo::CubicBez;
            CubicBez::new(pts[0], pts[1], pts[2], pts[3])
                .bounding_box()
                .inflate(STROKE_WIDTH, STROKE_WIDTH)
        }

        pub(super) fn map_rect_to_rect(src: &Rect, dst: &Rect) -> (Affine, f64) {
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

        // Flat conic with a cusp: (1,1) (2,1) (1,1), weight: 1
        let flat_quad = [
            // moveTo(1., 1.),
            [(2., 1.), (1., 1.)],
        ];
        // Flat conic with a cusp: (1,1) (100,1) (25,1), weight: 0.3
        let flat_conic_as_quads = [
            // moveTo(1., 1.),
            [(2.232486, 1.000000), (3.471740, 1.000000)],
            [(4.710995, 1.000000), (5.949262, 1.000000)],
            [(7.187530, 1.000000), (8.417061, 1.000000)],
            [(9.646591, 1.000000), (10.859690, 1.000000)],
            [(12.072789, 1.000000), (13.261865, 1.000000)],
            [(14.450940, 1.000000), (15.608549, 1.000000)],
            [(16.766161, 1.000000), (17.885059, 1.000000)],
            [(19.003958, 1.000000), (20.077141, 1.000000)],
            [(21.150328, 1.000000), (22.171083, 1.000000)],
            [(23.191839, 1.000000), (24.153776, 1.000000)],
            [(25.115715, 1.000000), (26.012812, 1.000000)],
            [(26.909912, 1.000000), (27.736557, 1.000000)],
            [(28.563202, 1.000000), (29.314220, 1.000000)],
            [(30.065239, 1.000000), (30.735928, 1.000000)],
            [(31.406620, 1.000000), (31.992788, 1.000000)],
            [(32.578957, 1.000000), (33.076927, 1.000000)],
            [(33.574905, 1.000000), (33.981567, 1.000000)],
            [(34.388233, 1.000000), (34.701038, 1.000000)],
            [(35.013851, 1.000000), (35.230850, 1.000000)],
            [(35.447845, 1.000000), (35.567669, 1.000000)],
            [(35.687500, 1.000000), (35.709404, 1.000000)],
            [(35.731312, 1.000000), (35.655155, 1.000000)],
            [(35.579006, 1.000000), (35.405273, 1.000000)],
            [(35.231541, 1.000000), (34.961311, 1.000000)],
            [(34.691086, 1.000000), (34.326057, 1.000000)],
            [(33.961029, 1.000000), (33.503479, 1.000000)],
            [(33.045937, 1.000000), (32.498734, 1.000000)],
            [(31.951530, 1.000000), (31.318098, 1.000000)],
            [(30.684669, 1.000000), (29.968971, 1.000000)],
            [(29.253277, 1.000000), (28.459791, 1.000000)],
            [(27.666309, 1.000000), (26.800005, 1.000000)],
            [(25.933704, 1.000000), (25.000000, 1.000000)],
        ];
        // Flat conic with a cusp: (1,1) (100,1) (25,1), weight: 1.5
        let bigger_flat_conic_as_quads = [
            // moveTo(1., 1.),
            [(8.979845, 1.000000), (15.795975, 1.000000)],
            [(22.612104, 1.000000), (28.363287, 1.000000)],
            [(34.114471, 1.000000), (38.884045, 1.000000)],
            [(43.653618, 1.000000), (47.510696, 1.000000)],
            [(51.367767, 1.000000), (54.368233, 1.000000)],
            [(57.368698, 1.000000), (59.556030, 1.000000)],
            [(61.743366, 1.000000), (63.149269, 1.000000)],
            [(64.555168, 1.000000), (65.200005, 1.000000)],
            [(65.844841, 1.000000), (65.737961, 1.000000)],
            [(65.631073, 1.000000), (64.770912, 1.000000)],
            [(63.910763, 1.000000), (62.284878, 1.000000)],
            [(60.658997, 1.000000), (58.243816, 1.000000)],
            [(55.828640, 1.000000), (52.589172, 1.000000)],
            [(49.349705, 1.000000), (45.239006, 1.000000)],
            [(41.128315, 1.000000), (36.086826, 1.000000)],
            [(31.045338, 1.000000), (25.000000, 1.000000)],
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
            scene.stroke(
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

        let flat_curves = [
            flat_quad.as_slice(),
            flat_conic_as_quads.as_slice(),
            bigger_flat_conic_as_quads.as_slice(),
        ];
        for quads in flat_curves.iter() {
            let mut path = BezPath::new();
            path.push(MoveTo((1., 1.).into()));
            for quad in quads.iter() {
                path.push(QuadTo(quad[0].into(), quad[1].into()));
            }
            let x = (idx % NUM_COLS) as f64 * CELL_SIZE;
            let y = (idx / NUM_COLS) as f64 * CELL_SIZE;
            let cell = Rect::new(x, y, x + CELL_SIZE, y + CELL_SIZE);
            let bounds = path.bounding_box().inflate(STROKE_WIDTH, STROKE_WIDTH);
            let (t, s) = map_rect_to_rect(&bounds, &cell);
            scene.stroke(
                &Stroke::new(STROKE_WIDTH / s)
                    .with_caps(Cap::Butt)
                    .with_join(Join::Miter),
                t,
                colors[color_idx],
                None,
                &path,
            );
            color_idx = (color_idx + 1) % colors.len();
            idx += 1;
        }

        let curve_count = tricky_cubics.len() + flat_curves.len();
        params.resolution = Some(Vec2::new(
            CELL_SIZE * NUM_COLS as f64,
            CELL_SIZE * (1 + curve_count / NUM_COLS) as f64,
        ));
    }

    pub(super) fn fill_types(scene: &mut Scene, params: &mut SceneParams) {
        use PathEl::*;
        params.resolution = Some((1400., 700.).into());
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
            params.text.add(scene, None, 24., None, t, rule.1);
            let t = Affine::translate((0., 5.)) * t * scale;
            scene.fill(Fill::NonZero, t, palette::css::GRAY, None, &rect);
            scene.fill(
                rule.0,
                Affine::translate((0., 10.)) * t,
                palette::css::YELLOW,
                None,
                &rule.2,
            );
        }

        // Draw blends
        let t = Affine::translate((700., 0.)) * t;
        for (i, rule) in rules.iter().enumerate() {
            let t = Affine::translate(((i % 2) as f64 * 306., (i / 2) as f64 * 340.)) * t;
            params.text.add(scene, None, 24., None, t, rule.1);
            let t = Affine::translate((0., 5.)) * t * scale;
            scene.fill(Fill::NonZero, t, palette::css::GRAY, None, &rect);
            scene.fill(
                rule.0,
                Affine::translate((0., 10.)) * t,
                palette::css::YELLOW,
                None,
                &rule.2,
            );
            scene.fill(
                rule.0,
                Affine::translate((0., 10.)) * t * Affine::rotate(0.06),
                Color::new([0., 1., 0.7, 0.6]),
                None,
                &rule.2,
            );
            scene.fill(
                rule.0,
                Affine::translate((0., 10.)) * t * Affine::rotate(-0.06),
                Color::new([0.9, 0.7, 0.5, 0.6]),
                None,
                &rule.2,
            );
        }
    }

    pub(super) fn cardioid_and_friends(scene: &mut Scene, _: &mut SceneParams) {
        render_cardioid(scene);
        render_clip_test(scene);
        render_alpha_test(scene);
        //render_tiger(scene, false);
    }

    pub(super) fn longpathdash(cap: Cap) -> impl FnMut(&mut Scene, &mut SceneParams) {
        use PathEl::*;
        move |scene, _| {
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
            scene.stroke(
                &Stroke::new(1.0)
                    .with_caps(cap)
                    .with_join(Join::Bevel)
                    .with_dashes(0.0, [1.0, 1.0]),
                Affine::translate((50.0, 50.0)),
                palette::css::YELLOW,
                None,
                &path,
            );
        }
    }

    pub(super) fn animated_text(scene: &mut Scene, params: &mut SceneParams) {
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
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            palette::css::GRAY,
            None,
            &rect,
        );
        let text_size = 60.0 + 40.0 * (params.time as f32).sin();
        let s = "\u{1f600}hello Vello text!";
        params.text.add(
            scene,
            None,
            text_size,
            None,
            Affine::translate((110.0, 600.0)),
            s,
        );
        params.text.add_run(
            scene,
            None,
            text_size,
            palette::css::WHITE,
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
            scene,
            None,
            72.0,
            &[("wght", weight), ("wdth", width)],
            palette::css::WHITE,
            Affine::translate((110.0, 800.0)),
            // Add a skew to simulate an oblique font.
            None,
            Fill::NonZero,
            "And some Vello\ntext with a newline",
        );
        let th = params.time;
        let center = Point::new(500.0, 500.0);
        let mut p1 = center;
        p1.x += 400.0 * th.cos();
        p1.y += 400.0 * th.sin();
        scene.stroke(
            &Stroke::new(5.0),
            Affine::IDENTITY,
            palette::css::MAROON,
            None,
            &[MoveTo(center), LineTo(p1)],
        );
        scene.fill(
            Fill::NonZero,
            Affine::translate((150.0, 150.0)) * Affine::scale(0.2),
            palette::css::RED,
            None,
            &rect,
        );
        let alpha = params.time.sin() as f32 * 0.5 + 0.5;
        scene.push_layer(Mix::Normal, alpha, Affine::IDENTITY, &rect);
        scene.fill(
            Fill::NonZero,
            Affine::translate((100.0, 100.0)) * Affine::scale(0.2),
            palette::css::BLUE,
            None,
            &rect,
        );
        scene.fill(
            Fill::NonZero,
            Affine::translate((200.0, 200.0)) * Affine::scale(0.2),
            palette::css::GREEN,
            None,
            &rect,
        );
        scene.pop_layer();
        scene.fill(
            Fill::NonZero,
            Affine::translate((400.0, 100.0)),
            palette::css::PURPLE,
            None,
            &star,
        );
        scene.fill(
            Fill::EvenOdd,
            Affine::translate((500.0, 100.0)),
            palette::css::PURPLE,
            None,
            &star,
        );
        scene.draw_image(
            &piet_logo,
            Affine::translate((800.0, 50.0)) * Affine::rotate(20f64.to_radians()),
        );
    }

    pub(super) fn brush_transform(scene: &mut Scene, params: &mut SceneParams) {
        let th = params.time;
        let linear = Gradient::new_linear((0.0, 0.0), (0.0, 200.0)).with_stops([
            palette::css::RED,
            palette::css::GREEN,
            palette::css::BLUE,
        ]);
        scene.fill(
            Fill::NonZero,
            Affine::rotate(25f64.to_radians()) * Affine::scale_non_uniform(2.0, 1.0),
            &Gradient::new_radial((200.0, 200.0), 80.0).with_stops([
                palette::css::RED,
                palette::css::GREEN,
                palette::css::BLUE,
            ]),
            None,
            &Rect::from_origin_size((100.0, 100.0), (200.0, 200.0)),
        );
        scene.fill(
            Fill::NonZero,
            Affine::translate((200.0, 600.0)),
            &linear,
            Some(around_center(Affine::rotate(th), Point::new(200.0, 100.0))),
            &Rect::from_origin_size(Point::default(), (400.0, 200.0)),
        );
        scene.stroke(
            &Stroke::new(40.0),
            Affine::translate((800.0, 600.0)),
            &linear,
            Some(around_center(Affine::rotate(th), Point::new(200.0, 100.0))),
            &Rect::from_origin_size(Point::default(), (400.0, 200.0)),
        );
    }

    pub(super) fn gradient_extend(scene: &mut Scene, params: &mut SceneParams) {
        enum Kind {
            Linear,
            Radial,
            Sweep,
        }
        pub(super) fn square(scene: &mut Scene, kind: Kind, transform: Affine, extend: Extend) {
            let colors = [palette::css::RED, palette::css::LIME, palette::css::BLUE];
            let width = 300f64;
            let height = 300f64;
            let gradient: Brush = match kind {
                Kind::Linear => {
                    Gradient::new_linear((width * 0.35, height * 0.5), (width * 0.65, height * 0.5))
                        .with_stops(colors)
                        .with_extend(extend)
                        .into()
                }
                Kind::Radial => {
                    let center = (width * 0.5, height * 0.5);
                    let radius = (width * 0.25) as f32;
                    Gradient::new_two_point_radial(center, radius * 0.25, center, radius)
                        .with_stops(colors)
                        .with_extend(extend)
                        .into()
                }
                Kind::Sweep => Gradient::new_sweep(
                    (width * 0.5, height * 0.5),
                    30f32.to_radians(),
                    150f32.to_radians(),
                )
                .with_stops(colors)
                .with_extend(extend)
                .into(),
            };
            scene.fill(
                Fill::NonZero,
                transform,
                &gradient,
                None,
                &Rect::new(0.0, 0.0, width, height),
            );
        }
        let extend_modes = [Extend::Pad, Extend::Repeat, Extend::Reflect];
        for (x, extend) in extend_modes.iter().enumerate() {
            for (y, kind) in [Kind::Linear, Kind::Radial, Kind::Sweep]
                .into_iter()
                .enumerate()
            {
                let transform =
                    Affine::translate((x as f64 * 350.0 + 50.0, y as f64 * 350.0 + 100.0));
                square(scene, kind, transform, *extend);
            }
        }
        for (i, label) in ["Pad", "Repeat", "Reflect"].iter().enumerate() {
            let x = i as f64 * 350.0 + 50.0;
            params.text.add(
                scene,
                None,
                32.0,
                Some(&palette::css::WHITE.into()),
                Affine::translate((x, 70.0)),
                label,
            );
        }
        params.resolution = Some((1200.0, 1200.0).into());
    }

    pub(super) fn two_point_radial(scene: &mut Scene, _params: &mut SceneParams) {
        pub(super) fn make(
            scene: &mut Scene,
            x0: f64,
            y0: f64,
            r0: f32,
            x1: f64,
            y1: f64,
            r1: f32,
            transform: Affine,
            extend: Extend,
        ) {
            let colors = [
                palette::css::RED,
                palette::css::YELLOW,
                Color::from_rgba8(6, 85, 186, 255),
            ];
            let width = 400f64;
            let height = 200f64;
            let rect = Rect::new(0.0, 0.0, width, height);
            scene.fill(Fill::NonZero, transform, palette::css::WHITE, None, &rect);
            scene.fill(
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
            scene.stroke(
                &Stroke::new(stroke_width),
                transform,
                palette::css::BLACK,
                None,
                &Ellipse::new((x0, y0), (r0, r0), 0.0),
            );
            scene.stroke(
                &Stroke::new(stroke_width),
                transform,
                palette::css::BLACK,
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
                scene,
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
                scene,
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
                scene,
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
                scene,
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
                scene,
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

    pub(super) fn blend_grid(scene: &mut Scene, _: &mut SceneParams) {
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
            scene.append(&square, Some(transform));
        }
    }

    pub(super) fn deep_blend(scene: &mut Scene, params: &mut SceneParams) {
        params.resolution = Some(Vec2::new(1000., 1000.));
        let main_rect = Rect::from_origin_size((10., 10.), (900., 900.));
        scene.fill(
            Fill::EvenOdd,
            Affine::IDENTITY,
            palette::css::RED,
            None,
            &main_rect,
        );
        let options = [
            (800., palette::css::AQUA),
            (700., palette::css::RED),
            (600., palette::css::ALICE_BLUE),
            (500., palette::css::YELLOW),
            (400., palette::css::GREEN),
            (300., palette::css::BLUE),
            (200., palette::css::ORANGE),
            (100., palette::css::WHITE),
        ];
        let mut depth = 0;
        for (width, color) in &options[..params.complexity.min(options.len() - 1)] {
            scene.push_layer(
                Mix::Normal,
                0.9,
                Affine::IDENTITY,
                &Rect::from_origin_size((10., 10.), (*width, *width)),
            );
            scene.fill(Fill::EvenOdd, Affine::IDENTITY, color, None, &main_rect);
            depth += 1;
        }
        for _ in 0..depth {
            scene.pop_layer();
        }
    }

    pub(super) fn many_clips(scene: &mut Scene, params: &mut SceneParams) {
        params.resolution = Some(Vec2::new(1000., 1000.));
        let mut rng = StdRng::seed_from_u64(42);
        let mut base_tri = BezPath::new();
        base_tri.move_to((-50.0, 0.0));
        base_tri.line_to((25.0, -43.3));
        base_tri.line_to((25.0, 43.3));
        for y in 0..10 {
            for x in 0..10 {
                let translate =
                    Affine::translate((100. * (x as f64 + 0.5), 100. * (y as f64 + 0.5)));
                const CLIPS_PER_FILL: usize = 3;
                for _ in 0..CLIPS_PER_FILL {
                    let rot = Affine::rotate(rng.gen_range(0.0..PI));
                    scene.push_layer(Mix::Clip, 1.0, translate * rot, &base_tri);
                }
                let rot = Affine::rotate(rng.gen_range(0.0..PI));
                let color = Color::new([rng.r#gen(), rng.r#gen(), rng.r#gen(), 1.]);
                scene.fill(Fill::NonZero, translate * rot, color, None, &base_tri);
                for _ in 0..CLIPS_PER_FILL {
                    scene.pop_layer();
                }
            }
        }
    }

    // Support functions

    pub(super) fn render_cardioid(scene: &mut Scene) {
        let n = 601;
        let dth = PI * 2.0 / (n as f64);
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
        scene.stroke(
            &Stroke::new(2.0),
            Affine::IDENTITY,
            palette::css::BLUE,
            None,
            &path,
        );
    }

    pub(super) fn render_clip_test(scene: &mut Scene) {
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
            scene.push_layer(Mix::Clip, 1.0, Affine::IDENTITY, &path);
        }
        let rect = Rect::new(X0, Y0, X1, Y1);
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            palette::css::LIME,
            None,
            &rect,
        );
        for _ in 0..N {
            scene.pop_layer();
        }
    }

    pub(super) fn render_alpha_test(scene: &mut Scene) {
        // Alpha compositing tests.
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            palette::css::RED,
            None,
            &make_diamond(1024.0, 100.0),
        );
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            palette::css::LIME.with_alpha(0.5),
            None,
            &make_diamond(1024.0, 125.0),
        );
        scene.push_layer(
            Mix::Clip,
            1.0,
            Affine::IDENTITY,
            &make_diamond(1024.0, 150.0),
        );
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            palette::css::BLUE.with_alpha(0.5),
            None,
            &make_diamond(1024.0, 175.0),
        );
        scene.pop_layer();
    }

    pub(super) fn render_blend_square(scene: &mut Scene, blend: BlendMode, transform: Affine) {
        // Inspired by https://developer.mozilla.org/en-US/docs/Web/CSS/mix-blend-mode
        let rect = Rect::from_origin_size(Point::new(0., 0.), (200., 200.));
        let linear = Gradient::new_linear((0.0, 0.0), (200.0, 0.0))
            .with_stops([palette::css::BLACK, palette::css::WHITE]);
        scene.fill(Fill::NonZero, transform, &linear, None, &rect);
        const GRADIENTS: &[(f64, f64, Color)] = &[
            (150., 0., Color::from_rgba8(255, 240, 64, 255)),
            (175., 100., Color::from_rgba8(255, 96, 240, 255)),
            (125., 200., Color::from_rgba8(64, 192, 255, 255)),
        ];
        for (x, y, c) in GRADIENTS {
            let color2 = c.with_alpha(0.);
            let radial = Gradient::new_radial((*x, *y), 100.0).with_stops([*c, color2]);
            scene.fill(Fill::NonZero, transform, &radial, None, &rect);
        }
        const COLORS: &[Color] = &[palette::css::RED, palette::css::LIME, palette::css::BLUE];
        scene.push_layer(Mix::Normal, 1.0, transform, &rect);
        for (i, c) in COLORS.iter().enumerate() {
            let linear = Gradient::new_linear((0.0, 0.0), (0.0, 200.0))
                .with_stops([palette::css::WHITE, *c]);
            scene.push_layer(blend, 1.0, transform, &rect);
            // squash the ellipse
            let a = transform
                * Affine::translate((100., 100.))
                * Affine::rotate(std::f64::consts::FRAC_PI_3 * (i * 2 + 1) as f64)
                * Affine::scale_non_uniform(1.0, 0.357)
                * Affine::translate((-100., -100.));
            scene.fill(
                Fill::NonZero,
                a,
                &linear,
                None,
                &Ellipse::new((100., 100.), (90., 90.), 0.),
            );
            scene.pop_layer();
        }
        scene.pop_layer();
    }

    pub(super) fn blend_square(blend: BlendMode) -> Scene {
        let mut fragment = Scene::default();
        render_blend_square(&mut fragment, blend, Affine::IDENTITY);
        fragment
    }

    pub(super) fn conflation_artifacts(scene: &mut Scene, _: &mut SceneParams) {
        use PathEl::*;
        const N: f64 = 50.0;
        const S: f64 = 4.0;

        let scale = Affine::scale(S);
        let x = N + 0.5; // Fractional pixel offset reveals the problem on axis-aligned edges.
        let mut y = N;

        let bg_color = Color::from_rgba8(255, 194, 19, 255);
        let fg_color = Color::from_rgba8(12, 165, 255, 255);

        // Two adjacent triangles touching at diagonal edge with opposing winding numbers
        scene.fill(
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
        scene.fill(
            Fill::EvenOdd,
            Affine::translate((x, y)) * scale,
            bg_color,
            None,
            &Rect::new(0.0, 0.0, N, N),
        );
        scene.fill(
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
        scene.fill(
            Fill::EvenOdd,
            Affine::translate((x, y)) * scale,
            bg_color,
            None,
            &Rect::new(0.0, 0.0, N, N),
        );
        scene.fill(
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

    pub(super) fn labyrinth(scene: &mut Scene, _: &mut SceneParams) {
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
        scene.fill(
            Fill::NonZero,
            Affine::translate((20.5, 20.5)) * Affine::scale(80.0),
            Color::from_rgba8(0x70, 0x80, 0x80, 0xff),
            None,
            &path,
        );
    }

    pub(super) fn robust_paths(scene: &mut Scene, _: &mut SceneParams) {
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
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            palette::css::YELLOW,
            None,
            &path,
        );
        scene.fill(
            Fill::EvenOdd,
            Affine::translate((300.0, 0.0)),
            palette::css::LIME,
            None,
            &path,
        );

        path.move_to((8.0, 4.0));
        path.line_to((8.0, 40.0));
        path.line_to((260.0, 40.0));
        path.line_to((260.0, 4.0));
        path.close_path();
        scene.fill(
            Fill::NonZero,
            Affine::translate((0.0, 100.0)),
            palette::css::YELLOW,
            None,
            &path,
        );
        scene.fill(
            Fill::EvenOdd,
            Affine::translate((300.0, 100.0)),
            palette::css::LIME,
            None,
            &path,
        );
    }

    pub(super) fn base_color_test(scene: &mut Scene, params: &mut SceneParams) {
        // Cycle through the hue value every 5 seconds (t % 5) * 360/5
        let color = AlphaColor::<Lch>::new([80., 80., (params.time % 5.) as f32 * 72., 1.]);
        params.base_color = Some(color.convert());

        // Blend a white square over it.
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            palette::css::WHITE.with_alpha(0.5),
            None,
            &Rect::new(50.0, 50.0, 500.0, 500.0),
        );
    }

    pub(super) fn clip_test(scene: &mut Scene, params: &mut SceneParams) {
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
        scene.push_layer(Mix::Clip, 1.0, Affine::IDENTITY, &clip);
        {
            let text_size = 60.0 + 40.0 * (params.time as f32).sin();
            let s = "Some clipped text!";
            params.text.add(
                scene,
                None,
                text_size,
                None,
                Affine::translate((110.0, 100.0)),
                s,
            );
        }
        scene.pop_layer();

        let large_background_rect = Rect::new(-1000.0, -1000.0, 2000.0, 2000.0);
        let inside_clip_rect = Rect::new(11.0, 13.399999999999999, 59.0, 56.6);
        let outside_clip_rect = Rect::new(
            12.599999999999998,
            12.599999999999998,
            57.400000000000006,
            57.400000000000006,
        );
        let clip_rect = Rect::new(0.0, 0.0, 74.4, 339.20000000000005);
        let scale = 2.0;

        scene.push_layer(
            BlendMode {
                mix: Mix::Normal,
                compose: Compose::SrcOver,
            },
            1.0,
            Affine::new([scale, 0.0, 0.0, scale, 27.07470703125, 176.40660533027858]),
            &clip_rect,
        );

        scene.fill(
            Fill::NonZero,
            Affine::new([scale, 0.0, 0.0, scale, 27.07470703125, 176.40660533027858]),
            palette::css::BLUE,
            None,
            &large_background_rect,
        );
        scene.fill(
            Fill::NonZero,
            Affine::new([
                scale,
                0.0,
                0.0,
                scale,
                29.027636718750003,
                182.9755506427786,
            ]),
            palette::css::LIME,
            None,
            &inside_clip_rect,
        );
        scene.fill(
            Fill::NonZero,
            Affine::new([
                scale,
                0.0,
                0.0,
                scale,
                29.027636718750003,
                scale * 559.3583631427786,
            ]),
            palette::css::RED,
            None,
            &outside_clip_rect,
        );

        scene.pop_layer();
    }

    pub(super) fn around_center(xform: Affine, center: Point) -> Affine {
        Affine::translate(center.to_vec2()) * xform * Affine::translate(-center.to_vec2())
    }

    pub(super) fn make_diamond(cx: f64, cy: f64) -> [PathEl; 5] {
        const SIZE: f64 = 50.0;
        [
            PathEl::MoveTo(Point::new(cx, cy - SIZE)),
            PathEl::LineTo(Point::new(cx + SIZE, cy)),
            PathEl::LineTo(Point::new(cx, cy + SIZE)),
            PathEl::LineTo(Point::new(cx - SIZE, cy)),
            PathEl::ClosePath,
        ]
    }

    pub(super) fn many_draw_objects(scene: &mut Scene, params: &mut SceneParams) {
        const N_WIDE: usize = 300;
        const N_HIGH: usize = 300;
        const SCENE_WIDTH: f64 = 2000.0;
        const SCENE_HEIGHT: f64 = 1500.0;
        params.resolution = Some((SCENE_WIDTH, SCENE_HEIGHT).into());
        for j in 0..N_HIGH {
            let y = (j as f64 + 0.5) * (SCENE_HEIGHT / N_HIGH as f64);
            for i in 0..N_WIDE {
                let x = (i as f64 + 0.5) * (SCENE_WIDTH / N_WIDE as f64);
                let c = Circle::new((x, y), 3.0);
                scene.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    palette::css::YELLOW,
                    None,
                    &c,
                );
            }
        }
    }

    pub(super) fn splash_screen(scene: &mut Scene, params: &mut SceneParams) {
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
                scene,
                None,
                text_size,
                None,
                a * Affine::translate((100.0, 100.0 + 60.0 * i as f64)),
                s,
            );
        }
    }

    pub(super) fn splash_with_tiger() -> impl FnMut(&mut Scene, &mut SceneParams) {
        let contents = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../assets/Ghostscript_Tiger.svg"
        ));
        let mut tiger =
            crate::svg::svg_function_of("Ghostscript Tiger".to_string(), move || contents);
        move |scene, params| {
            tiger(scene, params);
            splash_screen(scene, params);
        }
    }

    pub(super) fn blurred_rounded_rect(scene: &mut Scene, params: &mut SceneParams) {
        params.resolution = Some(Vec2::new(1200., 1200.));
        params.base_color = Some(palette::css::WHITE);

        let rect = Rect::from_center_size((0.0, 0.0), (300.0, 240.0));
        let radius = 50.0;
        scene.draw_blurred_rounded_rect(
            Affine::translate((300.0, 300.0)),
            rect,
            palette::css::BLUE,
            radius,
            params.time.sin() * 50.0 + 50.0,
        );

        // Skewed affine transformation.
        scene.draw_blurred_rounded_rect(
            Affine::translate((900.0, 300.0)) * Affine::skew(20f64.to_radians().tan(), 0.0),
            rect,
            palette::css::BLACK,
            radius,
            params.time.sin() * 50.0 + 50.0,
        );

        // Circle.
        scene.draw_blurred_rounded_rect(
            Affine::IDENTITY,
            Rect::new(100.0, 800.0, 400.0, 1100.0),
            palette::css::BLACK,
            150.0,
            params.time.sin() * 50.0 + 50.0,
        );

        // Radius larger than one size.
        scene.draw_blurred_rounded_rect(
            Affine::IDENTITY,
            Rect::new(600.0, 800.0, 900.0, 900.0),
            palette::css::BLACK,
            150.0,
            params.time.sin() * 50.0 + 50.0,
        );

        // An emulated box shadow, to demonstrate the use of `draw_blurred_rounded_rect_in`.
        let std_dev = params.time.sin() * 50.0 + 50.0;
        let kernel_size = 2.5 * std_dev;

        // TODO: Add utils to Kurbo for ad-hoc composed shapes
        let shape = BezPath::from_iter(
            rect.inflate(kernel_size, kernel_size)
                .path_elements(0.1)
                .chain(
                    RoundedRect::from_rect(rect, radius)
                        .to_path(0.1)
                        .reverse_subpaths(),
                ),
        );
        scene.draw_blurred_rounded_rect_in(
            &shape,
            Affine::translate((600.0, 600.0)) * Affine::scale_non_uniform(2.2, 0.9),
            rect,
            palette::css::BLACK,
            radius,
            std_dev,
        );
    }

    pub(super) fn image_sampling(scene: &mut Scene, params: &mut SceneParams) {
        params.resolution = Some(Vec2::new(1100., 1100.));
        params.base_color = Some(palette::css::WHITE);
        let mut blob: Vec<u8> = Vec::new();
        [
            palette::css::RED,
            palette::css::BLUE,
            palette::css::CYAN,
            palette::css::MAGENTA,
        ]
        .iter()
        .for_each(|c| {
            blob.extend(c.premultiply().to_rgba8().to_u8_array());
        });
        let data = Blob::new(Arc::new(blob));
        let image = Image::new(data, Format::Rgba8, 2, 2);

        scene.draw_image(
            &image,
            Affine::scale(200.).then_translate((100., 100.).into()),
        );
        scene.draw_image(
            &image,
            Affine::translate((-1., -1.))
                // 45° rotation
                .then_rotate(PI / 4.)
                .then_translate((1., 1.).into())
                // So the major axis is sqrt(2.) larger
                .then_scale(200. * FRAC_1_SQRT_2)
                .then_translate((100., 600.0).into()),
        );
        scene.draw_image(
            &image,
            Affine::scale_non_uniform(100., 200.).then_translate((600.0, 100.0).into()),
        );
        scene.draw_image(
            &image,
            Affine::skew(0.1, 0.25)
                .then_scale(200.0)
                .then_translate((600.0, 600.0).into()),
        );
    }

    pub(super) fn image_extend_modes(scene: &mut Scene, params: &mut SceneParams) {
        params.resolution = Some(Vec2::new(1500., 1500.));
        params.base_color = Some(palette::css::WHITE);
        let mut blob: Vec<u8> = Vec::new();
        [
            palette::css::RED,
            palette::css::BLUE,
            palette::css::CYAN,
            palette::css::MAGENTA,
        ]
        .iter()
        .for_each(|c| {
            blob.extend(c.premultiply().to_rgba8().to_u8_array());
        });
        let data = Blob::new(Arc::new(blob));
        let image = Image::new(data, Format::Rgba8, 2, 2);
        let image = image.with_extend(Extend::Pad);
        // Pad extend mode
        scene.fill(
            Fill::NonZero,
            Affine::scale(100.).then_translate((100., 100.).into()),
            &image,
            Some(Affine::translate((2., 2.)).then_scale(100.)),
            &Rect::new(0., 0., 6., 6.),
        );
        let image = image.with_extend(Extend::Reflect);
        scene.fill(
            Fill::NonZero,
            Affine::scale(100.).then_translate((100., 800.).into()),
            &image,
            Some(Affine::translate((2., 2.))),
            &Rect::new(0., 0., 6., 6.),
        );
        let image = image.with_extend(Extend::Repeat);
        scene.fill(
            Fill::NonZero,
            Affine::scale(100.).then_translate((800., 100.).into()),
            &image,
            Some(Affine::translate((2., 2.))),
            &Rect::new(0., 0., 6., 6.),
        );
    }
}
