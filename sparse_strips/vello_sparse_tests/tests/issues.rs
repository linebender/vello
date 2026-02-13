// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for GitHub issues.

use crate::renderer::Renderer;
use std::sync::Arc;
use vello_common::color::PremulRgba8;
use vello_common::color::palette::css::{DARK_BLUE, LIME, REBECCA_PURPLE};
use vello_common::kurbo::{Affine, BezPath, Rect, Shape, Stroke};
use vello_common::paint::Image;
use vello_common::peniko::{
    Color, ColorStop, Fill, Gradient, ImageQuality, ImageSampler, InterpolationAlphaSpace,
};
use vello_common::pixmap::Pixmap;
use vello_cpu::color::palette::css::{BLACK, RED};
use vello_cpu::peniko::{Compose, Extend};
use vello_cpu::{Level, RenderContext, RenderMode, RenderSettings};
use vello_dev_macros::vello_test;

#[vello_test(width = 8, height = 8)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/2
fn incorrect_filling_1(ctx: &mut impl Renderer) {
    let mut p = BezPath::default();
    p.move_to((4.0, 0.0));
    p.line_to((8.0, 4.0));
    p.line_to((4.0, 8.0));
    p.line_to((0.0, 4.0));
    p.close_path();

    ctx.set_paint(LIME);
    ctx.fill_path(&p);
}

#[vello_test(width = 64, height = 64)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/2
fn incorrect_filling_2(ctx: &mut impl Renderer) {
    let mut p = BezPath::default();
    p.move_to((16.0, 16.0));
    p.line_to((48.0, 16.0));
    p.line_to((48.0, 48.0));
    p.line_to((16.0, 48.0));
    p.close_path();

    ctx.set_paint(LIME);
    ctx.fill_path(&p);
}

#[vello_test(width = 9, height = 9)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/2
fn incorrect_filling_3(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((4.00001, 1e-45));
    path.line_to((8.00001, 4.00001));
    path.line_to((4.00001, 8.00001));
    path.line_to((1e-45, 4.00001));
    path.close_path();

    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test(width = 64, height = 64)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/2
fn incorrect_filling_4(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((16.000002, 8.));
    path.line_to((20.000002, 8.));
    path.line_to((24.000002, 8.));
    path.line_to((28.000002, 8.));
    path.line_to((32.000002, 8.));
    path.line_to((32.000002, 9.));
    path.line_to((28.000002, 9.));
    path.line_to((24.000002, 9.));
    path.line_to((20.000002, 9.));
    path.line_to((16.000002, 9.));
    path.close_path();

    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test(width = 32, height = 32)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/2
fn incorrect_filling_5(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((16., 8.));
    path.line_to((16., 9.));
    path.line_to((32., 9.));
    path.line_to((32., 8.));
    path.close_path();

    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test(width = 32, height = 32)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/2
fn incorrect_filling_6(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((16., 8.));
    path.line_to((31.999998, 8.));
    path.line_to((31.999998, 9.));
    path.line_to((16., 9.));
    path.close_path();

    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test(width = 32, height = 32)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/2
fn incorrect_filling_7(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((32.000002, 9.));
    path.line_to((28., 9.));
    path.line_to((28., 8.));
    path.line_to((32.000002, 8.));
    path.close_path();

    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test(width = 32, height = 32)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/2
fn incorrect_filling_8(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((16.000427, 8.));
    path.line_to((20.000427, 8.));
    path.line_to((24.000427, 8.));
    path.line_to((28.000427, 8.));
    path.line_to((32.000427, 8.));
    path.line_to((32.000427, 9.));
    path.line_to((28.000427, 9.));
    path.line_to((24.000427, 9.));
    path.line_to((20.000427, 9.));
    path.line_to((16.000427, 9.));
    path.close_path();

    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test(width = 256, height = 256, no_ref)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/11
fn out_of_bound_strip(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((258.0, 254.0));
    path.line_to((265.0, 254.0));
    let stroke = Stroke::new(1.0);

    ctx.set_paint(DARK_BLUE);
    ctx.set_stroke(stroke);
    // Just make sure we don't panic.
    ctx.stroke_path(&path);
}

#[vello_test]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/12
fn filling_unclosed_path_1(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((75.0, 25.0));
    path.line_to((25.0, 25.0));
    path.line_to((25.0, 75.0));

    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/12
fn filling_unclosed_path_2(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((50.0, 0.0));
    path.line_to((0.0, 0.0));
    path.line_to((0.0, 50.0));

    path.move_to((50.0, 50.0));
    path.line_to((100.0, 50.0));
    path.line_to((100.0, 100.0));
    path.line_to((50.0, 100.0));
    path.close_path();

    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test(width = 15, height = 8)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/28
fn triangle_exceeding_viewport_1(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((5.0, 0.0));
    path.line_to((12.0, 7.99));
    path.line_to((-4.0, 7.99));
    path.close_path();

    ctx.set_fill_rule(Fill::EvenOdd);
    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test(width = 15, height = 8)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/28
fn triangle_exceeding_viewport_2(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((4.0, 0.0));
    path.line_to((11.0, 7.99));
    path.line_to((-5.0, 7.99));
    path.close_path();

    ctx.set_fill_rule(Fill::EvenOdd);
    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test(width = 256, height = 4, no_ref)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/30
fn shape_at_wide_tile_boundary(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((248.0, 0.0));
    path.line_to((257.0, 0.0));
    path.line_to((257.0, 2.0));
    path.line_to((248.0, 2.0));
    path.close_path();

    ctx.set_fill_rule(Fill::EvenOdd);
    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test(width = 50, height = 50)]
fn eo_filling_missing_anti_aliasing(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((0.0, 0.0));
    path.line_to((50.0, 50.0));
    path.line_to((0.0, 50.0));
    path.line_to((50.0, 0.0));
    path.close_path();

    ctx.set_fill_rule(Fill::EvenOdd);
    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test(width = 600, height = 600, transparent)]
// https://github.com/linebender/vello/issues/906
fn fill_command_respects_clip_bounds(ctx: &mut impl Renderer) {
    ctx.push_clip_layer(&Rect::new(400.0, 400.0, 500.0, 500.0).to_path(0.1));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 600.0, 600.0));
    ctx.pop_layer();
}

#[vello_test(no_ref)]
fn out_of_viewport_clip(ctx: &mut impl Renderer) {
    ctx.push_clip_layer(&Rect::new(-100.0, -100.0, 0.0, 0.0).to_path(0.1));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.pop_layer();
}

#[vello_test(no_ref, width = 300, height = 4)]
// https://github.com/linebender/vello/issues/1032
fn nested_clip_path_panic(ctx: &mut impl Renderer) {
    let path1 = Rect::new(256.0, 0.0, 257.0, 2.0).to_path(0.1);
    ctx.push_clip_layer(&path1);
    let path2 = Rect::new(181.0, -200.0, 760.0, 618.0).to_path(0.1);
    ctx.push_clip_layer(&path2);
    ctx.pop_layer();
    ctx.pop_layer();
}

#[vello_test(width = 512, height = 4)]
// https://github.com/linebender/vello/issues/1034
fn nested_clip_path_panic_2(ctx: &mut impl Renderer) {
    let path1 = Rect::new(256.0, 0.0, 280.0, 2.0).to_path(0.1);
    ctx.push_clip_layer(&path1);
    let path2 = Rect::new(0.0, 0.0, 511.0, 4.0).to_path(0.1);
    ctx.push_clip_layer(&path2);
    ctx.set_paint(RED);
    ctx.fill_path(&Rect::new(0.0, 0.0, 511.0, 4.0).to_path(0.1));
    ctx.pop_layer();
    ctx.pop_layer();
}

#[vello_test(no_ref, width = 10, height = 16)]
// https://github.com/linebender/vello/issues/1072
fn intersected_clip_bbox_with_x0_gt_x1(ctx: &mut impl Renderer) {
    ctx.push_clip_layer(&Rect::new(0., 0., 4., 4.).to_path(0.1));
    ctx.push_clip_layer(&Rect::new(0., 8., 260., 16.).to_path(0.1));
    ctx.pop_layer();
    ctx.pop_layer();
}

// https://github.com/web-platform-tests/wpt/blob/cfd9285284893e6d63d7770deae0789d7f7457d4/html/canvas/element/fill-and-stroke-styles/2d.gradient.radial.inside3.html
// See <https://github.com/linebender/vello/issues/1124>.
#[vello_test(width = 100, height = 50)]
fn gradient_radial_inside(ctx: &mut impl Renderer) {
    ctx.set_paint(
        Gradient::new_two_point_radial((50., 25.), 200., (50., 25.), 100.).with_stops([
            ColorStop {
                offset: 0.,
                color: Color::from_rgb8(255, 0, 0).into(),
            },
            ColorStop {
                offset: 0.993,
                color: Color::from_rgb8(255, 0, 0).into(),
            },
            ColorStop {
                offset: 1.,
                color: Color::from_rgb8(0, 255, 0).into(),
            },
        ]),
    );
    ctx.fill_rect(&Rect::new(0., 0., 100., 50.));
}

// https://github.com/web-platform-tests/wpt/blob/cfd9285284893e6d63d7770deae0789d7f7457d4/html/canvas/element/fill-and-stroke-styles/2d.gradient.radial.outside3.html
// See <https://github.com/linebender/vello/issues/1124>.
#[vello_test(width = 100, height = 50)]
fn gradient_radial_outside(ctx: &mut impl Renderer) {
    ctx.set_paint(
        Gradient::new_two_point_radial((200., 25.), 20., (200., 25.), 10.).with_stops([
            ColorStop {
                offset: 0.,
                color: Color::from_rgb8(0, 255, 0).into(),
            },
            ColorStop {
                offset: 0.001,
                color: Color::from_rgb8(255, 0, 0).into(),
            },
            ColorStop {
                offset: 1.,
                color: Color::from_rgb8(255, 0, 0).into(),
            },
        ]),
    );
    ctx.fill_rect(&Rect::new(0., 0., 100., 50.));
}

#[vello_test(no_ref)]
/// <https://github.com/linebender/vello/issues/1113>
fn do_not_panic_on_multiple_flushes(ctx: &mut impl Renderer) {
    ctx.fill_rect(&Rect::new(0.0, 0.0, 4.0, 4.0));
    ctx.flush();
    ctx.fill_rect(&Rect::new(0.0, 0.0, 4.0, 4.0));
    ctx.flush();
    ctx.fill_rect(&Rect::new(0.0, 0.0, 4.0, 4.0));
}

/// <https://github.com/linebender/vello/issues/1119>
#[vello_test]
fn clip_clear(ctx: &mut impl Renderer) {
    // initial coloring
    ctx.set_paint(LIME);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.push_layer(
        Some(&Rect::new(0., 0., 50., 50.).to_path(0.1)),
        Some(Compose::Clear.into()),
        None,
        None,
        None,
    );
    ctx.pop_layer();
}

/// <https://github.com/web-platform-tests/wpt/blob/18c64a74b1/html/canvas/element/fill-and-stroke-styles/2d.gradient.interpolate.coloralpha.html>
/// See <https://github.com/linebender/vello/issues/1056>.
#[vello_test(width = 100, height = 50)]
fn gradient_color_alpha(ctx: &mut impl Renderer) {
    let viewport = Rect::new(0., 0., 100., 50.);
    ctx.set_paint(Gradient::new_linear((0., 0.), (100., 0.)).with_stops([
        ColorStop {
            offset: 0.,
            color: Color::from_rgba8(255, 255, 0, 0).into(),
        },
        ColorStop {
            offset: 1.,
            color: Color::from_rgba8(0, 0, 255, 255).into(),
        },
    ]));
    ctx.fill_rect(&viewport);
}

/// <https://github.com/web-platform-tests/wpt/blob/18c64a74b1/html/canvas/element/fill-and-stroke-styles/2d.gradient.interpolate.coloralpha.html>
/// See <https://github.com/linebender/vello/issues/1056>.
#[vello_test(width = 100, height = 50)]
fn gradient_color_alpha_unmul(ctx: &mut impl Renderer) {
    let viewport = Rect::new(0., 0., 100., 50.);
    ctx.push_blend_layer(Compose::Clear.into());
    ctx.pop_layer();
    ctx.set_paint(
        Gradient::new_linear((0., 0.), (100., 0.))
            .with_stops([
                ColorStop {
                    offset: 0.,
                    color: Color::from_rgba8(255, 255, 0, 0).into(),
                },
                ColorStop {
                    offset: 1.,
                    color: Color::from_rgba8(0, 0, 255, 255).into(),
                },
            ])
            .with_interpolation_alpha_space(InterpolationAlphaSpace::Unpremultiplied),
    );
    ctx.fill_rect(&viewport);
}

#[test]
fn multi_threading_oob_access() {
    let settings = RenderSettings {
        level: Level::try_detect().unwrap_or(Level::fallback()),
        num_threads: 4,
        render_mode: RenderMode::OptimizeQuality,
    };
    let mut ctx = RenderContext::new_with(100, 100, settings);
    let mut pixmap = Pixmap::new(100, 100);

    ctx.fill_path(&Rect::new(0.0, 0.0, 50.0, 50.0).to_path(0.1));
    ctx.flush();
    ctx.render_to_pixmap(&mut pixmap);
    ctx.fill_path(&Rect::new(50.0, 50.0, 100.0, 100.0).to_path(0.1));
    ctx.flush();
    ctx.render_to_pixmap(&mut pixmap);
}

/// See <https://github.com/linebender/vello/issues/1181>.
#[vello_test(width = 556, height = 8)]
fn tile_clamped_off_by_one(ctx: &mut impl Renderer) {
    let rect = Rect::new(0.0, 0.0, 556.0, 8.0);

    ctx.set_paint(BLACK);
    ctx.push_layer(Some(&rect.to_path(0.1)), None, None, None, None);
    ctx.fill_path(&rect.to_path(0.1));
    ctx.pop_layer();
}

/// See <https://github.com/linebender/vello/issues/1186>.
#[vello_test(width = 595, height = 20)]
fn clip_wrong_command(ctx: &mut impl Renderer) {
    ctx.set_paint(BLACK);
    ctx.set_transform(Affine::translate((0.0, -700.0)));
    ctx.push_clip_layer(&BezPath::from_svg("M551.704,721.115 C465.024,716.424 375.466,706.552 289.699,688.737 C290.316,688.60205 290.935,688.466 291.55,688.33 C377.059,705.978 466.259,715.75 552.629,720.39 C552.32,720.632 552.013,720.87305 551.704,721.115").unwrap());
    ctx.push_clip_layer(&BezPath::from_svg("M-133.795,680.40704 C390.292,801.45905 763.166,503.67102 666.575,258.86005 C1031.16,797.18604 -452.803,1197.37 -133.795,680.40704").unwrap());
    ctx.fill_path(&Rect::new(0.0, 0.0, 595.0, 808.0).to_path(0.1));
    ctx.pop_layer();
    ctx.pop_layer();
    ctx.flush();
}

/// See <https://github.com/linebender/vello/issues/1219>
#[vello_test]
fn basic_alpha_compositing(ctx: &mut impl Renderer) {
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 70.0, 70.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.9));
    ctx.fill_rect(&Rect::new(30.0, 30.0, 90.0, 90.0));
}

#[vello_test(no_ref)]
fn large_dimensions(ctx: &mut impl Renderer) {
    ctx.fill_rect(&Rect::new(0.0, 0.0, u16::MAX as f64 + 10.0, 8.0));
}

#[vello_test(width = 4, height = 4)]
fn issue_1433(ctx: &mut impl Renderer) {
    let r = PremulRgba8::from_u8_array([255, 0, 0, 255]);
    let b = PremulRgba8::from_u8_array([0, 0, 0, 0]);

    // Three red rows, one transparent row.
    #[rustfmt::skip]
    let image = vec![
        r, r, r, r,
        r, r, r, r,
        r, r, r, r,
        b, b, b, b
    ];

    let pixmap = Pixmap::from_parts(image, 4, 4);
    let source = ctx.get_image_source(Arc::new(pixmap));
    let image = Image {
        image: source,
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality: ImageQuality::Low,
            alpha: 1.0,
        },
    };

    ctx.set_paint(image);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 4.0, 4.0));
}
