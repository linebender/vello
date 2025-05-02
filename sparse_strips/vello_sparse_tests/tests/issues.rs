// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for GitHub issues.

use crate::renderer::Renderer;
use vello_common::color::palette::css::{DARK_BLUE, LIME, REBECCA_PURPLE};
use vello_common::kurbo::{BezPath, Rect, Shape, Stroke};
use vello_common::peniko::Fill;
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
