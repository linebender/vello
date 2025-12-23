// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for basic functionality.

use crate::renderer::Renderer;
use crate::util::{circular_star, crossed_line_star, miter_stroke_2};
use std::f64::consts::PI;
use vello_common::coarse::Cmd;
use vello_common::color::palette::css::{
    BEIGE, BLUE, DARK_BLUE, GREEN, LIME, MAROON, REBECCA_PURPLE, RED, TRANSPARENT,
};
use vello_common::kurbo::{Affine, BezPath, Circle, Join, Point, Rect, Shape, Stroke};
use vello_common::peniko::Fill;
use vello_dev_macros::vello_test;

#[vello_test(width = 8, height = 8)]
fn full_cover_1(ctx: &mut impl Renderer) {
    ctx.set_paint(BEIGE);
    ctx.fill_path(&Rect::new(0.0, 0.0, 8.0, 8.0).to_path(0.1));
}

#[vello_test(transparent)]
fn transparent_paint(ctx: &mut impl Renderer) {
    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((95.0, 50.0));
        path.line_to((5.0, 95.0));
        path.close_path();

        path
    };

    ctx.set_paint(TRANSPARENT);
    ctx.fill_path(&path);
    ctx.stroke_path(&path);
}

#[vello_test]
fn filled_triangle(ctx: &mut impl Renderer) {
    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((95.0, 50.0));
        path.line_to((5.0, 95.0));
        path.close_path();

        path
    };

    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test]
fn stroked_triangle(ctx: &mut impl Renderer) {
    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((95.0, 50.0));
        path.line_to((5.0, 95.0));
        path.close_path();

        path
    };

    ctx.set_stroke(Stroke::new(3.0));
    ctx.set_paint(LIME);
    ctx.stroke_path(&path);
}

#[vello_test]
fn filled_circle(ctx: &mut impl Renderer) {
    let circle = Circle::new((50.0, 50.0), 45.0);
    ctx.set_paint(LIME);
    ctx.fill_path(&circle.to_path(0.1));
}

#[vello_test]
fn filled_overflowing_circle(ctx: &mut impl Renderer) {
    let circle = Circle::new((50.0, 50.0), 50.0 + 1.0);

    ctx.set_paint(LIME);
    ctx.fill_path(&circle.to_path(0.1));
}

#[vello_test]
fn filled_fully_overflowing_circle(ctx: &mut impl Renderer) {
    let circle = Circle::new((50.0, 50.0), 80.0);

    ctx.set_paint(LIME);
    ctx.fill_path(&circle.to_path(0.1));
}

#[vello_test]
fn filled_circle_with_opacity(ctx: &mut impl Renderer) {
    let circle = Circle::new((50.0, 50.0), 45.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(&circle.to_path(0.1));
}

#[vello_test(cpu_u8_tolerance = 1)]
fn filled_overlapping_circles(ctx: &mut impl Renderer) {
    for e in [(35.0, 35.0, RED), (65.0, 35.0, GREEN), (50.0, 65.0, BLUE)] {
        let circle = Circle::new((e.0, e.1), 30.0);
        ctx.set_paint(e.2.with_alpha(0.5));
        ctx.fill_path(&circle.to_path(0.1));
    }
}

#[vello_test]
fn stroked_circle(ctx: &mut impl Renderer) {
    let circle = Circle::new((50.0, 50.0), 45.0);
    let stroke = Stroke::new(3.0);

    ctx.set_paint(LIME);
    ctx.set_stroke(stroke);
    ctx.stroke_path(&circle.to_path(0.1));
}

/// Requires winding of the first row of tiles to be calculated correctly for vertical lines.
#[vello_test(width = 10, height = 10)]
fn rectangle_above_viewport(ctx: &mut impl Renderer) {
    let rect = Rect::new(2.0, -5.0, 8.0, 8.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

/// Requires winding of the first row of tiles to be calculated correctly for sloped lines.
#[vello_test(width = 10, height = 10)]
fn triangle_above_and_wider_than_viewport(ctx: &mut impl Renderer) {
    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, -5.0));
        path.line_to((14., 6.));
        path.line_to((-8., 6.));
        path.close_path();

        path
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(&path);
}

/// Requires winding and pixel coverage to be calculated correctly for tiles preceding the
/// viewport in scan direction.
#[vello_test(width = 10, height = 10)]
fn rectangle_left_of_viewport(ctx: &mut impl Renderer) {
    let rect = Rect::new(-4.0, 3.0, 1.0, 8.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

#[vello_test]
fn filling_nonzero_rule(ctx: &mut impl Renderer) {
    let star = crossed_line_star();

    ctx.set_paint(MAROON);
    ctx.fill_path(&star);
}

#[vello_test]
fn filling_evenodd_rule(ctx: &mut impl Renderer) {
    let star = crossed_line_star();

    ctx.set_paint(MAROON);
    ctx.set_fill_rule(Fill::EvenOdd);
    ctx.fill_path(&star);
}

#[vello_test(width = 30, height = 20)]
fn filled_aligned_rect(ctx: &mut impl Renderer) {
    let rect = Rect::new(1.0, 1.0, 29.0, 19.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn stroked_unaligned_rect(ctx: &mut impl Renderer) {
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke {
        width: 1.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn stroked_unaligned_rect_as_path(ctx: &mut impl Renderer) {
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0).to_path(0.1);
    let stroke = Stroke {
        width: 1.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(stroke);
    ctx.stroke_path(&rect);
}

#[vello_test(width = 30, height = 30)]
fn stroked_aligned_rect(ctx: &mut impl Renderer) {
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = miter_stroke_2();

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn overflowing_stroked_rect(ctx: &mut impl Renderer) {
    let rect = Rect::new(12.5, 12.5, 17.5, 17.5);
    let stroke = Stroke {
        width: 5.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn round_stroked_rect(ctx: &mut impl Renderer) {
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke::new(3.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn bevel_stroked_rect(ctx: &mut impl Renderer) {
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke {
        width: 3.0,
        join: Join::Bevel,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);
}

#[vello_test(width = 30, height = 20)]
fn filled_unaligned_rect(ctx: &mut impl Renderer) {
    let rect = Rect::new(1.5, 1.5, 28.5, 18.5);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn filled_transformed_rect_1(ctx: &mut impl Renderer) {
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);

    ctx.set_transform(Affine::translate((10.0, 10.0)));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn filled_transformed_rect_2(ctx: &mut impl Renderer) {
    let rect = Rect::new(5.0, 5.0, 10.0, 10.0);

    ctx.set_transform(Affine::scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn filled_transformed_rect_3(ctx: &mut impl Renderer) {
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);

    ctx.set_transform(Affine::new([2.0, 0.0, 0.0, 2.0, 5.0, 5.0]));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn filled_transformed_rect_4(ctx: &mut impl Renderer) {
    let rect = Rect::new(10.0, 10.0, 20.0, 20.0);

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(15.0, 15.0),
    ));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn stroked_transformed_rect_1(ctx: &mut impl Renderer) {
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::translate((10.0, 10.0)));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn stroked_transformed_rect_2(ctx: &mut impl Renderer) {
    let rect = Rect::new(5.0, 5.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn stroked_transformed_rect_3(ctx: &mut impl Renderer) {
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::new([2.0, 0.0, 0.0, 2.0, 5.0, 5.0]));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);
}

#[vello_test(width = 30, height = 30)]
fn stroked_transformed_rect_4(ctx: &mut impl Renderer) {
    let rect = Rect::new(10.0, 10.0, 20.0, 20.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(15.0, 15.0),
    ));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);
}

#[vello_test(width = 30, height = 20)]
fn strip_inscribed_rect(ctx: &mut impl Renderer) {
    let rect = Rect::new(1.5, 9.5, 28.5, 11.5);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

#[vello_test(width = 5, height = 8)]
fn filled_vertical_hairline_rect(ctx: &mut impl Renderer) {
    let rect = Rect::new(2.25, 0.0, 2.75, 8.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

#[vello_test(width = 10, height = 10)]
fn filled_vertical_hairline_rect_2(ctx: &mut impl Renderer) {
    let rect = Rect::new(4.5, 0.5, 5.5, 9.5);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

#[vello_test]
fn oversized_star(ctx: &mut impl Renderer) {
    // Create a star path that extends beyond the render context boundaries
    // Center it in the middle of the viewport
    let star_path = circular_star(Point::new(50., 50.), 10, 30., 90.);

    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&star_path);

    let stroke = Stroke::new(2.0);
    ctx.set_paint(DARK_BLUE);
    ctx.set_stroke(stroke);
    ctx.stroke_path(&star_path);
}

#[vello_test(width = 100, height = 100)]
fn no_anti_aliasing(ctx: &mut impl Renderer) {
    let rect = Rect::new(30.0, 30.0, 70.0, 70.0);
    ctx.set_aliasing_threshold(Some(128));

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(50.0, 50.0),
    ));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_rect(&rect);
}

#[vello_test(width = 100, height = 100)]
fn no_anti_aliasing_clip_path(ctx: &mut impl Renderer) {
    ctx.set_aliasing_threshold(Some(128));
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    let star_path = crossed_line_star();

    ctx.set_fill_rule(Fill::NonZero);
    ctx.push_clip_layer(&star_path);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

#[vello_test(diff_pixels = 1)]
fn stroke_scaled(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((0.0, 0.0));
    path.curve_to((0.25, 1.0), (0.75, 1.0), (1.0, 0.0));

    // This path should be more or less completely covered.
    let mut stroke = Stroke::new(10.0);
    ctx.set_transform(Affine::IDENTITY);
    ctx.set_stroke(stroke);
    ctx.set_paint(RED);
    ctx.stroke_path(&(Affine::scale(100.0) * path.clone()));

    stroke = Stroke::new(0.1);
    ctx.set_transform(Affine::scale(100.0));
    ctx.set_stroke(stroke);
    ctx.set_paint(LIME);
    ctx.stroke_path(&path);
}

// Just so we can more closely observe changes in their size.
// We have this test here instead of in `vello_common` because
// the vello_common tests seemingly are not run for 32-bit in CI.
#[vello_test(no_ref)]
fn test_cmd_size(_: &mut impl Renderer) {
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        size_of::<Cmd>(),
        16,
        "size of a command didn't match the expected value"
    );
    #[cfg(target_pointer_width = "32")]
    assert_eq!(
        size_of::<Cmd>(),
        16,
        "size of a command didn't match the expected value"
    );
}
