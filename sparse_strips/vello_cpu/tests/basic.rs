// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for basic functionality.

use crate::util::{
    check_ref, circular_star, crossed_line_star, get_ctx, miter_stroke_2,
};
use std::f64::consts::PI;
use vello_common::color::palette::css::{
    BEIGE, BLUE, DARK_BLUE, GREEN, LIME, MAROON, REBECCA_PURPLE, RED,
};
use vello_common::kurbo::{Affine, BezPath, Circle, Join, Point, Rect, Shape, Stroke};
use vello_common::peniko::Fill;

#[test]
fn full_cover_1() {
    let mut ctx = get_ctx(8, 8, true);

    ctx.set_paint(BEIGE.into());
    ctx.fill_path(&Rect::new(0.0, 0.0, 8.0, 8.0).to_path(0.1));

    check_ref(&ctx, "full_cover_1");
}

#[test]
fn filled_triangle() {
    let mut ctx = get_ctx(100, 100, false);

    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((95.0, 50.0));
        path.line_to((5.0, 95.0));
        path.close_path();

        path
    };

    ctx.set_paint(LIME.into());
    ctx.fill_path(&path);

    check_ref(&ctx, "filled_triangle");
}

#[test]
fn stroked_triangle() {
    let mut ctx = get_ctx(100, 100, false);
    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((95.0, 50.0));
        path.line_to((5.0, 95.0));
        path.close_path();

        path
    };

    ctx.set_stroke(Stroke::new(3.0));
    ctx.set_paint(LIME.into());
    ctx.stroke_path(&path);

    check_ref(&ctx, "stroked_triangle");
}

#[test]
fn filled_circle() {
    let mut ctx = get_ctx(100, 100, false);
    let circle = Circle::new((50.0, 50.0), 45.0);
    ctx.set_paint(LIME.into());
    ctx.fill_path(&circle.to_path(0.1));

    check_ref(&ctx, "filled_circle");
}

#[test]
fn filled_overflowing_circle() {
    let mut ctx = get_ctx(100, 100, false);
    let circle = Circle::new((50.0, 50.0), 50.0 + 1.0);

    ctx.set_paint(LIME.into());
    ctx.fill_path(&circle.to_path(0.1));

    check_ref(&ctx, "filled_overflowing_circle");
}

#[test]
fn filled_fully_overflowing_circle() {
    let mut ctx = get_ctx(100, 100, false);
    let circle = Circle::new((50.0, 50.0), 80.0);

    ctx.set_paint(LIME.into());
    ctx.fill_path(&circle.to_path(0.1));

    check_ref(&ctx, "filled_fully_overflowing_circle");
}

#[test]
fn filled_circle_with_opacity() {
    let mut ctx = get_ctx(100, 100, false);
    let circle = Circle::new((50.0, 50.0), 45.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_path(&circle.to_path(0.1));

    check_ref(&ctx, "filled_circle_with_opacity");
}

#[test]
fn filled_overlapping_circles() {
    let mut ctx = get_ctx(100, 100, false);

    for e in [(35.0, 35.0, RED), (65.0, 35.0, GREEN), (50.0, 65.0, BLUE)] {
        let circle = Circle::new((e.0, e.1), 30.0);
        ctx.set_paint(e.2.with_alpha(0.5).into());
        ctx.fill_path(&circle.to_path(0.1));
    }

    check_ref(&ctx, "filled_overlapping_circles");
}

#[test]
fn stroked_circle() {
    let mut ctx = get_ctx(100, 100, false);
    let circle = Circle::new((50.0, 50.0), 45.0);
    let stroke = Stroke::new(3.0);

    ctx.set_paint(LIME.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(&circle.to_path(0.1));

    check_ref(&ctx, "stroked_circle");
}

/// Requires winding of the first row of tiles to be calculcated correctly for vertical lines.
#[test]
fn rectangle_above_viewport() {
    let mut ctx = get_ctx(10, 10, false);
    let rect = Rect::new(2.0, -5.0, 8.0, 8.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "rectangle_above_viewport");
}

/// Requires winding of the first row of tiles to be calculcated correctly for sloped lines.
#[test]
fn triangle_above_and_wider_than_viewport() {
    let mut ctx = get_ctx(10, 10, false);

    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, -5.0));
        path.line_to((14., 6.));
        path.line_to((-8., 6.));
        path.close_path();

        path
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_path(&path);

    check_ref(&ctx, "triangle_above_and_wider_than_viewport");
}

/// Requires winding and pixel coverage to be calculcated correctly for tiles preceding the
/// viewport in scan direction.
#[test]
fn rectangle_left_of_viewport() {
    let mut ctx = get_ctx(10, 10, false);
    let rect = Rect::new(-4.0, 3.0, 1.0, 8.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "rectangle_left_of_viewport");
}

#[test]
fn filling_nonzero_rule() {
    let mut ctx = get_ctx(100, 100, false);
    let star = crossed_line_star();

    ctx.set_paint(MAROON.into());
    ctx.fill_path(&star);

    check_ref(&ctx, "filling_nonzero_rule");
}

#[test]
fn filling_evenodd_rule() {
    let mut ctx = get_ctx(100, 100, false);
    let star = crossed_line_star();

    ctx.set_paint(MAROON.into());
    ctx.set_fill_rule(Fill::EvenOdd);
    ctx.fill_path(&star);

    check_ref(&ctx, "filling_evenodd_rule");
}

#[test]
fn filled_aligned_rect() {
    let mut ctx = get_ctx(30, 20, false);
    let rect = Rect::new(1.0, 1.0, 29.0, 19.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_aligned_rect");
}

#[test]
fn stroked_unaligned_rect() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke {
        width: 1.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_unaligned_rect");
}

#[test]
fn stroked_unaligned_rect_as_path() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0).to_path(0.1);
    let stroke = Stroke {
        width: 1.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(&rect);

    check_ref(&ctx, "stroked_unaligned_rect_as_path");
}

#[test]
fn stroked_aligned_rect() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = miter_stroke_2();

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_aligned_rect");
}

#[test]
fn overflowing_stroked_rect() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(12.5, 12.5, 17.5, 17.5);
    let stroke = Stroke {
        width: 5.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "overflowing_stroked_rect");
}

#[test]
fn round_stroked_rect() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke::new(3.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "round_stroked_rect");
}

#[test]
fn bevel_stroked_rect() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke {
        width: 3.0,
        join: Join::Bevel,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "bevel_stroked_rect");
}

#[test]
fn filled_unaligned_rect() {
    let mut ctx = get_ctx(30, 20, false);
    let rect = Rect::new(1.5, 1.5, 28.5, 18.5);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_unaligned_rect");
}

#[test]
fn filled_transformed_rect_1() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);

    ctx.set_transform(Affine::translate((10.0, 10.0)));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_transformed_rect_1");
}

#[test]
fn filled_transformed_rect_2() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 10.0, 10.0);

    ctx.set_transform(Affine::scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_transformed_rect_2");
}

#[test]
fn filled_transformed_rect_3() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);

    ctx.set_transform(Affine::new([2.0, 0.0, 0.0, 2.0, 5.0, 5.0]));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_transformed_rect_3");
}

#[test]
fn filled_transformed_rect_4() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(10.0, 10.0, 20.0, 20.0);

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(15.0, 15.0),
    ));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_transformed_rect_4");
}

#[test]
fn stroked_transformed_rect_1() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::translate((10.0, 10.0)));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_transformed_rect_1");
}

#[test]
fn stroked_transformed_rect_2() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_transformed_rect_2");
}

#[test]
fn stroked_transformed_rect_3() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::new([2.0, 0.0, 0.0, 2.0, 5.0, 5.0]));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_transformed_rect_3");
}

#[test]
fn stroked_transformed_rect_4() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(10.0, 10.0, 20.0, 20.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(15.0, 15.0),
    ));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_transformed_rect_4");
}

#[test]
fn strip_inscribed_rect() {
    let mut ctx = get_ctx(30, 20, false);
    let rect = Rect::new(1.5, 9.5, 28.5, 11.5);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "strip_inscribed_rect");
}

#[test]
fn filled_vertical_hairline_rect() {
    let mut ctx = get_ctx(5, 8, false);
    let rect = Rect::new(2.25, 0.0, 2.75, 8.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_vertical_hairline_rect");
}

#[test]
fn filled_vertical_hairline_rect_2() {
    let mut ctx = get_ctx(10, 10, false);
    let rect = Rect::new(4.5, 0.5, 5.5, 9.5);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_vertical_hairline_rect_2");
}

#[test]
fn oversized_star() {
    let mut ctx = get_ctx(100, 100, true);

    // Create a star path that extends beyond the render context boundaries
    // Center it in the middle of the viewport
    let star_path = circular_star(Point::new(50., 50.), 10, 30., 90.);

    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_path(&star_path);

    let stroke = Stroke::new(2.0);
    ctx.set_paint(DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(&star_path);

    check_ref(&ctx, "oversized_star");
}
