// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for basic functionality.
//!
//! Ported from [`crate::basic`] to use Vello API.
//!
//! These are all `existing_ref` as the versions in `crate::basic` are the references.

use std::f64::consts::PI;

use vello_api::{
    PaintScene,
    peniko::{Color, Fill},
};
use vello_cpu::{
    color::palette::css,
    kurbo::{Affine, BezPath, Circle, Join, Point, Rect, Stroke},
};
use vello_dev_macros::vello_api_test;

use crate::util::{circular_star, crossed_line_star, miter_stroke_2};

#[vello_api_test(width = 8, height = 8, existing_ref)]
fn full_cover_1(ctx: &mut impl PaintScene) {
    ctx.set_solid_brush(css::BEIGE);
    ctx.fill_path(
        Affine::IDENTITY,
        Fill::EvenOdd,
        Rect::new(0.0, 0.0, 8.0, 8.0),
    );
}

#[vello_api_test(transparent, existing_ref)]
fn transparent_paint(ctx: &mut impl PaintScene) {
    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((95.0, 50.0));
        path.line_to((5.0, 95.0));
        path.close_path();

        path
    };

    ctx.set_solid_brush(Color::TRANSPARENT);
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, &path);
    ctx.stroke_path(Affine::IDENTITY, &Stroke::default(), &path);
}

#[vello_api_test(existing_ref)]
fn filled_triangle(ctx: &mut impl PaintScene) {
    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((95.0, 50.0));
        path.line_to((5.0, 95.0));
        path.close_path();

        path
    };

    ctx.set_solid_brush(css::LIME);
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, &path);
}

#[vello_api_test(existing_ref)]
fn stroked_triangle(ctx: &mut impl PaintScene) {
    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((95.0, 50.0));
        path.line_to((5.0, 95.0));
        path.close_path();

        path
    };

    ctx.set_solid_brush(css::LIME);
    ctx.stroke_path(Affine::IDENTITY, &Stroke::new(3.0), &path);
}

#[vello_api_test(existing_ref)]
fn filled_circle(ctx: &mut impl PaintScene) {
    let circle = Circle::new((50.0, 50.0), 45.0);
    ctx.set_solid_brush(css::LIME);
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, circle);
}

#[vello_api_test(existing_ref)]
fn filled_overflowing_circle(ctx: &mut impl PaintScene) {
    let circle = Circle::new((50.0, 50.0), 50.0 + 1.0);

    ctx.set_solid_brush(css::LIME);
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, circle);
}

#[vello_api_test(existing_ref)]
fn filled_fully_overflowing_circle(ctx: &mut impl PaintScene) {
    let circle = Circle::new((50.0, 50.0), 80.0);

    ctx.set_solid_brush(css::LIME);
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, circle);
}

#[vello_api_test(existing_ref)]
fn filled_circle_with_opacity(ctx: &mut impl PaintScene) {
    let circle = Circle::new((50.0, 50.0), 45.0);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, circle);
}

#[vello_api_test(cpu_u8_tolerance = 1, existing_ref)]
fn filled_overlapping_circles(ctx: &mut impl PaintScene) {
    for (x, y, color) in [
        (35.0, 35.0, css::RED),
        (65.0, 35.0, css::GREEN),
        (50.0, 65.0, css::BLUE),
    ] {
        let circle = Circle::new((x, y), 30.0);
        ctx.set_solid_brush(color.with_alpha(0.5));
        ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, circle);
    }
}

#[vello_api_test(existing_ref)]
fn stroked_circle(ctx: &mut impl PaintScene) {
    let circle = Circle::new((50.0, 50.0), 45.0);
    let stroke = Stroke::new(3.0);

    ctx.set_solid_brush(css::LIME);
    ctx.stroke_path(Affine::IDENTITY, &stroke, circle);
}

/// Requires winding of the first row of tiles to be calculated correctly for vertical lines.
#[vello_api_test(width = 10, height = 10, existing_ref)]
fn rectangle_above_viewport(ctx: &mut impl PaintScene) {
    let rect = Rect::new(2.0, -5.0, 8.0, 8.0);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, rect);
}

/// Requires winding of the first row of tiles to be calculated correctly for sloped lines.
#[vello_api_test(width = 10, height = 10, existing_ref)]
fn triangle_above_and_wider_than_viewport(ctx: &mut impl PaintScene) {
    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, -5.0));
        path.line_to((14., 6.));
        path.line_to((-8., 6.));
        path.close_path();

        path
    };

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, &path);
}

/// Requires winding and pixel coverage to be calculated correctly for tiles preceding the
/// viewport in scan direction.
#[vello_api_test(width = 10, height = 10, existing_ref)]
fn rectangle_left_of_viewport(ctx: &mut impl PaintScene) {
    let rect = Rect::new(-4.0, 3.0, 1.0, 8.0);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, rect);
}

#[vello_api_test(existing_ref)]
fn filling_nonzero_rule(ctx: &mut impl PaintScene) {
    let star = crossed_line_star();

    ctx.set_solid_brush(css::MAROON);
    ctx.fill_path(Affine::IDENTITY, Fill::NonZero, &star);
}

#[vello_api_test(existing_ref)]
fn filling_evenodd_rule(ctx: &mut impl PaintScene) {
    let star = crossed_line_star();

    ctx.set_solid_brush(css::MAROON);
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, &star);
}

#[vello_api_test(width = 30, height = 20, existing_ref)]
fn filled_aligned_rect(ctx: &mut impl PaintScene) {
    let rect = Rect::new(1.0, 1.0, 29.0, 19.0);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn stroked_unaligned_rect(ctx: &mut impl PaintScene) {
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke {
        width: 1.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.stroke_path(Affine::IDENTITY, &stroke, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn stroked_unaligned_rect_as_path(ctx: &mut impl PaintScene) {
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke {
        width: 1.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.stroke_path(Affine::IDENTITY, &stroke, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn stroked_aligned_rect(ctx: &mut impl PaintScene) {
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = miter_stroke_2();

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.stroke_path(Affine::IDENTITY, &stroke, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn overflowing_stroked_rect(ctx: &mut impl PaintScene) {
    let rect = Rect::new(12.5, 12.5, 17.5, 17.5);
    let stroke = Stroke {
        width: 5.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.stroke_path(Affine::IDENTITY, &stroke, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn round_stroked_rect(ctx: &mut impl PaintScene) {
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke::new(3.0);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.stroke_path(Affine::IDENTITY, &stroke, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn bevel_stroked_rect(ctx: &mut impl PaintScene) {
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke {
        width: 3.0,
        join: Join::Bevel,
        ..Default::default()
    };

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.stroke_path(Affine::IDENTITY, &stroke, rect);
}

#[vello_api_test(width = 30, height = 20, existing_ref)]
fn filled_unaligned_rect(ctx: &mut impl PaintScene) {
    let rect = Rect::new(1.5, 1.5, 28.5, 18.5);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn filled_transformed_rect_1(ctx: &mut impl PaintScene) {
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(Affine::translate((10.0, 10.0)), Fill::EvenOdd, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn filled_transformed_rect_2(ctx: &mut impl PaintScene) {
    let rect = Rect::new(5.0, 5.0, 10.0, 10.0);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(Affine::scale(2.0), Fill::EvenOdd, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn filled_transformed_rect_3(ctx: &mut impl PaintScene) {
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(
        Affine::new([2.0, 0.0, 0.0, 2.0, 5.0, 5.0]),
        Fill::EvenOdd,
        rect,
    );
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn filled_transformed_rect_4(ctx: &mut impl PaintScene) {
    let rect = Rect::new(10.0, 10.0, 20.0, 20.0);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(
        Affine::rotate_about(45.0 * PI / 180.0, Point::new(15.0, 15.0)),
        Fill::EvenOdd,
        rect,
    );
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn stroked_transformed_rect_1(ctx: &mut impl PaintScene) {
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.stroke_path(Affine::translate((10.0, 10.0)), &stroke, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn stroked_transformed_rect_2(ctx: &mut impl PaintScene) {
    let rect = Rect::new(5.0, 5.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.stroke_path(Affine::scale(2.0), &stroke, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn stroked_transformed_rect_3(ctx: &mut impl PaintScene) {
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.stroke_path(Affine::new([2.0, 0.0, 0.0, 2.0, 5.0, 5.0]), &stroke, rect);
}

#[vello_api_test(width = 30, height = 30, existing_ref)]
fn stroked_transformed_rect_4(ctx: &mut impl PaintScene) {
    let rect = Rect::new(10.0, 10.0, 20.0, 20.0);
    let stroke = miter_stroke_2();

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.stroke_path(
        Affine::rotate_about(45.0 * PI / 180.0, Point::new(15.0, 15.0)),
        &stroke,
        rect,
    );
}

#[vello_api_test(width = 30, height = 20, existing_ref)]
fn strip_inscribed_rect(ctx: &mut impl PaintScene) {
    let rect = Rect::new(1.5, 9.5, 28.5, 11.5);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, rect);
}

#[vello_api_test(width = 5, height = 8, existing_ref)]
fn filled_vertical_hairline_rect(ctx: &mut impl PaintScene) {
    let rect = Rect::new(2.25, 0.0, 2.75, 8.0);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, rect);
}

#[vello_api_test(width = 10, height = 10, existing_ref)]
fn filled_vertical_hairline_rect_2(ctx: &mut impl PaintScene) {
    let rect = Rect::new(4.5, 0.5, 5.5, 9.5);

    ctx.set_solid_brush(css::REBECCA_PURPLE.with_alpha(0.5));
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, rect);
}

#[vello_api_test(existing_ref)]
fn oversized_star(ctx: &mut impl PaintScene) {
    // Create a star path that extends beyond the render context boundaries
    // Center it in the middle of the viewport
    let star_path = circular_star(Point::new(50., 50.), 10, 30., 90.);

    ctx.set_solid_brush(css::REBECCA_PURPLE);
    ctx.fill_path(Affine::IDENTITY, Fill::EvenOdd, &star_path);

    let stroke = Stroke::new(2.0);
    ctx.set_solid_brush(css::DARK_BLUE);
    ctx.stroke_path(Affine::IDENTITY, &stroke, &star_path);
}

// Vello API doesn't currently expose the anti-aliasing threshold
// fn no_anti_aliasing(ctx: &mut impl PaintScene) {

// fn no_anti_aliasing_clip_path(ctx: &mut impl PaintScene) {

#[vello_api_test(diff_pixels = 1, existing_ref)]
fn stroke_scaled(ctx: &mut impl PaintScene) {
    let mut path = BezPath::new();
    path.move_to((0.0, 0.0));
    path.curve_to((0.25, 1.0), (0.75, 1.0), (1.0, 0.0));

    // This path should be more or less completely covered.
    let mut stroke = Stroke::new(10.0);
    ctx.set_solid_brush(css::RED);
    ctx.stroke_path(
        Affine::IDENTITY,
        &stroke,
        &(Affine::scale(100.0) * path.clone()),
    );

    stroke = Stroke::new(0.1);

    ctx.set_solid_brush(css::LIME);
    ctx.stroke_path(Affine::scale(100.0), &stroke, &path);
}

// fn test_cmd_size(_) is not copied into this module
