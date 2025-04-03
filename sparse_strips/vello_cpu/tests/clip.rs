// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for clipping.

use crate::util::{check_ref, circular_star, crossed_line_star, get_ctx};
use std::f64::consts::PI;
use vello_common::color::palette::css::{DARK_BLUE, DARK_GREEN, REBECCA_PURPLE};
use vello_common::kurbo::{Affine, BezPath, Circle, Point, Rect, Shape, Stroke};
use vello_common::peniko::Fill;
use vello_cpu::RenderContext;

#[test]
fn clip_triangle_with_star() {
    let mut ctx: RenderContext = get_ctx(100, 100, true);

    let mut triangle_path = BezPath::new();
    triangle_path.move_to((10.0, 10.0));
    triangle_path.line_to((90.0, 20.0));
    triangle_path.line_to((20.0, 90.0));
    triangle_path.close_path();

    let stroke = Stroke::new(1.0);
    ctx.set_paint(DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(&triangle_path);

    let star_path = circular_star(Point::new(50., 50.), 13, 25., 45.);

    ctx.clip(&star_path);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_path(&triangle_path);
    ctx.finish();

    check_ref(&ctx, "clip_triangle_with_star");
}

#[test]
fn clip_rectangle_with_star_nonzero() {
    let mut ctx = get_ctx(100, 100, true);
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    // Create a self-intersecting star shape that will show the difference between fill rules
    let star_path = crossed_line_star();

    // Set the fill rule to NonZero before applying the clip
    ctx.set_fill_rule(Fill::NonZero);
    // Apply the star as a clip
    ctx.clip(&star_path);
    // Draw a rectangle that should be clipped by the star
    // The NonZero fill rule will treat self-intersecting regions as filled
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();
    check_ref(&ctx, "clip_rectangle_with_star_nonzero");
}

#[test]
fn clip_rectangle_with_star_evenodd() {
    let mut ctx = get_ctx(100, 100, true);
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    // Create a self-intersecting star shape that will show the difference between fill rules
    let star_path = crossed_line_star();

    // Set the fill rule to EvenOdd before applying the clip
    ctx.set_fill_rule(Fill::EvenOdd);
    // Apply the star as a clip
    ctx.clip(&star_path);
    // Draw a rectangle that should be clipped by the star
    // The EvenOdd rule should create a "hole" in the middle where the paths overlap
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();

    check_ref(&ctx, "clip_rectangle_with_star_evenodd");
}

#[test]
fn clip_rectangle_and_circle() {
    let mut ctx = get_ctx(100, 100, true);

    // Create first clipping region - a rectangle on the left side
    let clip_rect = Rect::new(10.0, 30.0, 50.0, 70.0);

    // Create second clipping region - a circle on the right side
    let circle_center = Point::new(65.0, 50.0);
    let circle_radius = 30.0;
    let clip_circle = Circle::new(circle_center, circle_radius).to_path(0.1);

    // Draw outlines of our clipping regions to visualize them
    let stroke = Stroke::new(1.0);
    ctx.set_paint(DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&clip_rect);
    ctx.stroke_path(&clip_circle);

    // Apply both clips
    ctx.clip(&clip_rect.to_path(0.1));
    ctx.clip(&clip_circle);

    // Then a filled rectangle that covers most of the canvas
    let large_rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&large_rect);
    ctx.finish();
    check_ref(&ctx, "clip_rectangle_and_circle");
}

#[test]
fn clip_with_translation() {
    let mut ctx = get_ctx(100, 100, true);

    // Apply a translation transform
    ctx.set_transform(Affine::translate((30.0, 30.0)));

    // Create and apply a clipping rectangle
    let clip_rect = Rect::new(0.0, 0.0, 40.0, 40.0);
    draw_clipping_outline(&mut ctx, &clip_rect.to_path(0.1));
    ctx.clip(&clip_rect.to_path(0.1));

    // Draw a rectangle that should be clipped
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();
    check_ref(&ctx, "clip_with_translation");
}

#[test]
fn clip_with_scale() {
    let mut ctx = get_ctx(100, 100, true);

    ctx.set_transform(Affine::scale(2.0));

    // Create and apply a clipping rectangle
    let clip_rect = Rect::new(10.0, 10.0, 40.0, 40.0);
    draw_clipping_outline(&mut ctx, &clip_rect.to_path(0.1));
    ctx.clip(&clip_rect.to_path(0.1));

    // Draw a rectangle that should be clipped
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();

    check_ref(&ctx, "clip_with_scale");
}

#[test]
fn clip_with_rotate() {
    let mut ctx = get_ctx(100, 100, true);

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(50.0, 50.0),
    ));

    // Create and apply a clipping rectangle
    let clip_rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    draw_clipping_outline(&mut ctx, &clip_rect.to_path(0.1));
    ctx.clip(&clip_rect.to_path(0.1));

    // Draw a rectangle that should be clipped
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();

    check_ref(&ctx, "clip_with_rotate");
}

#[test]
fn clip_transformed_rect() {
    let mut ctx = get_ctx(100, 100, true);

    let clip_rect = Rect::new(20.0, 20.0, 80.0, 80.0);

    draw_clipping_outline(&mut ctx, &clip_rect.to_path(0.1));

    ctx.clip(&clip_rect.to_path(0.1));

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(50.0, 50.0),
    ));

    // Draw a smaller rectangle that should be clipped
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();

    check_ref(&ctx, "clip_transformed_rect");
}

#[test]
fn clip_with_multiple_transforms() {
    let mut ctx = get_ctx(100, 100, true);

    // Apply initial transform
    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(50.0, 50.0),
    ));

    // Create and apply first clip
    let clip_rect1 = Rect::new(20.0, 20.0, 80.0, 80.0);
    draw_clipping_outline(&mut ctx, &clip_rect1.to_path(0.1));
    ctx.clip(&clip_rect1.to_path(0.1));

    // Apply another transform
    ctx.set_transform(Affine::scale(1.5));

    // Create and apply second clip
    let clip_rect2 = Rect::new(30.0, 30.0, 70.0, 70.0);
    draw_clipping_outline(&mut ctx, &clip_rect2.to_path(0.1));
    ctx.clip(&clip_rect2.to_path(0.1));

    // Draw a rectangle that should be clipped by both regions
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();

    check_ref(&ctx, "clip_with_multiple_transforms");
}

#[test]
fn clip_with_save_restore() {
    let mut ctx = get_ctx(100, 100, true);

    // Create first clipping region - a rectangle on the left side
    let clip_rect1 = Rect::new(10.0, 30.0, 50.0, 70.0);
    draw_clipping_outline(&mut ctx, &clip_rect1.to_path(0.1));
    ctx.clip(&clip_rect1.to_path(0.1));

    // Save the state after first clip
    ctx.save();

    // Add second clipping region - a circle on the right side
    let circle_center = Point::new(65.0, 50.0);
    let circle_radius = 30.0;
    let clip_circle = Circle::new(circle_center, circle_radius).to_path(0.1);
    draw_clipping_outline(&mut ctx, &clip_circle);
    ctx.clip(&clip_circle);

    // Draw a rectangle that should be clipped by both regions
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);

    // Restore to state before second clip
    ctx.restore();

    // Draw another rectangle that should only be clipped by the first region
    let rect2 = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(DARK_GREEN.with_alpha(0.5).into());
    ctx.fill_rect(&rect2);
    ctx.finish();
    check_ref(&ctx, "clip_with_save_restore");
}

fn draw_clipping_outline(ctx: &mut RenderContext, path: &BezPath) {
    let stroke = Stroke::new(1.0);
    ctx.set_paint(DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(path);
}
