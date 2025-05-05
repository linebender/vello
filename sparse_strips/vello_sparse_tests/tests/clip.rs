// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for clipping.

use crate::renderer::Renderer;
use crate::util::{circular_star, crossed_line_star};
use std::f64::consts::PI;
use vello_api::color::palette::css::{
    BLACK, BLUE, DARK_BLUE, DARK_GREEN, GREEN, REBECCA_PURPLE, RED,
};
use vello_api::peniko::Color;
use vello_common::coarse::WideTile;
use vello_common::kurbo::{Affine, BezPath, Circle, Point, Rect, Shape, Stroke};
use vello_common::peniko::Fill;
use vello_common::tile::Tile;
use vello_dev_macros::vello_test;

#[vello_test(height = 8)]
fn clip_single_wide_tile(ctx: &mut impl Renderer) {
    const WIDTH: f64 = 100.0;
    assert!(WIDTH <= WideTile::WIDTH as f64, "Width larger than a tile");
    const HEIGHT: f64 = Tile::HEIGHT as f64;
    const OFFSET: f64 = WIDTH / 3.0;

    let colors = [RED, GREEN, BLUE];

    for (i, color) in colors.iter().enumerate() {
        let clip_rect = Rect::new((i as f64) * OFFSET, 0.0, WIDTH, HEIGHT);
        ctx.push_clip_layer(&clip_rect.to_path(0.1));
        ctx.set_paint(*color);
        ctx.fill_rect(&Rect::new(0.0, 0.0, WIDTH, HEIGHT));
    }
    for _ in colors.iter() {
        ctx.pop_layer();
    }
}

#[vello_test]
fn clip_triangle_with_star(ctx: &mut impl Renderer) {
    let mut triangle_path = BezPath::new();
    triangle_path.move_to((10.0, 10.0));
    triangle_path.line_to((90.0, 20.0));
    triangle_path.line_to((20.0, 90.0));
    triangle_path.close_path();

    let stroke = Stroke::new(1.0);
    ctx.set_paint(DARK_BLUE);
    ctx.set_stroke(stroke);
    ctx.stroke_path(&triangle_path);

    let star_path = circular_star(Point::new(50., 50.), 13, 25., 45.);

    ctx.push_clip_layer(&star_path);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&triangle_path);
    ctx.pop_layer();
}

#[vello_test]
fn clip_rectangle_with_star_nonzero(ctx: &mut impl Renderer) {
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    // Create a self-intersecting star shape that will show the difference between fill rules
    let star_path = crossed_line_star();

    // Set the fill rule to NonZero before applying the clip
    ctx.set_fill_rule(Fill::NonZero);
    // Apply the star as a clip
    ctx.push_clip_layer(&star_path);
    // Draw a rectangle that should be clipped by the star
    // The NonZero fill rule will treat self-intersecting regions as filled
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

#[vello_test]
fn clip_rectangle_with_star_evenodd(ctx: &mut impl Renderer) {
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    // Create a self-intersecting star shape that will show the difference between fill rules
    let star_path = crossed_line_star();

    // Set the fill rule to EvenOdd before applying the clip
    ctx.set_fill_rule(Fill::EvenOdd);
    // Apply the star as a clip
    ctx.push_clip_layer(&star_path);
    // Draw a rectangle that should be clipped by the star
    // The EvenOdd rule should create a "hole" in the middle where the paths overlap
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

#[vello_test]
fn clip_deeply_nested_circles(ctx: &mut impl Renderer) {
    const INITIAL_RADIUS: f64 = 48.0;
    const RADIUS_DECREMENT: f64 = 2.5;
    const INNER_COUNT: usize = 10;
    // `.ceil()` is not constant-evaluatable, so we have to do this at runtime.
    let outer_count: usize =
        (INITIAL_RADIUS / RADIUS_DECREMENT / INNER_COUNT as f64).ceil() as usize;
    const COLORS: [Color; INNER_COUNT] = [
        RED,
        DARK_BLUE,
        DARK_GREEN,
        REBECCA_PURPLE,
        BLACK,
        BLUE,
        GREEN,
        RED,
        DARK_BLUE,
        DARK_GREEN,
    ];

    const COVER_RECT: Rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    const CENTER: Point = Point::new(50.0, 50.0);
    let mut radius = INITIAL_RADIUS;

    for _ in 0..outer_count {
        for color in COLORS.iter() {
            let clip_circle = Circle::new(CENTER, radius).to_path(0.1);
            draw_clipping_outline(ctx, &clip_circle);
            ctx.push_clip_layer(&clip_circle);

            ctx.set_paint(*color);
            ctx.fill_rect(&COVER_RECT);

            radius -= RADIUS_DECREMENT;
        }
    }
    for _ in 0..outer_count {
        for _ in COLORS.iter() {
            ctx.pop_layer();
        }
    }
}

#[vello_test]
fn clip_rectangle_and_circle(ctx: &mut impl Renderer) {
    // Create first clipping region - a rectangle on the left side
    let clip_rect = Rect::new(10.0, 30.0, 50.0, 70.0);

    // Create second clipping region - a circle on the right side
    let circle_center = Point::new(65.0, 50.0);
    let circle_radius = 30.0;
    let clip_circle = Circle::new(circle_center, circle_radius).to_path(0.1);

    // Draw outlines of our clipping regions to visualize them
    let stroke = Stroke::new(1.0);
    ctx.set_paint(DARK_BLUE);
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&clip_rect);
    ctx.stroke_path(&clip_circle);

    // Apply both clips
    ctx.push_clip_layer(&clip_rect.to_path(0.1));
    ctx.push_clip_layer(&clip_circle);

    // Then a filled rectangle that covers most of the canvas
    let large_rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&large_rect);
    ctx.pop_layer();
    ctx.pop_layer();
}

#[vello_test]
fn clip_with_translation(ctx: &mut impl Renderer) {
    // Apply a translation transform
    ctx.set_transform(Affine::translate((30.0, 30.0)));

    // Create and apply a clipping rectangle
    let clip_rect = Rect::new(0.0, 0.0, 40.0, 40.0);
    draw_clipping_outline(ctx, &clip_rect.to_path(0.1));
    ctx.push_clip_layer(&clip_rect.to_path(0.1));

    // Draw a rectangle that should be clipped
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

#[vello_test]
fn clip_with_scale(ctx: &mut impl Renderer) {
    ctx.set_transform(Affine::scale(2.0));

    // Create and apply a clipping rectangle
    let clip_rect = Rect::new(10.0, 10.0, 40.0, 40.0);
    draw_clipping_outline(ctx, &clip_rect.to_path(0.1));
    ctx.push_clip_layer(&clip_rect.to_path(0.1));

    // Draw a rectangle that should be clipped
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

#[vello_test]
fn clip_with_rotate(ctx: &mut impl Renderer) {
    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(50.0, 50.0),
    ));

    // Create and apply a clipping rectangle
    let clip_rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    draw_clipping_outline(ctx, &clip_rect.to_path(0.1));
    ctx.push_clip_layer(&clip_rect.to_path(0.1));

    // Draw a rectangle that should be clipped
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

#[vello_test]
fn clip_transformed_rect(ctx: &mut impl Renderer) {
    let clip_rect = Rect::new(20.0, 20.0, 80.0, 80.0);

    draw_clipping_outline(ctx, &clip_rect.to_path(0.1));

    ctx.push_clip_layer(&clip_rect.to_path(0.1));

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(50.0, 50.0),
    ));

    // Draw a smaller rectangle that should be clipped
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

#[vello_test]
fn clip_with_multiple_transforms(ctx: &mut impl Renderer) {
    // Apply initial transform
    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(50.0, 50.0),
    ));

    // Create and apply first clip
    let clip_rect1 = Rect::new(20.0, 20.0, 80.0, 80.0);
    draw_clipping_outline(ctx, &clip_rect1.to_path(0.1));
    ctx.push_clip_layer(&clip_rect1.to_path(0.1));

    // Apply another transform
    ctx.set_transform(Affine::scale(1.5));

    // Create and apply second clip
    let clip_rect2 = Rect::new(30.0, 30.0, 70.0, 70.0);
    draw_clipping_outline(ctx, &clip_rect2.to_path(0.1));
    ctx.push_clip_layer(&clip_rect2.to_path(0.1));

    // Draw a rectangle that should be clipped by both regions
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
    ctx.pop_layer();
}

#[vello_test]
fn clip_with_save_restore(ctx: &mut impl Renderer) {
    // Create first clipping region - a rectangle on the left side
    let clip_rect1 = Rect::new(10.0, 30.0, 50.0, 70.0);
    draw_clipping_outline(ctx, &clip_rect1.to_path(0.1));
    ctx.push_clip_layer(&clip_rect1.to_path(0.1));

    // Add second clipping region - a circle on the right side
    let circle_center = Point::new(65.0, 50.0);
    let circle_radius = 30.0;
    let clip_circle = Circle::new(circle_center, circle_radius).to_path(0.1);
    draw_clipping_outline(ctx, &clip_circle);
    ctx.push_clip_layer(&clip_circle);

    // Draw a rectangle that should be clipped by both regions
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&rect);

    // Restore to state before second clip
    ctx.pop_layer();

    // Draw another rectangle that should only be clipped by the first region
    let rect2 = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(DARK_GREEN.with_alpha(0.5));
    ctx.fill_rect(&rect2);
    ctx.pop_layer();
}

#[vello_test]
fn clip_with_opacity(ctx: &mut impl Renderer) {
    // Main body of the shape should be RGB 127, 127, 127. Anti-aliased part should be
    // 191, 191, 191.
    let clip_rect = Rect::new(10.5, 10.5, 89.5, 89.5);
    ctx.push_clip_layer(&clip_rect.to_path(0.1));
    ctx.set_paint(BLACK.with_alpha(0.5));
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.pop_layer();
}

fn draw_clipping_outline(ctx: &mut impl Renderer, path: &BezPath) {
    let stroke = Stroke::new(1.0);
    ctx.set_paint(DARK_BLUE);
    ctx.set_stroke(stroke);
    ctx.stroke_path(path);
}
