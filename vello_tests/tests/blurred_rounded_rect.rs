// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use vello_common::color::palette::css::{PALE_GOLDENROD, REBECCA_PURPLE};
use vello_common::kurbo::{Affine, BezPath, Circle, Point, Rect, RoundedRect, Shape, Vec2};
use vello_common::peniko::Fill;
use vello_dev_macros::vello_test;

fn rect_with(ctx: &mut impl Renderer, radius: f32, std_dev: f32, affine: Affine) {
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_transform(affine);
    ctx.fill_blurred_rounded_rect(&rect, radius, std_dev);
}

#[vello_test]
fn blurred_rounded_rect_zero(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 0.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_zero_with_radius(ctx: &mut impl Renderer) {
    rect_with(ctx, 10.0, 0.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_none(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 0.1, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_small_std_dev(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 5.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_medium_std_dev(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_large_std_dev(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 20.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_with_radius(ctx: &mut impl Renderer) {
    rect_with(ctx, 10.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_with_large_radius(ctx: &mut impl Renderer) {
    rect_with(ctx, 30.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_with_transform(ctx: &mut impl Renderer) {
    rect_with(
        ctx,
        10.0,
        10.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
    );
}

fn inverse_rect_with(ctx: &mut impl Renderer, radius: f32, std_dev: f32, affine: Affine) {
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    let path = rect.to_path(0.1);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_transform(affine);
    ctx.fill_blurred_rounded_rect_in(&path, &rect, radius, std_dev, true);
}

#[vello_test]
fn inverse_blurred_rounded_rect_small_std_dev(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 0.0, 5.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_medium_std_dev(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 0.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_large_std_dev(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 0.0, 20.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_with_radius(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 10.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_with_large_radius(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 30.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_with_transform(ctx: &mut impl Renderer) {
    inverse_rect_with(
        ctx,
        10.0,
        10.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
    );
}

fn rect_in_with(
    ctx: &mut impl Renderer,
    path: &BezPath,
    radius: f32,
    std_dev: f32,
    invert: bool,
    affine: Affine,
) {
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_transform(affine);
    ctx.fill_blurred_rounded_rect_in(path, &rect, radius, std_dev, invert);
}

#[vello_test]
fn blurred_rounded_rect_in_circle(ctx: &mut impl Renderer) {
    let path = Circle::new(Point::new(50.0, 50.0), 45.0).to_path(0.1);
    rect_in_with(ctx, &path, 10.0, 10.0, false, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_in_circle_with_transform(ctx: &mut impl Renderer) {
    let path = Circle::new(Point::new(50.0, 50.0), 45.0).to_path(0.1);
    rect_in_with(
        ctx,
        &path,
        10.0,
        10.0,
        false,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
    );
}

/// The typical inset box shadow use case: the inverse blur is clipped to the shape it is
/// inset into.
#[vello_test]
fn inverse_blurred_rounded_rect_in_rect(ctx: &mut impl Renderer) {
    let path = Rect::new(20.0, 20.0, 80.0, 80.0).to_path(0.1);
    rect_in_with(ctx, &path, 10.0, 10.0, true, Affine::IDENTITY);
}

/// Emulate a CSS inset box shadow: the inverse blurred rounded rectangle is offset relative to
/// the border box it is painted into, and clipped to that border box.
fn inset_box_shadow(
    ctx: &mut impl Renderer,
    offset: Vec2,
    radius: f32,
    std_dev: f32,
    affine: Affine,
) {
    let border_box = Rect::new(20.0, 20.0, 80.0, 80.0);
    let path = RoundedRect::from_rect(border_box, f64::from(radius)).to_path(0.1);

    ctx.set_transform(affine);
    ctx.set_paint(PALE_GOLDENROD);
    ctx.fill_path(&path);

    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_blurred_rounded_rect_in(&path, &(border_box + offset), radius, std_dev, true);
}

#[vello_test]
fn inset_box_shadow_offset_down_right(ctx: &mut impl Renderer) {
    inset_box_shadow(ctx, Vec2::new(8.0, 8.0), 10.0, 6.0, Affine::IDENTITY);
}

#[vello_test]
fn inset_box_shadow_offset_up_left(ctx: &mut impl Renderer) {
    inset_box_shadow(ctx, Vec2::new(-8.0, -8.0), 10.0, 6.0, Affine::IDENTITY);
}

#[vello_test]
fn inset_box_shadow_offset_horizontal(ctx: &mut impl Renderer) {
    inset_box_shadow(ctx, Vec2::new(12.0, 0.0), 10.0, 4.0, Affine::IDENTITY);
}

/// An offset shadow larger than the blur, so the shadow has a hard edge and only covers the
/// top and left of the border box.
#[vello_test]
fn inset_box_shadow_offset_without_blur(ctx: &mut impl Renderer) {
    inset_box_shadow(ctx, Vec2::new(15.0, 15.0), 0.0, 0.0, Affine::IDENTITY);
}

#[vello_test]
fn inset_box_shadow_offset_with_transform(ctx: &mut impl Renderer) {
    inset_box_shadow(
        ctx,
        Vec2::new(8.0, 8.0),
        10.0,
        6.0,
        Affine::rotate_about(20.0_f64.to_radians(), Point::new(50.0, 50.0)),
    );
}

/// The shape is filled with the current fill rule, so the overlapping region of the two
/// circles is not painted.
#[vello_test]
fn blurred_rounded_rect_in_even_odd(ctx: &mut impl Renderer) {
    let mut path = Circle::new(Point::new(35.0, 50.0), 40.0).to_path(0.1);
    path.extend(Circle::new(Point::new(65.0, 50.0), 40.0).to_path(0.1));

    ctx.set_fill_rule(Fill::EvenOdd);
    rect_in_with(ctx, &path, 10.0, 10.0, false, Affine::IDENTITY);
}
