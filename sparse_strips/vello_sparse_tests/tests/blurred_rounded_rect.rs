// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use vello_common::color::palette::css::REBECCA_PURPLE;
use vello_common::kurbo::{Affine, Point, Rect};
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
