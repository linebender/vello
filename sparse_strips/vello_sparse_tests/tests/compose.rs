// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use vello_common::color::palette::css::{BLUE, YELLOW};
use vello_common::kurbo::Rect;
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_dev_macros::vello_test;

fn compose(ctx: &mut impl Renderer, compose: Compose) {
    ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcOver));

    // Draw the destination layer.
    ctx.set_paint(YELLOW.with_alpha(1.0));
    ctx.fill_rect(&Rect::new(10.0, 10.0, 70.0, 70.0));
    // Draw the source layer.
    ctx.push_blend_layer(BlendMode::new(Mix::Normal, compose));
    ctx.set_paint(BLUE.with_alpha(1.0));
    ctx.fill_rect(&Rect::new(30.0, 30.0, 90.0, 90.0));
    // Compose.
    ctx.pop_layer();
    ctx.pop_layer();
}

#[vello_test]
fn compose_src_over(ctx: &mut impl Renderer) {
    compose(ctx, Compose::SrcOver);
}

#[vello_test]
fn compose_xor(ctx: &mut impl Renderer) {
    compose(ctx, Compose::Xor);
}

#[vello_test]
fn compose_clear(ctx: &mut impl Renderer) {
    compose(ctx, Compose::Clear);
}

#[vello_test]
fn compose_copy(ctx: &mut impl Renderer) {
    compose(ctx, Compose::Copy);
}

#[vello_test]
fn compose_dest(ctx: &mut impl Renderer) {
    compose(ctx, Compose::Dest);
}

#[vello_test]
fn compose_dest_over(ctx: &mut impl Renderer) {
    compose(ctx, Compose::DestOver);
}

#[vello_test]
fn compose_src_in(ctx: &mut impl Renderer) {
    compose(ctx, Compose::SrcIn);
}

#[vello_test]
fn compose_src_out(ctx: &mut impl Renderer) {
    compose(ctx, Compose::SrcOut);
}

#[vello_test]
fn compose_dest_in(ctx: &mut impl Renderer) {
    compose(ctx, Compose::DestIn);
}

#[vello_test]
fn compose_dest_out(ctx: &mut impl Renderer) {
    compose(ctx, Compose::DestOut);
}

#[vello_test]
fn compose_src_atop(ctx: &mut impl Renderer) {
    compose(ctx, Compose::SrcAtop);
}

#[vello_test]
fn compose_dest_atop(ctx: &mut impl Renderer) {
    compose(ctx, Compose::DestAtop);
}

#[vello_test]
fn compose_plus(ctx: &mut impl Renderer) {
    compose(ctx, Compose::Plus);
}
