// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use vello_common::color::palette::css::{BLUE, LIME, YELLOW};
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

fn compose_non_isolated(ctx: &mut impl Renderer, compose: Compose) {
    // Just to isolate from the white background.
    ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcOver));

    let rect = Rect::new(10.5, 10.5, 70.5, 70.5);
    ctx.set_paint(BLUE.with_alpha(0.5));
    ctx.fill_rect(&rect);
    ctx.set_blend_mode(BlendMode::new(Mix::Normal, compose));
    let rect = Rect::new(30.5, 30.5, 90.5, 90.5);
    ctx.set_paint(LIME.with_alpha(0.5));
    ctx.fill_rect(&rect);

    ctx.pop_layer();
}

#[vello_test]
fn compose_non_isolated_src_over(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::SrcOver);
}

#[vello_test]
fn compose_non_isolated_xor(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::Xor);
}

#[vello_test]
fn compose_non_isolated_clear(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::Clear);
}

#[vello_test]
fn compose_non_isolated_copy(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::Copy);
}

#[vello_test]
fn compose_non_isolated_dest(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::Dest);
}

#[vello_test]
fn compose_non_isolated_dest_over(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::DestOver);
}

#[vello_test]
fn compose_non_isolated_src_in(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::SrcIn);
}

#[vello_test]
fn compose_non_isolated_src_out(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::SrcOut);
}

#[vello_test]
fn compose_non_isolated_dest_in(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::DestIn);
}

#[vello_test]
fn compose_non_isolated_dest_out(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::DestOut);
}

#[vello_test]
fn compose_non_isolated_src_atop(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::SrcAtop);
}

#[vello_test]
fn compose_non_isolated_dest_atop(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::DestAtop);
}

#[vello_test]
fn compose_non_isolated_plus(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::Plus);
}
