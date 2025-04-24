// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_common::color::palette::css::{BLUE, YELLOW};
use vello_common::kurbo::Rect;
use vello_common::peniko::{BlendMode, Compose, Mix};
use crate::util::{check_ref, get_ctx};

fn compose(name: &str, compose: Compose) {
    let mut ctx = get_ctx(100, 100, false);
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

    check_ref(&ctx, name);
}

#[test]
fn compose_src_over() {
    compose("compose_src_over", Compose::SrcOver);
}

#[test]
fn compose_xor() {
    compose("compose_xor", Compose::Xor);
}

#[test]
fn compose_clear() {
    compose("compose_clear", Compose::Clear);
}

#[test]
fn compose_copy() {
    compose("compose_copy", Compose::Copy);
}

#[test]
fn compose_dest() {
    compose("compose_dest", Compose::Dest);
}

#[test]
fn compose_dest_over() {
    compose("compose_dest_over", Compose::DestOver);
}

#[test]
fn compose_src_in() {
    compose("compose_src_in", Compose::SrcIn);
}

#[test]
fn compose_src_out() {
    compose("compose_src_out", Compose::SrcOut);
}

#[test]
fn compose_dest_in() {
    compose("compose_dest_in", Compose::DestIn);
}

#[test]
fn compose_dest_out() {
    compose("compose_dest_out", Compose::DestOut);
}

#[test]
fn compose_src_atop() {
    compose("compose_src_atop", Compose::SrcAtop);
}

#[test]
fn compose_dest_atop() {
    compose("compose_dest_atop", Compose::DestAtop);
}

#[test]
fn compose_plus() {
    compose("compose_plus", Compose::Plus);
}
