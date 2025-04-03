// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Testing composition operators.

use crate::util::{bevel_stroke_2, check_ref, get_ctx};
use vello_common::color::palette::css::{DARK_GREEN, YELLOW};
use vello_common::kurbo::Rect;
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_cpu::RenderContext;

fn compose_destination() -> RenderContext {
    let mut ctx = get_ctx(50, 50, true);
    let rect = Rect::new(4.5, 4.5, 35.5, 35.5);
    ctx.set_paint(YELLOW.with_alpha(0.35).into());
    ctx.set_stroke(bevel_stroke_2());
    ctx.fill_rect(&rect);

    ctx
}

fn compose_source(ctx: &mut RenderContext) {
    let rect = Rect::new(14.5, 14.5, 45.5, 45.5);
    ctx.set_paint(DARK_GREEN.with_alpha(0.8).into());
    ctx.fill_rect(&rect);
}

macro_rules! compose_impl {
    ($mode:path, $name:expr) => {
        let mut ctx = compose_destination();
        ctx.set_blend_mode(BlendMode::new(Mix::Normal, $mode));
        compose_source(&mut ctx);

        check_ref(&ctx, $name);
    };
}

#[test]
fn compose_solid_src_over() {
    compose_impl!(Compose::SrcOver, "compose_solid_src_over");
}
