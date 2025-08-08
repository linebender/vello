// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_common::kurbo::Rect;
use vello_cpu::color::palette::css::{
    DARK_TURQUOISE, FUCHSIA, GREEN, LIGHT_SALMON, ORANGE, ORCHID, PALE_VIOLET_RED, REBECCA_PURPLE,
};
use vello_dev_macros::vello_test;

use crate::renderer::Renderer;

#[vello_test(skip_multithreaded)]
fn recording_basic(ctx: &mut impl Renderer) {
    ctx.set_paint(GREEN);
    ctx.fill_rect(&Rect::new(12.0, 12.0, 48.0, 48.0));
    ctx.set_paint(FUCHSIA);
    ctx.fill_rect(&Rect::new(52.0, 12.0, 88.0, 48.0));

    let mut recording1 = ctx.record(|ctx| {
        ctx.set_paint(ORANGE);
        ctx.fill_rect(&Rect::new(12.0, 52.0, 48.0, 88.0));
        ctx.set_paint(REBECCA_PURPLE);
        ctx.fill_rect(&Rect::new(52.0, 52.0, 88.0, 88.0));
    });

    let mut recording2 = ctx.record(|ctx| {
        ctx.set_paint(ORCHID);
        ctx.fill_rect(&Rect::new(4.0, 12.0, 8.0, 88.0));
        ctx.set_paint(PALE_VIOLET_RED);
        ctx.fill_rect(&Rect::new(92.0, 12.0, 96.0, 88.0));
    });

    ctx.prepare_recording(&mut recording1);
    ctx.render_recording(&mut recording1);

    ctx.set_paint(DARK_TURQUOISE);
    ctx.fill_rect(&Rect::new(12.0, 4.0, 88.0, 8.0));
    ctx.set_paint(LIGHT_SALMON);
    ctx.fill_rect(&Rect::new(12.0, 92.0, 88.0, 96.0));

    ctx.prepare_recording(&mut recording2);
    ctx.render_recording(&mut recording2);
}
