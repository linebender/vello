// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use vello_common::color::palette::css::{BLUE, GREEN, REBECCA_PURPLE, RED, YELLOW};
use vello_common::kurbo::{Circle, Rect, Shape};
use vello_dev_macros::vello_test;

#[vello_test]
fn opacity_on_layer(ctx: &mut impl Renderer) {
    ctx.push_opacity_layer(0.27);

    for e in [(35.0, 35.0, RED), (65.0, 35.0, GREEN), (50.0, 65.0, BLUE)] {
        let circle = Circle::new((e.0, e.1), 30.0);
        ctx.set_paint(e.2);
        ctx.fill_path(&circle.to_path(0.1));
    }

    ctx.pop_layer();
}

#[vello_test]
fn opacity_nested_on_layer(ctx: &mut impl Renderer) {
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.push_opacity_layer(0.5);
    ctx.set_paint(YELLOW);
    ctx.fill_rect(&Rect::new(25.0, 25.0, 75.0, 75.0));
    ctx.push_opacity_layer(0.5);
    ctx.set_paint(GREEN);
    ctx.fill_rect(&Rect::new(40.0, 40.0, 60.0, 60.0));
    ctx.pop_layer();
    ctx.pop_layer();
}
