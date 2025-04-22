// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_common::color::palette::css::{BLUE, GREEN, REBECCA_PURPLE, RED, YELLOW};
use vello_common::kurbo::{Circle, Rect, Shape};
use crate::util::{check_ref, get_ctx};

#[test]
fn opacity_on_layer() {
    let mut ctx = get_ctx(100, 100, false);
    ctx.push_opacity_layer(70);

    for e in [(35.0, 35.0, RED), (65.0, 35.0, GREEN), (50.0, 65.0, BLUE)] {
        let circle = Circle::new((e.0, e.1), 30.0);
        ctx.set_paint(e.2);
        ctx.fill_path(&circle.to_path(0.1));
    }
    
    ctx.pop_layer();

    check_ref(&ctx, "opacity_on_layer");
}

#[test]
fn opacity_nested_on_layer() {
    let mut ctx = get_ctx(100, 100, false);
    
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.push_opacity_layer(128);
    ctx.set_paint(YELLOW);
    ctx.fill_rect(&Rect::new(25.0, 25.0, 75.0, 75.0));
    ctx.push_opacity_layer(128);
    ctx.set_paint(GREEN);
    ctx.fill_rect(&Rect::new(40.0, 40.0, 60.0, 60.0));
    ctx.pop_layer();
    ctx.pop_layer();


    check_ref(&ctx, "opacity_nested_on_layer");
}

