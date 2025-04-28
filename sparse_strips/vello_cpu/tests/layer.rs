// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_common::color::palette::css::{BLUE, RED};
use vello_common::kurbo::Rect;
use vello_common::peniko::{BlendMode, Compose, Mix};
use crate::mask::example_mask;
use crate::util::{check_ref, crossed_line_star, get_ctx};

#[test]
fn layer_multiple_properties_1() {
    let mask = example_mask(true);
    let star = crossed_line_star();

    let mut ctx = get_ctx(100, 100, false);
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.push_layer(
        Some(&star), Some(BlendMode::new(Mix::Lighten, Compose::SrcOver)), Some(200), Some(mask)
    );
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.pop_layer();

    check_ref(&ctx, "layer_multiple_properties_1");
}
