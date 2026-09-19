// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::mask::example_mask;
use crate::renderer::Renderer;
use crate::util::crossed_line_star;
use vello_common::color::palette::css::{BLUE, RED};
use vello_common::kurbo::Rect;
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_dev_macros::vello_test;

#[vello_test(cpu_u8_tolerance = 1)]
fn layer_multiple_properties_1(ctx: &mut impl Renderer) {
    let mask = example_mask(true);
    let star = crossed_line_star();

    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.push_layer(
        Some(&star),
        Some(BlendMode::new(Mix::Lighten, Compose::SrcOver)),
        Some(0.78),
        Some(mask),
        None,
    );
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.pop_layer();
}
