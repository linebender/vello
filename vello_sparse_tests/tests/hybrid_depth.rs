// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Visual regression tests comparing depth-enabled and depth-disabled hybrid rendering.

use crate::renderer::Renderer;
use vello_common::color::palette::css::{BLUE, GREEN, RED};
use vello_common::kurbo::{Circle, Rect, Shape};
use vello_common::peniko::Color;
use vello_dev_macros::vello_test;

/// Exercise painter ordering when opaque and translucent draws alternate at the root.
#[vello_test(width = 96, height = 96, hybrid_only, hybrid_no_depth)]
fn hybrid_depth_modes_preserve_interleaved_painter_order(ctx: &mut impl Renderer) {
    ctx.set_paint(BLUE.with_alpha(0.5));
    ctx.fill_rect(&Rect::new(8.0, 8.0, 88.0, 88.0));

    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(24.0, 24.0, 72.0, 72.0));

    ctx.set_paint(GREEN.with_alpha(0.5));
    ctx.fill_rect(&Rect::new(40.0, 0.0, 96.0, 64.0));

    ctx.set_paint(Color::from_rgb8(126, 72, 196));
    ctx.fill_path(&Circle::new((68.0, 68.0), 20.0).to_path(0.1));
}

/// Exercise painter ordering and alpha output without an implicit opaque background.
#[vello_test(width = 96, height = 96, transparent, hybrid_only, hybrid_no_depth)]
fn hybrid_depth_modes_preserve_transparent_output(ctx: &mut impl Renderer) {
    ctx.set_paint(BLUE.with_alpha(0.35));
    ctx.fill_rect(&Rect::new(4.0, 4.0, 84.0, 84.0));

    ctx.set_paint(RED);
    ctx.fill_path(&Circle::new((48.0, 48.0), 24.0).to_path(0.1));

    ctx.set_paint(GREEN.with_alpha(0.6));
    ctx.fill_rect(&Rect::new(36.0, 16.0, 92.0, 76.0));
}
