// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for clearing the output target.

use crate::renderer::Renderer;
use crate::util::render_pixmap;
use vello_common::color::AlphaColor;
use vello_common::color::palette::css::{BLUE, LIME};
use vello_common::kurbo::Rect;
use vello_dev_macros::vello_test;
use vello_hybrid::{ClearSettings, RectU16};

const CLEAR_RECTS: &[RectU16] = &[
    RectU16::new(6, 6, 32, 32),
    RectU16::new(56, 10, 90, 44),
    RectU16::new(22, 58, 66, 92),
];
const EMPTY_CLEAR_RECTS: &[RectU16] = &[
    RectU16::new(100, 100, u16::MAX, u16::MAX),
    RectU16::new(24, 24, 12, 12),
];

fn prepare_clear_test(ctx: &mut impl Renderer) {
    ctx.flush();
    let _ = render_pixmap(ctx);
    ctx.reset();
}

#[vello_test]
fn clear_area_viewport_uses_requested_color(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_clear(ClearSettings::Viewport {
        color: AlphaColor::from_rgba8(18, 52, 86, 120),
    });
    ctx.set_paint(LIME);
    ctx.fill_rect(&Rect::new(16.0, 16.0, 48.0, 48.0));
}

#[vello_test]
fn clear_area_rects_use_requested_color(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_clear(ClearSettings::Rects {
        color: AlphaColor::from_rgba8(18, 52, 86, 120),
        rects: CLEAR_RECTS,
    });
    ctx.set_paint(LIME);
    ctx.fill_rect(&Rect::new(8.0, 8.0, 24.0, 24.0));
}

#[vello_test]
fn clear_area_rects_clear_to_transparent(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_clear(ClearSettings::Rects {
        color: AlphaColor::TRANSPARENT,
        rects: CLEAR_RECTS,
    });
}

#[vello_test]
fn clear_area_rects_clear_to_translucent_color(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_clear(ClearSettings::Rects {
        color: BLUE.with_alpha(0.1),
        rects: CLEAR_RECTS,
    });
}

#[vello_test]
fn clear_area_empty_rects_preserve_existing_pixels(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_clear(ClearSettings::Rects {
        color: AlphaColor::from_rgba8(18, 52, 86, 120),
        rects: EMPTY_CLEAR_RECTS,
    });
    ctx.set_paint(LIME);
    ctx.fill_rect(&Rect::new(16.0, 16.0, 48.0, 48.0));
}

#[vello_test]
fn clear_disabled_preserves_existing_pixels(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_clear(ClearSettings::DontClear);
}
