// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for initializing the output target.

use crate::renderer::Renderer;
use vello_common::color::AlphaColor;
use vello_common::color::palette::css::{BLUE, LIME};
use vello_common::kurbo::{Rect, Shape};
use vello_common::peniko::{BlendMode, Compose, Mix};
use vello_dev_macros::vello_test;
use vello_hybrid::{ClearSettings, RectU16, TargetInit};

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
    ctx.render();
    ctx.reset();
}

fn draw_destructive_root_blend(ctx: &mut impl Renderer) {
    ctx.set_paint(LIME);
    ctx.fill_rect(&Rect::new(8.0, 24.0, 92.0, 76.0));

    let rect = Rect::new(36.0, 36.0, 64.0, 64.0);
    let clip = rect.to_path(0.1);
    ctx.push_layer(
        Some(&clip),
        Some(BlendMode::new(Mix::Normal, Compose::Clear)),
        None,
        None,
        None,
    );
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

#[vello_test]
fn clear_area_viewport_uses_requested_color(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Viewport {
        color: AlphaColor::from_rgba8(18, 52, 86, 120),
    }));
    ctx.set_paint(LIME);
    ctx.fill_rect(&Rect::new(16.0, 16.0, 48.0, 48.0));
}

#[vello_test]
fn clear_area_viewport_is_preserved_under_src_over(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Viewport { color: BLUE }));
    ctx.set_blend_mode(BlendMode::default());
    ctx.set_paint(LIME.with_alpha(0.5));
    ctx.fill_rect(&Rect::new(16.0, 16.0, 48.0, 48.0));
}

#[vello_test]
fn clear_background_isolates_root_blend(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Viewport { color: BLUE }));
    draw_destructive_root_blend(ctx);
}

#[vello_test(skip_webgl)]
fn src_over_isolates_root_blend(ctx: &mut impl Renderer) {
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    prepare_clear_test(ctx);

    ctx.set_target_init(TargetInit::SrcOver);
    draw_destructive_root_blend(ctx);
}

#[vello_test(skip_webgl)]
fn clear_area_rects_use_requested_color(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Rects {
        color: AlphaColor::from_rgba8(18, 52, 86, 120),
        rects: CLEAR_RECTS,
    }));
    ctx.set_paint(LIME);
    ctx.fill_rect(&Rect::new(8.0, 8.0, 24.0, 24.0));
}

#[vello_test(skip_webgl)]
fn clear_area_rects_clear_to_transparent(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Rects {
        color: AlphaColor::TRANSPARENT,
        rects: CLEAR_RECTS,
    }));
}

#[vello_test(skip_webgl)]
fn clear_area_rects_clear_to_translucent_color(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Rects {
        color: BLUE.with_alpha(0.1),
        rects: CLEAR_RECTS,
    }));
}

#[vello_test(skip_webgl)]
fn clear_area_empty_rects_preserve_existing_pixels(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Rects {
        color: AlphaColor::from_rgba8(18, 52, 86, 120),
        rects: EMPTY_CLEAR_RECTS,
    }));
    ctx.set_paint(LIME);
    ctx.fill_rect(&Rect::new(16.0, 16.0, 48.0, 48.0));
}

#[vello_test(skip_webgl)]
fn clear_disabled_preserves_existing_pixels(ctx: &mut impl Renderer) {
    prepare_clear_test(ctx);
    ctx.set_target_init(TargetInit::SrcOver);
}
