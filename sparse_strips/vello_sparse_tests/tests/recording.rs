// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::util::layout_glyphs_roboto;
use vello_common::kurbo::{Affine, Rect};
use vello_common::recording::Recording;
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

    let mut recording1 = Recording::new();
    ctx.record(&mut recording1, |ctx| {
        ctx.set_paint(ORANGE);
        ctx.fill_rect(&Rect::new(12.0, 52.0, 48.0, 88.0));
        ctx.set_paint(REBECCA_PURPLE);
        ctx.fill_rect(&Rect::new(52.0, 52.0, 88.0, 88.0));
    });

    let mut recording2 = Recording::new();
    ctx.record(&mut recording2, |ctx| {
        ctx.set_paint(ORCHID);
        ctx.fill_rect(&Rect::new(4.0, 12.0, 8.0, 88.0));
        ctx.set_paint(PALE_VIOLET_RED);
        ctx.fill_rect(&Rect::new(92.0, 12.0, 96.0, 88.0));
    });

    ctx.prepare_recording(&mut recording1);
    ctx.execute_recording(&recording1);

    ctx.set_paint(DARK_TURQUOISE);
    ctx.fill_rect(&Rect::new(12.0, 4.0, 88.0, 8.0));
    ctx.set_paint(LIGHT_SALMON);
    ctx.fill_rect(&Rect::new(12.0, 92.0, 88.0, 96.0));

    ctx.prepare_recording(&mut recording2);
    ctx.execute_recording(&recording2);
}

#[vello_test(skip_multithreaded)]
fn recording_incremental_build(ctx: &mut impl Renderer) {
    let mut recording = Recording::new();

    ctx.record(&mut recording, |ctx| {
        ctx.set_paint(GREEN);
        ctx.fill_rect(&Rect::new(12.0, 12.0, 48.0, 48.0));
        ctx.set_paint(FUCHSIA);
        ctx.fill_rect(&Rect::new(52.0, 12.0, 88.0, 48.0));
        ctx.set_paint(ORANGE);
        ctx.fill_rect(&Rect::new(12.0, 52.0, 48.0, 88.0));
        ctx.set_paint(REBECCA_PURPLE);
        ctx.fill_rect(&Rect::new(52.0, 52.0, 88.0, 88.0));
    });
    ctx.prepare_recording(&mut recording);

    ctx.record(&mut recording, |ctx| {
        ctx.set_paint(ORCHID);
        ctx.fill_rect(&Rect::new(4.0, 12.0, 8.0, 88.0));
        ctx.set_paint(PALE_VIOLET_RED);
        ctx.fill_rect(&Rect::new(92.0, 12.0, 96.0, 88.0));
        ctx.set_paint(DARK_TURQUOISE);
        ctx.fill_rect(&Rect::new(12.0, 4.0, 88.0, 8.0));
        ctx.set_paint(LIGHT_SALMON);
        ctx.fill_rect(&Rect::new(12.0, 92.0, 88.0, 96.0));
    });
    ctx.prepare_recording(&mut recording);

    ctx.execute_recording(&recording);
}

#[vello_test(width = 300, height = 70, skip_multithreaded)]
fn recording_glyphs(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    let mut recording = Recording::new();
    ctx.record(&mut recording, |ctx| {
        ctx.set_transform(Affine::translate((0., f64::from(font_size))));
        ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
        ctx.glyph_run(&font)
            .font_size(font_size)
            .hint(true)
            .fill_glyphs(glyphs.into_iter());
    });

    ctx.prepare_recording(&mut recording);
    ctx.execute_recording(&recording);
}
