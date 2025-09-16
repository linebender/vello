// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::util::layout_glyphs_roboto;
use vello_common::color::palette::css::{
    DARK_TURQUOISE, FUCHSIA, GOLD, GREEN, LIGHT_SALMON, ORANGE, ORCHID, PALE_VIOLET_RED, PURPLE,
    REBECCA_PURPLE,
};
use vello_common::kurbo::BezPath;
use vello_common::kurbo::{Affine, Rect};
use vello_common::recording::Recording;
use vello_dev_macros::vello_test;

use crate::renderer::Renderer;

#[vello_test]
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

#[vello_test]
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

#[vello_test(width = 300, height = 70)]
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

#[vello_test(width = 300, height = 70)]
fn glyph_recording_outside_transform(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    // Test differs from `recording_glyphs` as transform is set outside the recording context.
    ctx.set_transform(Affine::translate((0., f64::from(font_size))));

    let mut recording = Recording::new();
    ctx.record(&mut recording, |ctx| {
        ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
        ctx.glyph_run(&font)
            .font_size(font_size)
            .hint(true)
            .fill_glyphs(glyphs.into_iter());
    });

    ctx.prepare_recording(&mut recording);
    ctx.execute_recording(&recording);
}

#[vello_test(width = 50, height = 50)]
fn recording_is_executed_at_recorded_transform(ctx: &mut impl Renderer) {
    ctx.set_transform(Affine::translate((10., 10.)));

    let mut recording = Recording::new();
    ctx.record(&mut recording, |ctx| {
        ctx.set_paint(FUCHSIA);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 30.0, 30.0));
    });

    ctx.prepare_recording(&mut recording);

    ctx.set_transform(Affine::IDENTITY);
    ctx.execute_recording(&recording);
}

#[vello_test(width = 300, height = 100)]
fn recording_mixed_with_direct_drawing(ctx: &mut impl Renderer) {
    let mut recording = Recording::new();
    // Record a rectangle on the left.
    ctx.set_transform(Affine::translate((20.0, 30.0)));
    ctx.record(&mut recording, |ctx| {
        ctx.set_paint(FUCHSIA);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 60.0, 40.0));
    });

    // Do not record central rectangle.
    ctx.set_transform(Affine::IDENTITY);
    ctx.set_paint(LIGHT_SALMON);
    ctx.fill_rect(&Rect::new(120.0, 30.0, 180.0, 70.0));

    // Record a glyph on the right.
    let font_size: f32 = 40_f32;
    let (font, glyphs) = layout_glyphs_roboto("A", font_size);
    ctx.set_transform(Affine::translate((220.0, 60.0)));
    ctx.record(&mut recording, |ctx| {
        ctx.set_paint(ORANGE);
        ctx.glyph_run(&font)
            .font_size(font_size)
            .hint(true)
            .fill_glyphs(glyphs.into_iter());
    });

    ctx.prepare_recording(&mut recording);

    // Paint half the canvas and half of the salmon rectangle.
    ctx.set_transform(Affine::IDENTITY);
    ctx.set_paint(GREEN);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 150.0, 100.0));

    ctx.execute_recording(&recording);
}

#[vello_test(width = 100, height = 100)]
fn recording_can_be_repeatedly_executed_in_layers(ctx: &mut impl Renderer) {
    let mut recording = Recording::new();
    ctx.set_paint(DARK_TURQUOISE);
    ctx.record(&mut recording, |ctx| {
        ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    });
    ctx.prepare_recording(&mut recording);

    for _ in 0..10 {
        ctx.push_opacity_layer(0.02);
        ctx.set_paint(FUCHSIA);
        ctx.execute_recording(&recording);
        ctx.pop_layer();
    }
}

#[vello_test(width = 100, height = 100)]
fn recording_can_be_cleared(ctx: &mut impl Renderer) {
    let mut recording = Recording::new();
    ctx.set_transform(Affine::translate((10., 10.)));
    ctx.record(&mut recording, |ctx| {
        ctx.set_paint(ORANGE);
        ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    });
    ctx.prepare_recording(&mut recording);

    recording.clear();

    ctx.set_transform(Affine::IDENTITY);
    ctx.record(&mut recording, |ctx| {
        ctx.set_paint(GREEN);
        let mut triangle_path = BezPath::new();
        triangle_path.move_to((50.0, 10.0));
        triangle_path.line_to((10.0, 90.0));
        triangle_path.line_to((90.0, 90.0));
        triangle_path.close_path();
        ctx.fill_path(&triangle_path);
    });
    ctx.prepare_recording(&mut recording);
    ctx.execute_recording(&recording);
}

#[vello_test(width = 50, height = 50)]
fn recording_is_executed_with_multiple_transforms(ctx: &mut impl Renderer) {
    ctx.set_transform(Affine::translate((15., 15.)));

    let font_size: f32 = 10_f32;
    let (font, glyphs) = layout_glyphs_roboto("A", font_size);
    let mut recording = Recording::new();
    ctx.record(&mut recording, |ctx| {
        ctx.set_paint(GOLD);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 5.0, 5.0));
        ctx.set_paint(FUCHSIA);
        ctx.set_transform(Affine::translate((20., 20.)));
        ctx.fill_rect(&Rect::new(0.0, 0.0, 5.0, 5.0));
        ctx.set_paint(GREEN);
        ctx.set_transform(Affine::translate((25., 25.)));
        ctx.fill_rect(&Rect::new(0.0, 0.0, 5.0, 5.0));

        ctx.set_transform(Affine::translate((5.0, 15.0)));
        ctx.set_paint(ORANGE);
        ctx.glyph_run(&font)
            .font_size(font_size)
            .hint(true)
            .fill_glyphs(glyphs.into_iter());
    });

    ctx.set_transform(Affine::translate((30., 30.)));
    ctx.set_paint(PALE_VIOLET_RED);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 5.0, 5.0));
    ctx.prepare_recording(&mut recording);

    ctx.set_transform(Affine::translate((35., 35.)));
    ctx.set_paint(PURPLE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 5.0, 5.0));
    ctx.execute_recording(&recording);
}

#[vello_test(width = 20, height = 20, no_ref)]
fn recording_handles_completely_offscreen_content(ctx: &mut impl Renderer) {
    let mut recording = Recording::new();
    ctx.record(&mut recording, |ctx| {
        ctx.set_paint(ORCHID);
        // A rectangle completely outside the viewport in then negative x and y direction.
        ctx.set_transform(Affine::translate((-200., -200.)));
        ctx.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
        // A rectangle completely outside the viewport in then positive x and y direction.
        ctx.set_transform(Affine::translate((200., 200.)));
        ctx.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
    });

    ctx.prepare_recording(&mut recording);
    ctx.execute_recording(&recording);
}
