// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for glyph rendering.

use crate::renderer::Renderer;
#[cfg(target_os = "macos")]
use crate::util::layout_glyphs_apple_color_emoji;
use crate::util::{layout_glyphs_noto_cbtf, layout_glyphs_noto_colr, layout_glyphs_roboto};
use std::iter;
use std::sync::Arc;
use vello_common::color::palette::css::{BLACK, BLUE, GREEN, REBECCA_PURPLE};
use vello_common::glyph::Glyph;
use vello_common::kurbo::Affine;
use vello_common::peniko::{Blob, FontData};
use vello_dev_macros::vello_test;

#[vello_test(width = 300, height = 70)]
fn glyphs_filled(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 70)]
fn glyphs_filled_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 70)]
fn glyphs_stroked(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .stroke_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 70)]
fn glyphs_stroked_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .stroke_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 70)]
fn glyphs_skewed(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 70)]
fn glyphs_skewed_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 250, height = 75)]
fn glyphs_skewed_long(ctx: &mut impl Renderer) {
    let font_size: f32 = 20_f32;
    let (font, glyphs) = layout_glyphs_roboto(
        "Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit.\nSed ornare arcu lectus.",
        font_size,
    );

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(-10_f64.to_radians().tan(), 0.))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 250, height = 75)]
fn glyphs_skewed_long_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 20_f32;
    let (font, glyphs) = layout_glyphs_roboto(
        "Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit.\nSed ornare arcu lectus.",
        font_size,
    );

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(-10_f64.to_radians().tan(), 0.))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125)]
fn glyphs_skewed_unskewed(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(
        Affine::translate((0., f64::from(font_size)))
            * Affine::skew(-20_f64.to_radians().tan(), 0.),
    );
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(20_f64.to_radians().tan(), 0.))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125)]
fn glyphs_skewed_unskewed_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(
        Affine::translate((0., f64::from(font_size)))
            * Affine::skew(-20_f64.to_radians().tan(), 0.),
    );
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(20_f64.to_radians().tan(), 0.))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125)]
fn glyphs_scaled(ctx: &mut impl Renderer) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125)]
fn glyphs_scaled_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125)]
fn glyphs_glyph_transform(ctx: &mut impl Renderer) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::translate((10., 10.)))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125)]
fn glyphs_glyph_transform_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::translate((10., 10.)))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 60, height = 12)]
fn glyphs_small(ctx: &mut impl Renderer) {
    let font_size: f32 = 10_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 60, height = 12)]
fn glyphs_small_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 10_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 250, height = 70, skip_hybrid)]
fn glyphs_bitmap_noto(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_noto_cbtf("✅👀🎉🤠", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 250, height = 70, skip_hybrid, cpu_u8_tolerance = 1)]
fn glyphs_colr_noto(ctx: &mut impl Renderer) {
    render_colr_noto_with_transform(ctx, Affine::translate((0., 50.)));
}

#[vello_test(width = 500, height = 140, skip_hybrid, cpu_u8_tolerance = 1)]
fn glyphs_colr_noto_scaled_2x(ctx: &mut impl Renderer) {
    render_colr_noto_with_transform(ctx, Affine::translate((0., 50.)).then_scale(2.0));
}

#[vello_test(width = 125, height = 35, skip_hybrid, cpu_u8_tolerance = 1)]
fn glyphs_colr_noto_scaled_half(ctx: &mut impl Renderer) {
    render_colr_noto_with_transform(ctx, Affine::translate((0., 50.)).then_scale(0.5));
}

#[vello_test(width = 350, height = 350, skip_hybrid, cpu_u8_tolerance = 3)]
fn glyphs_colr_noto_rotated(ctx: &mut impl Renderer) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((175., 100.)) * Affine::rotate(std::f64::consts::FRAC_PI_4),
    );
}

#[vello_test(width = 600, height = 600, skip_hybrid, cpu_u8_tolerance = 2)]
fn glyphs_colr_noto_rotated_scaled(ctx: &mut impl Renderer) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((300., 150.))
            * Affine::rotate(std::f64::consts::FRAC_PI_4)
            * Affine::scale(2.0),
    );
}

#[vello_test(width = 250, height = 140, skip_hybrid, cpu_u8_tolerance = 1)]
fn glyphs_colr_noto_scaled_non_uniform(ctx: &mut impl Renderer) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((0., 50.)) * Affine::scale_non_uniform(1.0, 2.0),
    );
}

#[vello_test(width = 300, height = 300, skip_hybrid, cpu_u8_tolerance = 2)]
fn glyphs_colr_noto_rotated_scaled_non_uniform(ctx: &mut impl Renderer) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((150., 150.))
            * Affine::rotate(std::f64::consts::FRAC_PI_4)
            * Affine::scale_non_uniform(1.0, 2.0),
    );
}

#[cfg(target_os = "macos")]
#[vello_test(width = 200, height = 70, skip_hybrid, cpu_u8_tolerance = 2)]
fn glyphs_bitmap_apple(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_apple_color_emoji("✅👀🎉🤠", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 400, height = 960, skip_hybrid, diff_pixels = 50)]
fn glyphs_colr_test_glyphs(ctx: &mut impl Renderer) {
    const TEST_FONT: &[u8] =
        include_bytes!("../../../examples/assets/colr_test_glyphs/test_glyphs-glyf_colr_1.ttf");
    let font = FontData::new(Blob::new(Arc::new(TEST_FONT)), 0);
    let num_glyphs = 221;

    let font_size = 40_f64;
    let cols = 10.0;

    let width = (font_size * cols) as u16;

    ctx.set_paint(BLACK);

    let mut cur_x = 0.0;
    let mut cur_y = font_size;

    let draw_glyphs = (0..=num_glyphs).filter(|n| match n {
        // Those are not COLR glyphs.
        0..8 => false,
        161..=165 => false,
        170..=176 => false,
        _ => true,
    });

    for id in draw_glyphs {
        if cur_x >= width as f64 {
            cur_x = 0.0;
            cur_y += font_size;
        }

        let glyph_iter = iter::once(Glyph { id, x: 0.0, y: 0.0 });

        ctx.set_transform(Affine::translate((cur_x, cur_y)));
        ctx.glyph_run(&font)
            .font_size(font_size as f32)
            .fill_glyphs(glyph_iter);

        cur_x += font_size;
    }

    cur_y += font_size;

    for color in [BLUE, GREEN] {
        cur_x = 0.0;
        cur_y += font_size;

        ctx.set_paint(color);

        for g in 148..=153 {
            let glyph_iter = iter::once(Glyph {
                id: g,
                x: 0.0,
                y: 0.0,
            });

            ctx.set_transform(Affine::translate((cur_x, cur_y)));
            ctx.glyph_run(&font)
                .font_size(font_size as f32)
                .fill_glyphs(glyph_iter);

            cur_x += font_size;
        }
    }
}

/// Hinting is disabled to preserve transforms passed to `prepare_colr_glyph`.
fn render_colr_noto_with_transform(ctx: &mut impl Renderer, transform: Affine) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_noto_colr("✅👀🎉🤠", font_size);

    ctx.set_transform(transform);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}
