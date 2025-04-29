// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for glyph rendering.

use std::iter;
use std::sync::Arc;
use crate::util::{check_ref, get_ctx, layout_glyphs_roboto, layout_glyphs_noto_cbtf, layout_glyphs_noto_colr};
use vello_common::color::palette::css::{BLACK, BLUE, GREEN, REBECCA_PURPLE, WHITE};
use vello_common::glyph::Glyph;
use vello_common::kurbo::{Affine, Rect};
use vello_common::peniko::{Blob, Font};
use vello_cpu::{Pixmap, RenderContext};

#[test]
fn glyphs_filled() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_filled");
}

#[test]
fn glyphs_filled_unhinted() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_filled_unhinted");
}

#[test]
fn glyphs_stroked() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .stroke_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_stroked");
}

#[test]
fn glyphs_stroked_unhinted() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .stroke_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_stroked_unhinted");
}

#[test]
fn glyphs_skewed() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_skewed");
}

#[test]
fn glyphs_skewed_unhinted() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_skewed_unhinted");
}

#[test]
fn glyphs_skewed_long() {
    let mut ctx = get_ctx(250, 75, false);
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

    check_ref(&ctx, "glyphs_skewed_long");
}

#[test]
fn glyphs_skewed_long_unhinted() {
    let mut ctx = get_ctx(250, 75, false);
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

    check_ref(&ctx, "glyphs_skewed_long_unhinted");
}

#[test]
fn glyphs_skewed_unskewed() {
    let mut ctx = get_ctx(150, 125, false);
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

    check_ref(&ctx, "glyphs_skewed_unskewed");
}

#[test]
fn glyphs_skewed_unskewed_unhinted() {
    let mut ctx = get_ctx(150, 125, false);
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

    check_ref(&ctx, "glyphs_skewed_unskewed_unhinted");
}

#[test]
fn glyphs_scaled() {
    let mut ctx = get_ctx(150, 125, false);
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_scaled");
}

#[test]
fn glyphs_scaled_unhinted() {
    let mut ctx = get_ctx(150, 125, false);
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_scaled_unhinted");
}

#[test]
fn glyphs_glyph_transform() {
    let mut ctx = get_ctx(150, 125, false);
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::translate((10., 10.)))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_glyph_transform");
}

#[test]
fn glyphs_glyph_transform_unhinted() {
    let mut ctx = get_ctx(150, 125, false);
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::translate((10., 10.)))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_glyph_transform_unhinted");
}

#[test]
fn glyphs_small() {
    let mut ctx = get_ctx(60, 12, false);
    let font_size: f32 = 10_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_small");
}

#[test]
fn glyphs_small_unhinted() {
    let mut ctx = get_ctx(60, 12, false);
    let font_size: f32 = 10_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_small_unhinted");
}

#[test]
fn glyphs_bitmap_cbdt() {
    let mut ctx = get_ctx(250, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_noto_cbtf("✅👀🎉🤠", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_bitmap_cbdt");
}

#[test]
fn glyphs_colr() {
    let mut ctx = get_ctx(250, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_noto_colr("✅👀🎉🤠", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_colr");
}

#[test]
fn glyphs_colr_test_glyphs() {
    const TEST_FONT: &[u8] = include_bytes!("../../../examples/assets/colr_test_glyphs/test_glyphs-glyf_colr_1.ttf");
    let font = Font::new(Blob::new(Arc::new(TEST_FONT)), 0);
    let num_glyphs = 221;

    let font_size = 40f64;
    let cols = 10.0;
    let rows = 24.0;
    
    let width = (font_size * cols) as u16;
    let height = (font_size * rows) as u16;
    
    let mut ctx = RenderContext::new(width, height);
    let mut pixmap = Pixmap::new(width, height);
    
    ctx.set_paint(WHITE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, width as f64, height as f64));
    ctx.set_paint(BLACK);
    
    let mut cur_x = 0.0;
    let mut cur_y = font_size;
    
    let draw_glyphs = (0..=num_glyphs).into_iter().filter(|n| match n {
        // Those are not COLR glyphs.
        0..8 => false,
        161..=165 => false,
        170..=176 => false,
        _ => true
    });

    for id in draw_glyphs {
        if cur_x >= width as f64 {
            cur_x = 0.0;
            cur_y += font_size;
        }

        let glyph_iter = iter::once(Glyph {
            id,
            x: 0.0,
            y: 0.0,
        });

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
    
    ctx.render_to_pixmap(&mut pixmap);
    
    check_ref(&ctx, "glyphs_colr_test_glyphs");
}
