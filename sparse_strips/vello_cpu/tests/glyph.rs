// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for glyph rendering.

use crate::util::{check_ref, get_ctx, layout_glyphs};
use vello_common::color::palette::css::REBECCA_PURPLE;
use vello_common::kurbo::Affine;

#[test]
fn glyphs_filled() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
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
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
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
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
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
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
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
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.glyph_run(&font)
        .font_size(font_size)
        .horizontal_skew(-20_f32.to_radians())
        .hint(true)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_skewed");
}

#[test]
fn glyphs_skewed_unhinted() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.glyph_run(&font)
        .font_size(font_size)
        .horizontal_skew(-20_f32.to_radians())
        .hint(false)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_skewed_unhinted");
}

#[test]
fn glyphs_scaled() {
    let mut ctx = get_ctx(150, 125, false);
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
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
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
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
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
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
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::translate((10., 10.)))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "glyphs_glyph_transform_unhinted");
}
