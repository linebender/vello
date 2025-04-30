// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for glyph rendering.

use crate::renderer::Renderer;
use crate::util::layout_glyphs;
use vello_common::color::palette::css::REBECCA_PURPLE;
use vello_common::kurbo::Affine;
use vello_macros::v_test;

#[v_test(width = 300, height = 70)]
fn glyphs_filled(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[v_test(width = 300, height = 70)]
fn glyphs_filled_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[v_test(width = 300, height = 70)]
fn glyphs_stroked(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .stroke_glyphs(glyphs.into_iter());
}

#[v_test(width = 300, height = 70)]
fn glyphs_stroked_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .stroke_glyphs(glyphs.into_iter());
}

#[v_test(width = 300, height = 70)]
fn glyphs_skewed(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[v_test(width = 300, height = 70)]
fn glyphs_skewed_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[v_test(width = 250, height = 75)]
fn glyphs_skewed_long(ctx: &mut impl Renderer) {
    let font_size: f32 = 20_f32;
    let (font, glyphs) = layout_glyphs(
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

#[v_test(width = 250, height = 75)]
fn glyphs_skewed_long_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 20_f32;
    let (font, glyphs) = layout_glyphs(
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

#[v_test(width = 150, height = 125)]
fn glyphs_skewed_unskewed(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

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

#[v_test(width = 150, height = 125)]
fn glyphs_skewed_unskewed_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

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

#[v_test(width = 150, height = 125)]
fn glyphs_scaled(ctx: &mut impl Renderer) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[v_test(width = 150, height = 125)]
fn glyphs_scaled_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[v_test(width = 150, height = 125)]
fn glyphs_glyph_transform(ctx: &mut impl Renderer) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::translate((10., 10.)))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[v_test(width = 150, height = 125)]
fn glyphs_glyph_transform_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::translate((10., 10.)))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[v_test(width = 60, height = 12)]
fn glyphs_small(ctx: &mut impl Renderer) {
    let font_size: f32 = 10_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[v_test(width = 60, height = 12)]
fn glyphs_small_unhinted(ctx: &mut impl Renderer) {
    let font_size: f32 = 10_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}
