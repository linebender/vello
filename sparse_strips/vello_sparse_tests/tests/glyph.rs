// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for glyph rendering.

use crate::renderer::Renderer;
#[cfg(target_os = "macos")]
use crate::util::layout_glyphs_apple_color_emoji;
use crate::util::{layout_glyphs_noto_cbtf, layout_glyphs_noto_colr, layout_glyphs_roboto};
use glifo::Glyph;
use std::f64::consts::FRAC_PI_4;
use std::iter;
use std::sync::Arc;
use vello_common::color::palette::css::{BLACK, BLUE, GREEN, REBECCA_PURPLE};
use vello_common::kurbo::{Affine, Stroke};
use vello_common::peniko::{Blob, FontData};
use vello_dev_macros::vello_test;

fn render_transform_composition_rows(
    ctx: &mut impl Renderer,
    enable_caching: bool,
    hint: bool,
    reverse_x_shift: f64,
    paint: impl Into<vello_common::paint::PaintType>,
    layout: impl Fn(f32) -> (FontData, Vec<Glyph>),
) {
    let rows = [
        (Affine::IDENTITY, 20.0_f32, Affine::IDENTITY),
        (Affine::scale(20.0), 1.0_f32, Affine::IDENTITY),
        (Affine::IDENTITY, 10.0_f32, Affine::scale(2.0)),
        (Affine::scale(2.0), 5.0_f32, Affine::scale(2.0)),
        (
            Affine::translate((-4.0, 0.0)),
            20.0_f32,
            Affine::translate((4.0, 0.0)),
        ),
        (
            Affine::translate((-4.0, 0.0)) * Affine::scale(4.0),
            5.0_f32,
            Affine::translate((1.0, 0.0)),
        ),
        (
            Affine::translate((-1.0, 0.0)),
            40.0_f32,
            Affine::scale(0.5) * Affine::translate((2.0, 0.0)),
        ),
        (
            Affine::IDENTITY,
            20.0_f32,
            Affine::translate((10.0, -10.0))
                * Affine::rotate(FRAC_PI_4)
                * Affine::translate((-10.0, 10.0)),
        ),
        (
            Affine::IDENTITY,
            20.0_f32,
            Affine::translate((10.0, -10.0))
                * Affine::skew(0.35, 0.0)
                * Affine::translate((-10.0, 10.0)),
        ),
        (
            Affine::IDENTITY,
            20.0_f32,
            Affine::translate((10.0, -10.0))
                * Affine::skew(0.0, 0.2)
                * Affine::translate((-10.0, 10.0)),
        ),
        (
            Affine::scale_non_uniform(1.0, -1.0) * Affine::translate((0.0, 20.0)),
            20.0_f32,
            Affine::IDENTITY,
        ),
        (
            Affine::scale_non_uniform(-1.0, 1.0) * Affine::translate((-reverse_x_shift, 0.0)),
            20.0_f32,
            Affine::IDENTITY,
        ),
        (
            Affine::scale_non_uniform(-1.0, -1.0) * Affine::translate((-reverse_x_shift, 20.0)),
            20.0_f32,
            Affine::IDENTITY,
        ),
    ];

    ctx.set_paint(paint);

    let mut y = 28.35;
    for (run_scale_transform, font_size, glyph_transform) in rows {
        let (font, glyphs) = layout(font_size);
        ctx.set_transform(Affine::translate((16.0, y)) * run_scale_transform);
        ctx.glyph_run(&font)
            .font_size(font_size)
            .atlas_cache(enable_caching)
            .glyph_transform(glyph_transform)
            .hint(hint)
            .fill_glyphs(glyphs.into_iter());
        y += 30.0;
    }
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_filled(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_filled_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_stroked(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(true)
        .stroke_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_stroked_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false)
        .stroke_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_stroked_scaled_up(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 5_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(10.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(Stroke {
        width: 0.3,
        ..Stroke::default()
    });
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false)
        .stroke_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_large_stroke_width(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(Stroke {
        width: 3.0,
        ..Stroke::default()
    });
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .stroke_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 120)]
fn glyphs_stroked_then_filled(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    render_roboto_with_mode(
        ctx,
        &font,
        font_size,
        glyphs.iter().copied(),
        Affine::translate((0., f64::from(font_size))),
        DrawMode::Stroke,
    );
    render_roboto_with_mode(
        ctx,
        &font,
        font_size,
        glyphs.into_iter(),
        Affine::translate((0., f64::from(font_size * 2.0))),
        DrawMode::Fill,
    );
}

#[vello_test(width = 300, height = 120)]
fn glyphs_filled_then_stroked(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    render_roboto_with_mode(
        ctx,
        &font,
        font_size,
        glyphs.iter().copied(),
        Affine::translate((0., f64::from(font_size))),
        DrawMode::Fill,
    );
    render_roboto_with_mode(
        ctx,
        &font,
        font_size,
        glyphs.into_iter(),
        Affine::translate((0., f64::from(font_size * 2.0))),
        DrawMode::Stroke,
    );
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_skewed(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_skewed_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 250, height = 75, glyph)]
fn glyphs_skewed_long(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 20_f32;
    let (font, glyphs) = layout_glyphs_roboto(
        "Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit.\nSed ornare arcu lectus.",
        font_size,
    );

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(-10_f64.to_radians().tan(), 0.))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 250, height = 75, glyph)]
fn glyphs_skewed_long_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 20_f32;
    let (font, glyphs) = layout_glyphs_roboto(
        "Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit.\nSed ornare arcu lectus.",
        font_size,
    );

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(-10_f64.to_radians().tan(), 0.))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_skewed_unskewed(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(
        Affine::translate((0., f64::from(font_size)))
            * Affine::skew(-20_f64.to_radians().tan(), 0.),
    );
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(20_f64.to_radians().tan(), 0.))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_skewed_unskewed_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(
        Affine::translate((0., f64::from(font_size)))
            * Affine::skew(-20_f64.to_radians().tan(), 0.),
    );
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(20_f64.to_radians().tan(), 0.))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_scaled(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_scaled_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_glyph_transform(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::translate((10., 10.)))
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_glyph_transform_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::translate((10., 10.)))
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 110, height = 410, glyph, hybrid_tolerance = 1)]
fn glyphs_transform_composition_rows_outline(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(
        ctx,
        enable_caching,
        false,
        47.0,
        REBECCA_PURPLE.with_alpha(0.5),
        |font_size| layout_glyphs_roboto("Hello", font_size),
    );
}

#[vello_test(width = 110, height = 410, glyph, hybrid_tolerance = 1)]
fn glyphs_transform_composition_rows_outline_hinted(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(
        ctx,
        enable_caching,
        true,
        47.0,
        REBECCA_PURPLE.with_alpha(0.5),
        |font_size| layout_glyphs_roboto("Hello", font_size),
    );
}

// Next two tests require high tolerance on CPU likely due to having to use bicubic interpolation, since
// we downscale a lot.

#[vello_test(width = 210, height = 410, skip_hybrid, glyph, cpu_u8_tolerance = 3)]
fn glyphs_transform_composition_rows_bitmap(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(ctx, enable_caching, false, 100.0, BLACK, |font_size| {
        layout_glyphs_noto_cbtf("✅👀🎉🤠", font_size)
    });
}

#[vello_test(width = 210, height = 410, skip_hybrid, glyph, cpu_u8_tolerance = 3)]
fn glyphs_transform_composition_rows_bitmap_hinted(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(ctx, enable_caching, true, 100.0, BLACK, |font_size| {
        layout_glyphs_noto_cbtf("✅👀🎉🤠", font_size)
    });
}

// TODO: The cached versions of COLR glyphs seem to have a slight shift, investigate.

#[vello_test(
    width = 210,
    height = 410,
    cpu_u8_tolerance = 3,
    hybrid_tolerance = 3,
    glyph
)]
fn glyphs_transform_composition_rows_colr(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(ctx, enable_caching, false, 100.0, BLACK, |font_size| {
        layout_glyphs_noto_colr("✅👀🎉🤠", font_size)
    });
}

#[vello_test(
    width = 210,
    height = 410,
    cpu_u8_tolerance = 3,
    hybrid_tolerance = 3,
    glyph
)]
fn glyphs_transform_composition_rows_colr_hinted(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(ctx, enable_caching, true, 100.0, BLACK, |font_size| {
        layout_glyphs_noto_colr("✅👀🎉🤠", font_size)
    });
}

#[vello_test(width = 60, height = 12, glyph)]
fn glyphs_small(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 10_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(true)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 60, height = 12, glyph)]
fn glyphs_small_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 10_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());
}

#[vello_test(width = 250, height = 70, skip_hybrid, glyph)]
fn glyphs_bitmap_noto(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_noto_cbtf("✅👀🎉🤠", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .fill_glyphs(glyphs.into_iter());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrawMode {
    Fill,
    Stroke,
}

fn render_roboto_with_mode(
    ctx: &mut impl Renderer,
    font: &FontData,
    font_size: f32,
    glyphs: impl Iterator<Item = Glyph> + Clone,
    transform: Affine,
    mode: DrawMode,
) {
    ctx.set_transform(transform);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_stroke(Stroke {
        width: 3.0,
        ..Stroke::default()
    });
    let builder = ctx.glyph_run(font).font_size(font_size);

    match mode {
        DrawMode::Fill => {
            builder.fill_glyphs(glyphs);
        }
        DrawMode::Stroke => {
            builder.stroke_glyphs(glyphs);
        }
    }
}

#[vello_test(
    width = 250,
    height = 70,
    cpu_u8_tolerance = 1,
    hybrid_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((0., 50.)),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 250,
    height = 70,
    cpu_u8_tolerance = 1,
    hybrid_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_stroked(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((0., 50.)),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 500,
    height = 140,
    cpu_u8_tolerance = 1,
    hybrid_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_scaled_2x(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((0., 50.)).then_scale(2.0),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 125,
    height = 35,
    cpu_u8_tolerance = 1,
    hybrid_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_scaled_half(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((0., 50.)).then_scale(0.5),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 350,
    height = 350,
    cpu_u8_tolerance = 3,
    hybrid_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_rotated(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((175., 100.)) * Affine::rotate(FRAC_PI_4),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 600,
    height = 600,
    cpu_u8_tolerance = 2,
    hybrid_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_rotated_scaled(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((300., 150.)) * Affine::rotate(FRAC_PI_4) * Affine::scale(2.0),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 250,
    height = 140,
    cpu_u8_tolerance = 1,
    hybrid_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_scaled_non_uniform(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((0., 50.)) * Affine::scale_non_uniform(1.0, 2.0),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 300,
    height = 300,
    cpu_u8_tolerance = 2,
    hybrid_tolerance = 1,
    glyph
)]
fn glyphs_colr_noto_rotated_scaled_non_uniform(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((150., 150.))
            * Affine::rotate(FRAC_PI_4)
            * Affine::scale_non_uniform(1.0, 2.0),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(width = 250, height = 70, skip_hybrid)]
fn glyphs_bitmap_noto_stroked(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_noto_cbtf("✅👀🎉🤠", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .stroke_glyphs(glyphs.into_iter());
}

#[cfg(target_os = "macos")]
#[vello_test(width = 200, height = 70, skip_hybrid, cpu_u8_tolerance = 2, glyph)]
fn glyphs_bitmap_apple(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_apple_color_emoji("✅👀🎉🤠", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .fill_glyphs(glyphs.into_iter());
}

// TODO: TEMPORARILY DISABLED, SEE https://github.com/linebender/vello/pull/1562#issuecomment-4206435802.
// In case anything changes here, compare to https://chromium.googlesource.com/chromium/src/+/main/third_party/blink/web_tests/platform/linux/virtual/text-antialias/colrv1-expected.png
// TODO: The reference image for f32_scalar_cached is still wrong, see https://github.com/linebender/vello/pull/1562.
#[vello_test(
    width = 400,
    height = 960,
    hybrid_tolerance = 1,
    diff_pixels = 55,
    ignore,
    glyph
)]
fn glyphs_colr_test_glyphs(ctx: &mut impl Renderer, enable_caching: bool) {
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
            .atlas_cache(enable_caching)
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
                .atlas_cache(enable_caching)
                .fill_glyphs(glyph_iter);

            cur_x += font_size;
        }
    }
}

/// Hinting is disabled to preserve transforms passed to `prepare_colr_glyph`.
fn render_colr_noto_with_transform(
    ctx: &mut impl Renderer,
    transform: Affine,
    enable_caching: bool,
    mode: DrawMode,
) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_noto_colr("✅👀🎉🤠", font_size);

    ctx.set_transform(transform);
    let run = ctx
        .glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false);

    if mode == DrawMode::Stroke {
        run.stroke_glyphs(glyphs.into_iter());
    } else {
        run.fill_glyphs(glyphs.into_iter());
    }
}
