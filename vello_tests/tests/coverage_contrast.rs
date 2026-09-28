// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for [`CoverageContrast`], the coverage transfer used to sharpen glyphs.

use crate::renderer::{CpuRenderer, Renderer};
use crate::util::{layout_glyphs_roboto, stops_blue_green_red_yellow};
use glifo::Glyph;
use std::sync::Arc;
use vello_common::color::PremulRgba8;
use vello_common::kurbo::{Affine, Point, Rect};
use vello_common::paint::{Color, CoverageContrast, Image, PaintType, Tint, TintMode};
use vello_common::peniko::{Gradient, ImageQuality, ImageSampler, LinearGradientPosition};
use vello_common::pixmap::Pixmap;
use vello_cpu::{Level, RenderMode};

#[cfg(not(target_arch = "wasm32"))]
use crate::renderer::GpuRenderer;

fn detected_level() -> Level {
    Level::try_detect().unwrap_or(Level::baseline())
}

/// Draw an image whose alpha steps through every 8-bit value with a white alpha-mask tint, so
/// that the output alpha is the transferred alpha.
fn render_alpha_ramp<R: Renderer>(
    level: Level,
    render_mode: RenderMode,
    contrast: CoverageContrast,
) -> Pixmap {
    let mut ramp = Pixmap::new(256, 1);
    for x in 0..=u8::MAX {
        ramp.set_pixel(
            u16::from(x),
            0,
            PremulRgba8 {
                r: x,
                g: x,
                b: x,
                a: x,
            },
        );
    }
    ramp.recompute_may_have_transparency();

    let mut ctx = R::new(256, 1, 0, level, render_mode);
    let image = ctx.get_image_source(Arc::new(ramp));
    ctx.set_tint(Some(Tint {
        color: Color::WHITE,
        mode: TintMode::AlphaMask,
        contrast,
    }));
    ctx.set_paint(Image {
        image,
        sampler: ImageSampler {
            quality: ImageQuality::Low,
            ..ImageSampler::default()
        },
    });
    ctx.fill_rect(&Rect::new(0.0, 0.0, 256.0, 1.0));
    ctx.flush();
    ctx.render();
    ctx.snapshot()
}

fn check_alpha_ramp<R: Renderer>(level: Level, render_mode: RenderMode, tolerance: u8) {
    for (contrast, weight) in [(0, 0), (153, 0), (204, 51), (0, 128), (255, 0)] {
        let contrast = CoverageContrast::from_bits(contrast, weight);
        let output = render_alpha_ramp::<R>(level, render_mode, contrast);
        for (a, pixel) in (0..=u8::MAX).zip(output.data()) {
            let expected = contrast.apply_u8(a);
            assert!(
                pixel.a.abs_diff(expected) <= tolerance,
                "{contrast:?} at {a}: {} instead of {expected}",
                pixel.a
            );
        }
    }
}

#[test]
fn coverage_contrast_tint_matches_reference_cpu_u8() {
    // The u8 pipeline evaluates the curve with the same operations as `apply_u8`.
    check_alpha_ramp::<CpuRenderer>(detected_level(), RenderMode::OptimizeSpeed, 0);
    check_alpha_ramp::<CpuRenderer>(Level::baseline(), RenderMode::OptimizeSpeed, 0);
}

#[test]
fn coverage_contrast_tint_matches_reference_cpu_f32() {
    // The f32 pipeline only rounds once, at the end.
    check_alpha_ramp::<CpuRenderer>(detected_level(), RenderMode::OptimizeQuality, 1);
    check_alpha_ramp::<CpuRenderer>(Level::baseline(), RenderMode::OptimizeQuality, 1);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn coverage_contrast_tint_matches_reference_gpu() {
    check_alpha_ramp::<GpuRenderer>(Level::baseline(), RenderMode::OptimizeSpeed, 1);
}

/// A light color, so that the weight resolves to a non-zero value.
const TEXT_COLOR: Color = Color::from_rgb8(250, 210, 90);

/// Render single glyphs at integer positions, so that the atlas holds the same rasterization
/// as the one drawn directly.
fn render_glyphs<R: Renderer>(
    render_mode: RenderMode,
    num_threads: u16,
    paint: &PaintType,
    atlas_cache: bool,
    contrast: CoverageContrast,
) -> Pixmap {
    const FONT_SIZE: f32 = 40.0;
    let (font, glyphs) = layout_glyphs_roboto("Sg&", FONT_SIZE);

    let mut ctx = R::new(128, 48, num_threads, detected_level(), render_mode);
    ctx.set_paint(paint.clone());
    for (i, glyph) in glyphs.iter().enumerate() {
        ctx.set_transform(Affine::translate((4.0 + 40.0 * i as f64, 38.0)));
        ctx.glyph_run(&font)
            .font_size(FONT_SIZE)
            .atlas_cache(atlas_cache)
            .hint(false)
            .coverage_contrast(contrast)
            .fill_glyphs(std::iter::once(Glyph {
                id: glyph.id,
                x: 0.0,
                y: 0.0,
            }))
            .unwrap();
    }
    ctx.flush();
    ctx.render();
    ctx.snapshot()
}

fn max_channel_diff(a: &Pixmap, b: &Pixmap) -> u8 {
    a.data_as_u8_slice()
        .iter()
        .zip(b.data_as_u8_slice())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap()
}

/// A glyph drawn from the atlas, where the transfer is applied to the sampled coverage, matches
/// one drawn from its outline, where it is applied during strip generation, to within one 8-bit
/// level.
fn check_cached_matches_direct<R: Renderer>(render_mode: RenderMode) {
    let contrast = CoverageContrast::new(0.8, 0.3);
    let paint = TEXT_COLOR.into();

    let direct = render_glyphs::<R>(render_mode, 0, &paint, false, contrast);
    let linear = render_glyphs::<R>(render_mode, 0, &paint, false, CoverageContrast::NONE);
    assert!(
        max_channel_diff(&direct, &linear) > 16,
        "the transfer must apply to glyphs drawn from their outlines"
    );

    let cached = render_glyphs::<R>(render_mode, 0, &paint, true, contrast);
    let diff = max_channel_diff(&cached, &direct);
    assert!(diff <= 1, "cached and direct glyphs differ by {diff}");
}

#[test]
fn coverage_contrast_glyph_cached_matches_direct_cpu_u8() {
    check_cached_matches_direct::<CpuRenderer>(RenderMode::OptimizeSpeed);
}

#[test]
fn coverage_contrast_glyph_cached_matches_direct_cpu_f32() {
    check_cached_matches_direct::<CpuRenderer>(RenderMode::OptimizeQuality);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn coverage_contrast_glyph_cached_matches_direct_gpu() {
    check_cached_matches_direct::<GpuRenderer>(RenderMode::OptimizeSpeed);
}

#[test]
fn coverage_contrast_glyph_multithreaded_matches_single_threaded() {
    let contrast = CoverageContrast::new(0.8, 0.3);
    let paint = TEXT_COLOR.into();
    let render = |num_threads| {
        render_glyphs::<CpuRenderer>(
            RenderMode::OptimizeSpeed,
            num_threads,
            &paint,
            false,
            contrast,
        )
    };

    assert!(
        render(3).data() == render(0).data(),
        "multi-threaded rendering must apply the same transfer"
    );
}

/// The weight depends on the text color, so glyphs filled with other paints only get the
/// contrast.
#[test]
fn coverage_contrast_glyph_non_solid_paint_ignores_weight() {
    let paint = Gradient {
        kind: LinearGradientPosition {
            start: Point::new(0.0, 0.0),
            end: Point::new(128.0, 0.0),
        }
        .into(),
        stops: stops_blue_green_red_yellow(),
        ..Default::default()
    }
    .into();
    let render = |contrast| {
        render_glyphs::<CpuRenderer>(RenderMode::OptimizeSpeed, 0, &paint, true, contrast)
    };

    let contrast_only = render(CoverageContrast::from_bits(204, 0));
    assert!(
        contrast_only.data() != render(CoverageContrast::NONE).data(),
        "the contrast must apply to glyphs with non-solid paints"
    );
    assert!(
        render(CoverageContrast::from_bits(204, 51)).data() == contrast_only.data(),
        "the weight must not apply to glyphs with non-solid paints"
    );
}
