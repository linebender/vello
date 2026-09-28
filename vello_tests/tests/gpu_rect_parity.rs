// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Byte-parity tests for the two rectangle rasterizers of `vello_gpu`.
//!
//! Axis-aligned rectangles are drawn as quads with analytic anti-aliasing, unless a clip path is
//! active: then they go through the strip renderer in `vello_common::rect`, like on the CPU. The
//! strip renderer is the reference, and the quads must reproduce its alphas exactly. Most tests
//! render the same content with and without a clip path covering the whole viewport. That clip
//! removes nothing, so any byte difference is a divergence between the two rasterizers.

// On wasm32, only the WebGL harness can run these.
#![cfg(any(not(target_arch = "wasm32"), feature = "webgl"))]

use vello_common::geometry::RectU16;
use vello_common::kurbo::{BezPath, Rect, Shape, Stroke};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::{Color, Extend, ImageQuality, ImageSampler};
use vello_common::pixmap::Pixmap;
use vello_cpu::RenderMode;

use crate::load_image;
use crate::renderer::{GpuRenderer, Renderer};
use crate::util::get_ctx;

const W: u16 = 128;
const H: u16 = 64;

/// Render `paint` over an opaque backdrop, optionally inside a clip path covering the viewport,
/// which moves every rectangle from the quad path to the strip path.
fn render(
    width: u16,
    height: u16,
    clip_path: bool,
    paint: impl FnOnce(&mut GpuRenderer),
) -> Pixmap {
    let mut ctx = get_ctx::<GpuRenderer>(
        width,
        height,
        false,
        0,
        "baseline",
        RenderMode::OptimizeSpeed,
    );
    ctx.set_paint(Color::from_rgb8(22, 27, 34));
    ctx.fill_rect(&viewport(width, height));
    if clip_path {
        ctx.push_clip_path(&viewport(width, height).to_path(0.1));
    }
    paint(&mut ctx);
    if clip_path {
        ctx.pop_clip();
    }
    ctx.render();
    ctx.snapshot()
}

fn viewport(width: u16, height: u16) -> Rect {
    Rect::new(0.0, 0.0, f64::from(width), f64::from(height))
}

/// Fractional-edge content of every fast-path kind (opaque, translucent, blurred and image-painted
/// rectangles, split into parts or not), plus a path and a stroke, which take the strip path either
/// way.
fn paint_content(ctx: &mut GpuRenderer) {
    // Edges less than half an alpha step inside the pixel grid: the boundary alphas are 255, but
    // the top-left alpha is 254.
    ctx.set_paint(Color::new([0.15, 0.45, 0.85, 1.0]));
    ctx.fill_rect(&Rect::new(102.001, 44.001, 122.0, 60.0));
    ctx.set_paint(Color::new([0.35, 0.7, 0.75, 1.0]));
    ctx.fill_rect(&Rect::new(4.001, 28.001, 40.3, 62.6));
    ctx.set_paint(Color::new([0.941, 0.533, 0.243, 1.0]));
    ctx.fill_rect(&Rect::new(10.3, 8.7, 40.6, 30.2));
    ctx.set_paint(Color::new([0.2, 0.8, 0.4, 0.5]));
    ctx.fill_rect(&Rect::new(60.5, 12.25, 95.75, 40.5));
    ctx.set_paint(Color::new([0.3, 0.2, 0.7, 0.9]));
    ctx.fill_blurred_rounded_rect(&Rect::new(44.4, 34.6, 74.2, 58.9), 4.0, 3.0, false);
    let mut path = BezPath::new();
    path.move_to((100.3, 10.4));
    path.line_to((120.7, 18.9));
    path.line_to((104.2, 44.6));
    path.close_path();
    ctx.set_paint(Color::new([0.4, 0.4, 0.9, 1.0]));
    ctx.fill_path(&path);
    ctx.set_paint(Color::new([0.9, 0.2, 0.3, 0.8]));
    ctx.set_stroke(Stroke::new(2.5));
    ctx.stroke_path(&Rect::new(30.6, 35.3, 70.4, 55.8).to_path(0.1));
    for (rect, color) in fractional_rects(40) {
        ctx.set_paint(color);
        ctx.fill_rect(&rect);
    }
    // Image paints take a different fragment branch.
    let texture_id = ctx.register_external_texture(load_image!("color_grid_16x16"));
    ctx.set_paint(Image {
        image: ImageSource::external_texture(texture_id, RectU16::new(0, 0, 16, 16), false),
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality: ImageQuality::Low,
            alpha: 1.0,
        },
    });
    ctx.fill_rect(&Rect::new(2.3, 1.6, 16.7, 14.4));
}

/// Deterministic rectangles from sub-pixel slivers to multiple tiles, opaque and translucent.
fn fractional_rects(count: usize) -> Vec<(Rect, Color)> {
    // A simple LCG keeps the scene deterministic without a `rand` dependency.
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as f64 / f64::from(u32::MAX >> 1)
    };

    (0..count)
        .map(|_| {
            let x0 = next() * f64::from(W - 12);
            let y0 = next() * f64::from(H - 12);
            let w = 0.3 + next() * 11.0;
            let h = 0.3 + next() * 11.0;
            let color = Color::new([
                next() as f32,
                next() as f32,
                next() as f32,
                (0.3 + 0.7 * next()) as f32,
            ]);
            (Rect::new(x0, y0, x0 + w, y0 + h), color)
        })
        .collect()
}

fn assert_identical(actual: &Pixmap, expected: &Pixmap, what: &str) {
    let diffs = actual
        .data_as_u8_slice()
        .iter()
        .zip(expected.data_as_u8_slice())
        .filter(|(a, e)| a != e)
        .count();
    assert_eq!(diffs, 0, "{what}: {diffs} differing bytes");
}

/// A clip that removes nothing must change nothing, even though it moves every rectangle from the
/// quad path to the strip path.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn gpu_rect_parity_full_viewport_clip() {
    let plain = render(W, H, false, paint_content);
    let clipped = render(W, H, true, paint_content);
    assert_identical(&clipped, &plain, "full-viewport clip");
}

/// Converting an `f64` edge to `f32` moves it by up to half an `f32` ulp (about 2.4e-4 px at
/// x ~ 4400), which is enough to change the rounding of a coverage byte. The quads must derive
/// their coverage from the same `f32` edges as the strip renderer.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn gpu_rect_parity_large_coordinates() {
    const BIG_W: u16 = 4400;
    const BIG_H: u16 = 16;

    let paint = |ctx: &mut GpuRenderer| {
        ctx.set_paint(Color::new([0.941, 0.533, 0.243, 1.0]));
        ctx.fill_rect(&Rect::new(4300.3, 2.7, 4390.6, 12.2));
        ctx.set_paint(Color::new([0.2, 0.8, 0.4, 0.5]));
        ctx.fill_rect(&Rect::new(4210.55, 5.25, 4285.75, 9.5));
    };
    let plain = render(BIG_W, BIG_H, false, paint);
    let clipped = render(BIG_W, BIG_H, true, paint);
    assert_identical(&clipped, &plain, "large-coordinate rects");
}

/// Under a rectangular clip, rectangles stay on the quad path and are intersected with the clip
/// geometrically, so their coverage is that of the intersection, quantized once. That must be
/// exactly what the strip renderer produces for the intersection.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn gpu_rect_parity_rect_clip_intersection() {
    // Fractional, near-integer (less than half an alpha step inside) and half-pixel edges.
    let clip = Rect::new(13.37, 6.001, 109.5, 55.8);
    let mut rects = fractional_rects(60);
    rects.push((
        Rect::new(0.0, 0.0, 30.0, 20.0),
        Color::new([0.1, 0.6, 0.9, 1.0]),
    ));
    rects.push((
        Rect::new(90.25, 40.0, 128.0, 64.0),
        Color::new([0.9, 0.6, 0.1, 1.0]),
    ));

    let analytic = render(W, H, false, |ctx| {
        ctx.push_clip_rect(&clip);
        for (rect, color) in &rects {
            ctx.set_paint(*color);
            ctx.fill_rect(rect);
        }
        ctx.pop_clip();
    });
    let strips = render(W, H, true, |ctx| {
        for (rect, color) in &rects {
            ctx.set_paint(*color);
            ctx.fill_rect(&rect.intersect(clip));
        }
    });
    assert_identical(&analytic, &strips, "rect clip intersection");
}
