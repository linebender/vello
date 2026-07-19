// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use vello_common::color::palette::css::MIDNIGHT_BLUE;
use vello_common::fearless_simd::{Simd, SimdBase, SimdFloat, dispatch, f32x4, f32x16, u8x16};
use vello_common::kurbo::{Affine, Point, Rect};
use vello_cpu::fine::{NumericVec, PosExt};
use vello_cpu::{CustomPaint, Level, PaintShader, PaintSpan};
use vello_dev_macros::vello_test;

const RECT: Rect = Rect::new(5.0, 5.0, 95.0, 95.0);

struct MeshPaint {
    level: Level,
}

impl PaintShader for MeshPaint {
    fn paint_u8(&self, buffer: &mut [u8], span: PaintSpan) {
        dispatch!(self.level, simd => paint_u8(simd, buffer, span));
    }

    fn paint_f32(&self, buffer: &mut [f32], span: PaintSpan) {
        dispatch!(self.level, simd => paint_f32(simd, buffer, span));
    }
}

fn paint_u8<S: Simd>(simd: S, buffer: &mut [u8], span: PaintSpan) {
    simd.vectorize(
        #[inline(always)]
        || {
            let mut position = span.start;
            for column in buffer.chunks_exact_mut(16) {
                let colors = mesh_colors(simd, position, span);
                let mut interleaved = [0.0; 16];
                simd.store_interleaved_128_f32x16(
                    colors,
                    <&mut [f32; 16]>::try_from(interleaved.as_mut_slice()).unwrap(),
                );
                let colors = f32x16::from_slice(simd, &interleaved);
                <u8x16<S> as NumericVec<S>>::from_f32(simd, colors).store_slice(column);
                position += span.x_advance;
            }
        },
    );
}

fn paint_f32<S: Simd>(simd: S, buffer: &mut [f32], span: PaintSpan) {
    simd.vectorize(
        #[inline(always)]
        || {
            let mut position = span.start;
            for column in buffer.chunks_exact_mut(16) {
                let colors = mesh_colors(simd, position, span);
                simd.store_interleaved_128_f32x16(
                    colors,
                    <&mut [f32; 16]>::try_from(column).unwrap(),
                );
                position += span.x_advance;
            }
        },
    );
}

#[inline(always)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the shader intentionally evaluates paint coordinates in f32 precision"
)]
fn mesh_colors<S: Simd>(simd: S, position: Point, span: PaintSpan) -> f32x16<S> {
    let x = f32x4::splat_pos(
        simd,
        position.x as f32,
        span.x_advance.x as f32,
        span.y_advance.x as f32,
    );
    let y = f32x4::splat_pos(
        simd,
        position.y as f32,
        span.x_advance.y as f32,
        span.y_advance.y as f32,
    );
    let zero = f32x4::splat(simd, 0.0);
    let one = f32x4::splat(simd, 1.0);
    let scale = f32x4::splat(simd, 1.0 / 90.0);
    let u = ((x - f32x4::splat(simd, 5.0)) * scale).max(zero).min(one);
    let v = ((y - f32x4::splat(simd, 5.0)) * scale).max(zero).min(one);
    let bend = f32x4::splat(simd, 0.7);
    let u = u + (v - f32x4::splat(simd, 0.5)) * u * (one - u) * bend;
    let v = v - (u - f32x4::splat(simd, 0.5)) * v * (one - v) * bend;
    let top = one - v;
    let left = one - u;
    let red = left * top + left * v + u * v * f32x4::splat(simd, 0.48);
    let green = u * top * f32x4::splat(simd, 0.72)
        + left * v * f32x4::splat(simd, 0.65)
        + u * v * f32x4::splat(simd, 0.12);
    let blue = left * top * f32x4::splat(simd, 0.62) + u * top + u * v * f32x4::splat(simd, 0.92);

    simd.combine_f32x8(
        simd.combine_f32x4(red, green),
        simd.combine_f32x4(blue, one),
    )
}

fn draw_paint_custom(ctx: &mut impl Renderer, paint_transform: Affine) {
    ctx.set_paint(MIDNIGHT_BLUE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.set_paint_transform(paint_transform);
    ctx.set_custom_paint(
        CustomPaint::new(MeshPaint {
            level: Level::fallback(),
        })
        .opaque(),
    );
    ctx.fill_rect(&RECT);
}

fn about_center(transform: Affine) -> Affine {
    let center = RECT.center();
    Affine::translate((center.x, center.y)) * transform * Affine::translate((-center.x, -center.y))
}

#[vello_test(skip_hybrid)]
fn paint_custom(ctx: &mut impl Renderer) {
    draw_paint_custom(ctx, Affine::IDENTITY);
}

#[vello_test(skip_hybrid)]
fn paint_custom_scaled(ctx: &mut impl Renderer) {
    draw_paint_custom(ctx, about_center(Affine::scale(2.0)));
}

#[vello_test(skip_hybrid)]
fn paint_custom_center_rotation(ctx: &mut impl Renderer) {
    draw_paint_custom(
        ctx,
        about_center(Affine::rotate(core::f64::consts::FRAC_PI_4)),
    );
}
