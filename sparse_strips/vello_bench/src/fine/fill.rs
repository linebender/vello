// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::default_blend;
use criterion::{Bencher, Criterion};
use vello_common::color::palette::css::ROYAL_BLUE;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::Simd;
use vello_common::paint::{Paint, PremulColor};
use vello_common::peniko::BlendMode;
use vello_cpu::fine::{Fine, FineKernel};
use vello_dev_macros::vello_bench;

pub fn fill(c: &mut Criterion) {
    opaque_short(c);
    opaque_long(c);
    transparent_short(c);
    transparent_long(c);
}

#[vello_bench]
pub fn opaque_short<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    let paint = Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE));
    let width = 32;

    fill_single(&paint, &[], width, b, default_blend(), fine);
}

#[vello_bench]
pub fn opaque_long<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    let paint = Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE));
    let width = 256;

    fill_single(&paint, &[], width, b, default_blend(), fine);
}

#[vello_bench]
pub fn transparent_short<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    let paint = Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE.with_alpha(0.3)));
    let width = 32;

    fill_single(&paint, &[], width, b, default_blend(), fine);
}

#[vello_bench]
pub fn transparent_long<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    let paint = Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE.with_alpha(0.3)));
    let width = 256;

    fill_single(&paint, &[], width, b, default_blend(), fine);
}

pub(crate) fn fill_single<S: Simd, N: FineKernel<S>>(
    paint: &Paint,
    encoded_paints: &[EncodedPaint],
    width: usize,
    b: &mut Bencher<'_>,
    blend_mode: BlendMode,
    fine: &mut Fine<S, N>,
) {
    b.iter(|| {
        fine.fill(0, width, paint, blend_mode, encoded_paints, None, None);

        std::hint::black_box(&fine);
    });
}
