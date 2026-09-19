// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::SEED;
use crate::fine::{BENCH_WIDTH, default_blend};
use criterion::{Bencher, Criterion};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vello_common::color::palette::css::ROYAL_BLUE;
use vello_common::encode::EncodedPaint;
use vello_common::fearless_simd::Simd;
use vello_common::paint::{NoOpImageResolver, Paint, PremulColor};
use vello_common::tile::Tile;
use vello_cpu::fine::{Fine, FineKernel, FineResources, PaintFillAttrs, Span};
use vello_dev_macros::vello_bench;

pub fn strip(c: &mut Criterion) {
    solid_single(c);
    solid_short(c);
    solid_medium(c);
    solid_long(c);
}

#[vello_bench]
pub fn solid_single<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    let paint = Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE));
    let width = Tile::WIDTH;

    strip_single(&paint, &[], width, b, fine);
}

#[vello_bench]
pub fn solid_short<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    let paint = Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE));
    let width = 8;

    strip_single(&paint, &[], width, b, fine);
}

#[vello_bench]
pub fn solid_medium<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    let paint = Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE));
    let width = 16;

    strip_single(&paint, &[], width, b, fine);
}

#[vello_bench]
pub fn solid_long<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    let paint = Paint::Solid(PremulColor::from_alpha_color(ROYAL_BLUE));
    let width = 64;

    strip_single(&paint, &[], width, b, fine);
}

fn strip_single<S: Simd, N: FineKernel<S>>(
    paint: &Paint,
    encoded_paints: &[EncodedPaint],
    width: u16,
    b: &mut Bencher<'_>,
    fine: &mut Fine<S, N>,
) {
    let mut rng = StdRng::from_seed(SEED);
    let mut alphas = vec![];
    let attrs = PaintFillAttrs {
        paint: paint.clone(),
        blend_mode: default_blend(),
        mask: None,
        draw_id: 1,
        thread_idx: 0,
        origin: (0, 0),
    };

    for _ in 0..BENCH_WIDTH * Tile::HEIGHT {
        alphas.push(rng.random());
    }

    b.iter(|| {
        fine.paint_fill(
            Span::new(0, width),
            &attrs,
            FineResources {
                alpha_buffers: &[],
                encoded_paints,
                filter_paints: &[],
                image_resolver: &NoOpImageResolver,
            },
            Some(&alphas),
        );

        std::hint::black_box(&fine);
    });
}
