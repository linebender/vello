// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::{default_blend, fill_single};
use criterion::{Bencher, Criterion};
use vello_common::blurred_rounded_rect::BlurredRoundedRectangle;
use vello_common::coarse::WideTile;
use vello_common::color::palette::css::GREEN;
use vello_common::encode::EncodeExt;
use vello_common::fearless_simd::Simd;
use vello_common::kurbo::{Affine, Point, Rect};
use vello_common::tile::Tile;
use vello_cpu::fine::{Fine, FineKernel};
use vello_dev_macros::vello_bench;

pub fn rounded_blurred_rect(c: &mut Criterion) {
    with_transform(c);
    no_transform(c);
}

#[vello_bench]
fn with_transform<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    let center = Point::new(WideTile::WIDTH as f64 / 2.0, Tile::HEIGHT as f64 / 2.0);

    base(b, fine, Affine::rotate_about(1.0, center));
}

#[vello_bench]
fn no_transform<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>) {
    base(b, fine, Affine::IDENTITY);
}

fn base<S: Simd, N: FineKernel<S>>(b: &mut Bencher<'_>, fine: &mut Fine<S, N>, transform: Affine) {
    let mut paints = vec![];

    let rect = BlurredRoundedRectangle {
        rect: Rect::new(0.0, 0.0, WideTile::WIDTH as f64, Tile::HEIGHT as f64),
        color: GREEN,
        radius: 30.0,
        std_dev: 10.0,
    };

    let paint = rect.encode_into(&mut paints, transform);
    fill_single(
        &paint,
        &paints,
        WideTile::WIDTH as usize,
        b,
        default_blend(),
        fine,
    );
}
