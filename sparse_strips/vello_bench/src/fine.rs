// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::SEED;
use criterion::Criterion;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vello_common::coarse::WideTile;
use vello_common::color::palette::css::ROYAL_BLUE;
use vello_common::tile::Tile;
use vello_cpu::fine::Fine;

pub fn fill(c: &mut Criterion) {
    let mut g = c.benchmark_group("fine/fill");

    macro_rules! fill_single {
        ($name:ident, $paint:expr) => {
            g.bench_function(stringify!($name), |b| {
                let mut out = vec![];
                let mut fine = Fine::new(WideTile::WIDTH, Tile::HEIGHT, &mut out);

                b.iter(|| {
                    fine.fill(0, WideTile::WIDTH as usize, $paint);

                    std::hint::black_box(&fine);
                })
            });
        };
    }

    fill_single!(opaque, &ROYAL_BLUE.into());
    fill_single!(transparent, &ROYAL_BLUE.with_alpha(0.2).into());
}

pub fn strip(c: &mut Criterion) {
    let mut g = c.benchmark_group("fine/strip");
    let mut rng = StdRng::from_seed(SEED);

    let mut alphas = vec![];

    for _ in 0..WideTile::WIDTH * Tile::HEIGHT {
        alphas.push(rng.random());
    }

    macro_rules! strip_single {
        ($name:ident, $paint:expr) => {
            g.bench_function(stringify!($name), |b| {
                let mut out = vec![];
                let mut fine = Fine::new(WideTile::WIDTH, Tile::HEIGHT, &mut out);

                b.iter(|| {
                    fine.strip(0, WideTile::WIDTH as usize, &alphas, $paint);

                    std::hint::black_box(&fine);
                })
            });
        };
    }

    strip_single!(basic, &ROYAL_BLUE.into());
}
