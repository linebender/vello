// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::FINE_ITERS;
use criterion::Criterion;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use vello_common::coarse::WideTile;
use vello_common::color::palette::css::ROYAL_BLUE;
use vello_common::tile::Tile;
use vello_cpu::fine::Fine;

const SEED: [u8; 32] = [0; 32];

pub fn strip(c: &mut Criterion) {
    let mut g = c.benchmark_group("strip");
    let mut rng = StdRng::from_seed(SEED);

    let mut alphas = vec![];

    for _ in 0..WideTile::WIDTH*Tile::HEIGHT {
        
        alphas.push(rng.random());
    }

    macro_rules! strip_single {
        ($name:ident, $paint:expr) => {
            g.bench_function(stringify!($name), |b| {
                b.iter(|| {
                    let mut out = vec![];
                    let mut fine = Fine::new(WideTile::WIDTH, Tile::HEIGHT, &mut out);

                    for _ in 0..FINE_ITERS {
                        fine.strip(0, WideTile::WIDTH as usize, &alphas, $paint);
                    }
                })
            });
        };
    }

    strip_single!(transparent, &ROYAL_BLUE.into());
}
