// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]
#![allow(unreachable_pub, reason = "Otherwise benchmarks won't compile")]

use criterion::Criterion;
use rand::prelude::StdRng;
use rand::{RngCore, SeedableRng};
use vello_common::coarse::WideTile;
use vello_common::color::AlphaColor;
use vello_common::paint::Paint;
use vello_common::tile::Tile;
use vello_cpu::fine::Fine;

const STRIP_ITERS: usize = 30;
const SEED: [u8; 32] = [0; 32];

pub fn strip(c: &mut Criterion) {
    let mut g = c.benchmark_group("strip");
    let mut rng = StdRng::from_seed(SEED);

    let mut alphas = vec![];

    for _ in 0..WideTile::WIDTH {
        alphas.push(rng.next_u32());
    }

    macro_rules! strip_single {
        ($name:ident, $paint:expr) => {
            g.bench_function(stringify!($name), |b| {
                b.iter(|| {
                    let mut out = vec![];
                    let mut fine = Fine::new(WideTile::WIDTH, Tile::HEIGHT, &mut out);

                    for _ in 0..STRIP_ITERS {
                        fine.strip(0, WideTile::WIDTH as usize, &alphas, $paint);
                    }
                })
            });
        };
    }

    let solid_paint: Paint = AlphaColor::from_rgba8(230, 129, 185, 160).into();
    strip_single!(fill_transparent, &solid_paint);
}
