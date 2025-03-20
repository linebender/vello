// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Not needed for benchmarks")]
#![allow(unreachable_pub, reason = "Otherwise benchmarks won't compile")]

use criterion::Criterion;
use vello_common::coarse::WideTile;
use vello_common::color::AlphaColor;
use vello_common::paint::Paint;
use vello_common::tile::Tile;
use vello_cpu::fine::Fine;

const FILL_ITERS: usize = 50;

pub fn fill(c: &mut Criterion) {
    let mut g = c.benchmark_group("fill");

    macro_rules! fill_single {
        ($name:ident, $alpha:expr) => {
            g.bench_function(stringify!($name), |b| {
                b.iter(|| {
                    let mut out = vec![];
                    let mut fine = Fine::new(WideTile::WIDTH, Tile::HEIGHT, &mut out);
                    let paint: Paint = AlphaColor::from_rgba8(230, 129, 185, $alpha).into();
                    
                    for _ in 0..FILL_ITERS {
                        fine.fill(0, WideTile::WIDTH as usize, &paint);
                    }
                })
            });
        };
    }

    fill_single!(fill_transparent, 30);
    fill_single!(fill_opaque, 255);
}
