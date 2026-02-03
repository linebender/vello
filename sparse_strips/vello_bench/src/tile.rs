// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use criterion::{BenchmarkId, Criterion};
use vello_common::fearless_simd::Level;
use vello_common::tile::Tiles;

fn run_msaa_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("tile_msaa");
    g.sample_size(50);

    for item in get_data_items() {
        let lines = item.lines();
        g.bench_with_input(BenchmarkId::from_parameter(&item.name), &item, |b, item| {
            b.iter(|| {
                let mut tiler = Tiles::new(Level::new());
                tiler.make_tiles_msaa(&lines, item.width, item.height);
            });
        });
    }
    g.finish();
}

fn run_aaa_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("tile_aaa");
    g.sample_size(50);

    for item in get_data_items() {
        let lines = item.lines();
        g.bench_with_input(BenchmarkId::from_parameter(&item.name), &item, |b, _| {
            b.iter(|| {
                let mut tiler = Tiles::new(Level::new());
                tiler.make_tiles_analytic_aa::<false>(
                    Level::fallback(),
                    &lines,
                    item.width,
                    item.height,
                    &mut Vec::new(),
                    &mut Vec::new(),
                );
            });
        });
    }
    g.finish();
}

pub fn tile(c: &mut Criterion) {
    run_msaa_benchmark(c);
    run_aaa_benchmark(c);
}
