// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use criterion::{BenchmarkId, Criterion};
use vello_common::flatten::Line;
use vello_common::tile::Tiles;
use vello_cpu::Level;

fn run_tile_benchmark<F>(c: &mut Criterion, group_name: &str, mut op: F)
where
    F: FnMut(&mut Tiles, &[Line], u16, u16),
{
    let mut g = c.benchmark_group(group_name);
    g.sample_size(50);

    for item in get_data_items() {
        let lines = item.lines();
        g.bench_with_input(BenchmarkId::from_parameter(&item.name), &item, |b, item| {
            b.iter(|| {
                let mut tiler = Tiles::new(Level::new());
                op(&mut tiler, &lines, item.width, item.height);
            });
        });
    }
    g.finish();
}

pub fn tile(c: &mut Criterion) {
    run_tile_benchmark(c, "tile_aaa", |tiler, lines, w, h| {
        tiler.make_tiles_analytic_aa::<false>(
            Level::new(),
            lines,
            w,
            h,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut Vec::new(),
        );
    });

    run_tile_benchmark(c, "tile_msaa", |tiler, lines, w, h| {
        tiler.make_tiles_msaa(lines, w, h);
    });
}
