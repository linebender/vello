// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use criterion::{BenchmarkId, Criterion};
use vello_common::flatten::Line;
use vello_common::tile::Tiles;
use vello_cpu::Level;

fn run_tile_benchmark<F>(c: &mut Criterion, group_name: &str, op: F)
where
    F: Fn(&mut Tiles, &[Line], u16, u16) + Copy,
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
    #[expect(clippy::type_complexity, reason = "tiler variants")]
    let methods: &[(&str, fn(&mut Tiles, &[Line], u16, u16))] = &[
        ("tile_aaa", |t, l, w, h| t.make_tiles_analytic_aa(l, w, h)),
        ("tile_msaa", |t, l, w, h| t.make_tiles_msaa(l, w, h)),
    ];

    for (name, op) in methods {
        run_tile_benchmark(c, name, *op);
    }
}
