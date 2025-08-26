// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use criterion::Criterion;
use vello_common::tile::Tiles;
use vello_cpu::Level;

pub fn tile(c: &mut Criterion) {
    let mut g = c.benchmark_group("tile");
    g.sample_size(50);

    macro_rules! tile_single {
        ($item:expr) => {
            let lines = $item.lines();

            g.bench_function($item.name.clone(), |b| {
                b.iter(|| {
                    let mut tiler = Tiles::new(Level::new());
                    tiler.make_tiles(&lines, $item.width, $item.height);
                })
            });
        };
    }

    for item in get_data_items() {
        tile_single!(item);
    }
}
