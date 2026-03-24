// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use criterion::{BenchmarkId, Criterion};

pub fn sort(c: &mut Criterion) {
    let mut g = c.benchmark_group("sort");
    g.sample_size(50);

    for item in get_data_items() {
        let unsorted = item.unsorted_tiles();
        g.bench_with_input(
            BenchmarkId::from_parameter(&item.name),
            &unsorted,
            |b, unsorted| {
                b.iter_batched(
                    || unsorted.clone(),
                    |mut tiles| tiles.sort_tiles(),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    g.finish();
}
