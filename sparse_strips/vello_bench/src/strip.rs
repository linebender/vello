// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::{COAT_OF_ARMS, GHOSTSCRIPT_TIGER, PARIS_30K};
use crate::read::PathContainer;
use criterion::Criterion;
use vello_common::peniko::Fill;
use vello_common::strip;

pub fn render_strips(c: &mut Criterion) {
    let mut g = c.benchmark_group("render_strips");
    g.sample_size(50);

    macro_rules! strip_single {
        ($item:expr) => {
            let container = PathContainer::from_data_file(&$item);

            let lines = container.lines();
            let tiles = container.sorted_tiles();

            g.bench_function($item.name, |b| {
                b.iter(|| {
                    let mut strip_buf = vec![];
                    let mut alpha_buf = vec![];

                    strip::render(
                        &tiles,
                        &mut strip_buf,
                        &mut alpha_buf,
                        Fill::NonZero,
                        &lines,
                    );
                })
            });
        };
    }

    strip_single!(GHOSTSCRIPT_TIGER);
    strip_single!(PARIS_30K);
    strip_single!(COAT_OF_ARMS);
}
