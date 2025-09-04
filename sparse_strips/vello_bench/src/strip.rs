// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use criterion::Criterion;
use vello_common::fearless_simd::Level;
use vello_common::peniko::Fill;

pub fn render_strips(c: &mut Criterion) {
    let mut g = c.benchmark_group("render_strips");
    g.sample_size(50);

    macro_rules! strip_single {
        ($item:expr, $level:expr, $suffix:expr) => {
            let lines = $item.lines();
            let tiles = $item.sorted_tiles();

            g.bench_function(format!("{}_{}", $item.name.clone(), $suffix), |b| {
                let mut strip_buf = vec![];
                let mut alpha_buf = vec![];

                b.iter(|| {
                    strip_buf.clear();
                    alpha_buf.clear();

                    vello_common::strip::render(
                        $level,
                        &tiles,
                        &mut strip_buf,
                        &mut alpha_buf,
                        Fill::NonZero,
                        None,
                        &lines,
                    );
                    std::hint::black_box((&strip_buf, &alpha_buf));
                })
            });
        };
    }

    for item in get_data_items() {
        // Commenting this out by default since SIMD is what we care about most.
        // strip_single!(item, Level::fallback(), "fallback");
        let simd_level = Level::new();
        if !matches!(simd_level, Level::Fallback(_)) {
            strip_single!(item, simd_level, "simd");
        }
    }
}
