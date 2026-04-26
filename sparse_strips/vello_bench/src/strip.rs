// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use criterion::Criterion;
use vello_common::fearless_simd::Level;
use vello_common::flatten::Line;
use vello_common::kurbo::{Affine, Rect, Shape};
use vello_common::peniko::Fill;
use vello_common::strip_generator::{StripGenerator, StripStorage};
use vello_common::tile::Tiles;

pub fn shift_lines_50_percent(lines: &[Line]) -> Vec<Line> {
    if lines.is_empty() {
        return vec![];
    }

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    for line in lines {
        min_x = min_x.min(line.p0.x).min(line.p1.x);
        max_x = max_x.max(line.p0.x).max(line.p1.x);
    }

    let shift_amount = (min_x + max_x) / 2.0;

    let mut shifted = lines.to_vec();
    for line in &mut shifted {
        line.p0.x -= shift_amount;
        line.p1.x -= shift_amount;
    }
    shifted
}

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
                        false,
                    );
                    std::hint::black_box((&strip_buf, &alpha_buf));
                })
            });
        };
    }

    for item in get_data_items() {
        // Commenting this out by default since SIMD is what we care about most.
        // strip_single!(item, Level::baseline(), "fallback");
        let simd_level = Level::new();
        if !simd_level.is_fallback() {
            strip_single!(item, simd_level, "simd");
        }
    }
    g.finish();
}

pub fn render_strips_cull(c: &mut Criterion) {
    let mut g_cull = c.benchmark_group("render_strips_culled50");
    g_cull.sample_size(50);

    for item in get_data_items() {
        let simd_level = Level::new();
        if simd_level.is_fallback() {
            continue;
        }

        let shifted_lines = shift_lines_50_percent(&item.lines());

        let mut tiler = Tiles::new(simd_level, item.height);
        let is_culled = tiler.make_tiles_analytic_aa::<true>(
            simd_level,
            &shifted_lines,
            item.width,
            item.height,
        );
        tiler.sort_tiles();

        g_cull.bench_function(item.name.clone().to_string(), |b| {
            let mut strip_buf = vec![];
            let mut alpha_buf = vec![];

            b.iter(|| {
                strip_buf.clear();
                alpha_buf.clear();

                vello_common::strip::render(
                    simd_level,
                    &tiler,
                    &mut strip_buf,
                    &mut alpha_buf,
                    Fill::NonZero,
                    None,
                    &shifted_lines,
                    is_culled,
                );
                std::hint::black_box((&strip_buf, &alpha_buf));
            });
        });
    }
    g_cull.finish();
}

pub fn render_rect(c: &mut Criterion) {
    let mut g = c.benchmark_group("render_rect");
    g.sample_size(50);

    let rect = Rect::new(10.0, 10.0, 24.0, 24.0);
    let width = 100;
    let height = 100;
    let level = Level::new();

    // Benchmark: generate_filled_path (path-based approach)
    g.bench_function("14x14_via_path", |b| {
        let mut generator = StripGenerator::new(width, height, level);
        let mut storage = StripStorage::default();

        b.iter(|| {
            storage.clear();
            generator.generate_filled_path(
                rect.to_path(0.1),
                Fill::NonZero,
                Affine::IDENTITY,
                None,
                &mut storage,
                None,
            );
            generator.reset();
            std::hint::black_box(&storage);
        });
    });

    // Benchmark: generate_rect_strips_with_clip (optimized rect approach)
    g.bench_function("14x14_via_rect", |b| {
        let mut generator = StripGenerator::new(width, height, level);
        let mut storage = StripStorage::default();

        b.iter(|| {
            storage.clear();
            generator.generate_filled_rect_fast(&rect, &mut storage, None);
            generator.reset();
            std::hint::black_box(&storage);
        });
    });

    g.finish();
}
