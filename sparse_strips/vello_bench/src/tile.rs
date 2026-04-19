// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use criterion::{BenchmarkId, Criterion};
use vello_common::flatten::Line;
use vello_common::tile::Tiles;
use vello_cpu::Level;

fn shift_lines_50_percent(lines: &[Line]) -> Vec<Line> {
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

fn run_tile_benchmark<const SHIFT: bool, F>(c: &mut Criterion, group_name: &str, mut op: F)
where
    F: FnMut(&mut Tiles, &[Line], u16, u16),
{
    let mut g = c.benchmark_group(group_name);
    g.sample_size(50);

    for item in get_data_items() {
        let lines = if SHIFT {
            shift_lines_50_percent(&item.lines())
        } else {
            item.lines()
        };

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
    let max_rows = (u16::MAX / 4) as usize + 1;
    let mut partial_windings = vec![[0.0; 4]; max_rows];
    let mut coarse_windings = vec![0; max_rows];
    let mut active_rows = vec![0; (max_rows >> 5) + 1];

    run_tile_benchmark::<false, _>(c, "tile_aaa", |tiler, lines, w, h| {
        tiler.make_tiles_analytic_aa::<false>(
            Level::new(),
            lines,
            w,
            h,
            &mut partial_windings,
            &mut coarse_windings,
            &mut active_rows,
        );
    });

    run_tile_benchmark::<false, _>(c, "tile_msaa", |tiler, lines, w, h| {
        tiler.make_tiles_msaa(lines, w, h);
    });

    run_tile_benchmark::<true, _>(c, "tile_aaa_shift50", |tiler, lines, w, h| {
        tiler.make_tiles_analytic_aa::<false>(
            Level::new(),
            lines,
            w,
            h,
            &mut partial_windings,
            &mut coarse_windings,
            &mut active_rows,
        );
    });

    run_tile_benchmark::<true, _>(c, "tile_aaa_shift50_cull", |tiler, lines, w, h| {
        tiler.make_tiles_analytic_aa::<true>(
            Level::new(),
            lines,
            w,
            h,
            &mut partial_windings,
            &mut coarse_windings,
            &mut active_rows,
        );
    });
}