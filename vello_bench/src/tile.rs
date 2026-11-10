// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use crate::registry::Registry;
use crate::strip::shift_lines_50_percent;
use vello_common::flatten::Line;
use vello_common::tile::Tiles;
use vello_cpu::Level;

fn run_tile_benchmark<const SHIFT: bool, F>(r: &mut Registry, group_name: &str, op: F)
where
    F: Fn(&mut Tiles, &[Line], u16, u16) + Copy + 'static,
{
    for item in get_data_items() {
        let lines = if SHIFT {
            shift_lines_50_percent(&item.lines())
        } else {
            item.lines()
        };

        r.bench(format!("{group_name}/{}", item.name), move |b| {
            b.iter(|| {
                let mut tiler = Tiles::new(Level::new(), item.width, item.height);
                op(&mut tiler, &lines, item.width, item.height);
            });
        });
    }
}

pub fn tile(r: &mut Registry) {
    run_tile_benchmark::<false, _>(r, "tile_aaa", |tiler, lines, w, h| {
        tiler.make_tiles_analytic_aa(Level::new(), lines, w, h);
    });

    if crate::EXTENDED {
        run_tile_benchmark::<false, _>(r, "tile_msaa", |tiler, lines, w, h| {
            tiler.make_tiles_msaa(lines, w, h);
        });

        run_tile_benchmark::<true, _>(r, "tile_aaa_shift50", |tiler, lines, w, h| {
            tiler.make_tiles_analytic_aa(Level::new(), lines, w, h);
        });
    }
}
