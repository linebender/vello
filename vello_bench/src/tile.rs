// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use crate::harness::Registry;
use crate::strip::shift_lines_50_percent;
use vello_common::flatten::Line;
use vello_common::tile::Tiles;
use vello_cpu::Level;

fn register_tile_benchmark<const SHIFT: bool>(
    registry: &mut Registry,
    group_name: &str,
    op: fn(&mut Tiles, &[Line], u16, u16),
) {
    for item in get_data_items() {
        let lines = if SHIFT {
            shift_lines_50_percent(&item.lines())
        } else {
            item.lines()
        };

        registry.add(format!("{group_name}/{}", item.name), move |b| {
            b.iter(|| {
                let mut tiler = Tiles::new(Level::new(), item.width, item.height);
                op(&mut tiler, &lines, item.width, item.height);
            });
        });
    }
}

pub fn register(registry: &mut Registry) {
    register_tile_benchmark::<false>(registry, "tile_aaa", |tiler, lines, w, h| {
        tiler.make_tiles_analytic_aa(Level::new(), lines, w, h);
    });

    registry.extended(|registry| {
        register_tile_benchmark::<false>(registry, "tile_msaa", |tiler, lines, w, h| {
            tiler.make_tiles_msaa(lines, w, h);
        });

        register_tile_benchmark::<true>(registry, "tile_aaa_shift50", |tiler, lines, w, h| {
            tiler.make_tiles_analytic_aa(Level::new(), lines, w, h);
        });
    });
}
