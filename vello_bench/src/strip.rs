// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use crate::harness::Registry;
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

pub fn register(registry: &mut Registry) {
    for item in get_data_items() {
        registry.non_simd(|registry| {
            register_render_strips(registry, item, Level::baseline(), "fallback");
        });
        let simd_level = Level::new();
        if !simd_level.is_fallback() {
            register_render_strips(registry, item, simd_level, "simd");
        }
    }

    register_render_rect(registry);

    registry.extended(register_render_strips_culled);
}

fn register_render_strips(
    registry: &mut Registry,
    item: &'static crate::data::DataItem,
    level: Level,
    suffix: &str,
) {
    let lines = item.lines();
    let tiles = item.sorted_tiles();
    registry.add(format!("render_strips/{}_{suffix}", item.name), move |b| {
        let mut strip_buf = vec![];
        let mut alpha_buf = vec![];

        b.iter(|| {
            strip_buf.clear();
            alpha_buf.clear();

            vello_common::strip::render(
                level,
                &tiles,
                &mut strip_buf,
                &mut alpha_buf,
                Fill::NonZero,
                None,
                &lines,
            );
            std::hint::black_box((&strip_buf, &alpha_buf));
        });
    });
}

fn register_render_strips_culled(registry: &mut Registry) {
    for item in get_data_items() {
        let simd_level = Level::new();
        if simd_level.is_fallback() {
            continue;
        }

        let shifted_lines = shift_lines_50_percent(&item.lines());

        let mut tiler = Tiles::new(simd_level, item.width, item.height);
        tiler.make_tiles_analytic_aa(simd_level, &shifted_lines, item.width, item.height);
        tiler.sort_tiles();

        registry.add(format!("render_strips_culled50/{}", item.name), move |b| {
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
                );
                std::hint::black_box((&strip_buf, &alpha_buf));
            });
        });
    }
}

fn register_render_rect(registry: &mut Registry) {
    for (name, size) in [("small", 20_u16), ("medium", 300), ("large", 1200)] {
        if name == "medium" {
            registry.extended(|registry| register_rect_size(registry, name, size));
        } else {
            register_rect_size(registry, name, size);
        }
    }
}

fn register_rect_size(registry: &mut Registry, name: &'static str, size: u16) {
    let rect = Rect::new(10.0, 10.0, f64::from(size) + 10.0, f64::from(size) + 10.0);
    let viewport_size = size + 20;
    let level = Level::new();

    registry.add(format!("render_rect/{name}"), move |b| {
        let mut generator = StripGenerator::new(viewport_size, viewport_size, level);
        let mut storage = StripStorage::default();

        b.iter(|| {
            storage.clear();
            generator.generate_filled_rect_fast(&rect, &mut storage, None);
            generator.reset(viewport_size, viewport_size);
            std::hint::black_box(&storage);
        });
    });

    registry.add(format!("render_rect/{name}_via_path"), move |b| {
        let mut generator = StripGenerator::new(viewport_size, viewport_size, level);
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
            generator.reset(viewport_size, viewport_size);
            std::hint::black_box(&storage);
        });
    });
}
