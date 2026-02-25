// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmarks for wide tile command generation.

use crate::data::get_data_items;
use criterion::Criterion;

use vello_common::coarse::{MODE_CPU, Wide};
use vello_common::color::palette::css::PURPLE;
use vello_common::paint::{Paint, PremulColor};
use vello_common::peniko::BlendMode;
use vello_common::render_graph::RenderGraph;
use vello_common::strip::Strip;

/// Benchmark wide tile command generation without layers at content size.
pub fn coarse(c: &mut Criterion) {
    let mut g = c.benchmark_group("coarse");
    g.sample_size(50);

    for item in get_data_items() {
        let (_alpha_buf, strip_buf) = item.strips();

        g.bench_function(&item.name, |b| {
            let mut wide = Wide::<MODE_CPU>::new(item.width, item.height);

            b.iter(|| {
                wide.reset();
                wide.generate(
                    &strip_buf,
                    Paint::Solid(PremulColor::from_alpha_color(PURPLE)),
                    BlendMode::default(),
                    0,
                    None,
                    &[],
                );
                std::hint::black_box(&wide);
            });
        });
    }

    g.finish();
}

/// Benchmark wide tile command generation with 3 nested layers at content size.
pub fn coarse_with_layer(c: &mut Criterion) {
    let mut g = c.benchmark_group("coarse_with_layer");
    g.sample_size(50);

    for item in get_data_items() {
        let (_alpha_buf, strip_buf) = item.strips();

        g.bench_function(&item.name, |b| {
            let mut wide = Wide::<MODE_CPU>::new(item.width, item.height);
            let mut render_graph = RenderGraph::new();

            b.iter(|| {
                wide.reset();
                render_graph.clear();

                // Add root node
                let wtile_bbox = vello_common::coarse::WideTilesBbox::new([
                    0,
                    0,
                    wide.width_tiles(),
                    wide.height_tiles(),
                ]);
                render_graph.add_node(vello_common::render_graph::RenderNodeKind::RootLayer {
                    layer_id: 0,
                    wtile_bbox,
                });

                // Push 3 nested layers
                for layer_id in 1..=3 {
                    wide.push_layer(
                        layer_id,
                        None::<&[Strip]>,
                        BlendMode::default(),
                        None,
                        1.0,
                        None,
                        vello_common::kurbo::Affine::IDENTITY,
                        &mut render_graph,
                        0,
                    );
                }

                // Generate commands inside the innermost layer
                wide.generate(
                    &strip_buf,
                    Paint::Solid(PremulColor::from_alpha_color(PURPLE)),
                    BlendMode::default(),
                    0,
                    None,
                    &[],
                );

                // Pop all layers
                for _ in 1..=3 {
                    wide.pop_layer(&mut render_graph);
                }

                std::hint::black_box(&wide);
            });
        });
    }

    g.finish();
}

/// Benchmark wide tile command generation with 3 layers in a 4K viewport.
///
/// This times coarse rasterization in a 4K viewport (3840x2160) with content that may only cover a
/// portion of it. Smaller SVGs like Ghostscript Tiger (~200x200 natural size) may cover very few
/// tiles.
pub fn coarse_with_layer_large_viewport(c: &mut Criterion) {
    let mut g = c.benchmark_group("coarse_with_layer_4k");
    g.sample_size(50);

    for item in get_data_items() {
        let (_alpha_buf, strip_buf) = item.strips();

        // Use a 4K viewport to amplify the benefit of lazy pushing
        const WIDTH_4K: u16 = 3840;
        const HEIGHT_4K: u16 = 2160;

        g.bench_function(&item.name, |b| {
            let mut wide = Wide::<MODE_CPU>::new(WIDTH_4K, HEIGHT_4K);
            let mut render_graph = RenderGraph::new();

            b.iter(|| {
                wide.reset();
                render_graph.clear();

                let wtile_bbox = vello_common::coarse::WideTilesBbox::new([
                    0,
                    0,
                    wide.width_tiles(),
                    wide.height_tiles(),
                ]);
                render_graph.add_node(vello_common::render_graph::RenderNodeKind::RootLayer {
                    layer_id: 0,
                    wtile_bbox,
                });

                // Push 3 nested layers
                for layer_id in 1..=3 {
                    wide.push_layer(
                        layer_id,
                        None::<&[Strip]>,
                        BlendMode::default(),
                        None,
                        1.0,
                        None,
                        vello_common::kurbo::Affine::IDENTITY,
                        &mut render_graph,
                        0,
                    );
                }

                // Content is drawn at original size. For smaller SVGs, many wide tiles won't
                // receive any content.
                wide.generate(
                    &strip_buf,
                    Paint::Solid(PremulColor::from_alpha_color(PURPLE)),
                    BlendMode::default(),
                    0,
                    None,
                    &[],
                );

                // Pop all layers
                for _ in 1..=3 {
                    wide.pop_layer(&mut render_graph);
                }

                std::hint::black_box(&wide);
            });
        });
    }

    g.finish();
}
