// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use crate::registry::Registry;
use criterion::Criterion;
use vello_common::flatten;
use vello_common::flatten::FlattenCtx;
use vello_common::geometry::RectU16;
use vello_common::kurbo::Stroke;
use vello_common::kurbo::StrokeCtx;
use vello_cpu::Level;
use vello_cpu::kurbo::Affine;

/// Buffers reused across iterations to better simulate real-world use.
#[derive(Default)]
struct FlattenBuffers {
    lines: Vec<flatten::Line>,
    temp: Vec<flatten::Line>,
    ctx: FlattenCtx,
}

pub fn flatten(r: &mut impl Registry) {
    for item in get_data_items() {
        let expanded_strokes = item.expanded_strokes();
        let bounds = RectU16::new(0, 0, item.width, item.height);

        r.bench(
            "flatten",
            &item.name,
            FlattenBuffers::default,
            move |buffers| {
                let FlattenBuffers { lines, temp, ctx } = buffers;
                lines.clear();

                for path in &item.fills {
                    flatten::fill(Level::new(), &path.path, path.transform, temp, ctx, bounds);
                    lines.extend(&*temp);
                }

                for stroke in &expanded_strokes {
                    flatten::fill(Level::new(), stroke, Affine::IDENTITY, temp, ctx, bounds);
                    lines.extend(&*temp);
                }

                std::hint::black_box(&*lines);
            },
        );
    }
}

pub fn strokes(c: &mut Criterion) {
    let mut g = c.benchmark_group("strokes");

    macro_rules! expand_single {
        ($item:expr) => {
            g.bench_function($item.name.clone(), |b| {
                let mut stroke_ctx = StrokeCtx::default();

                b.iter(|| {
                    let mut paths = vec![];

                    for path in &$item.strokes {
                        let stroke = Stroke {
                            width: path.stroke_width as f64,
                            ..Default::default()
                        };
                        flatten::expand_stroke(path.path.iter(), &stroke, 0.25, &mut stroke_ctx);
                        paths.push(stroke_ctx.output().clone());
                    }

                    std::hint::black_box(&paths);
                })
            });
        };
    }

    for item in get_data_items() {
        expand_single!(item);
    }
}
