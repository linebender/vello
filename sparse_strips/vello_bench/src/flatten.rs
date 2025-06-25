// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use criterion::Criterion;
use vello_common::flatten;
use vello_common::kurbo::Stroke;
use vello_cpu::kurbo::Affine;

pub fn flatten(c: &mut Criterion) {
    let mut g = c.benchmark_group("flatten");

    macro_rules! flatten_single {
        ($item:expr) => {
            let expanded_strokes = $item.expanded_strokes();

            g.bench_function($item.name.clone(), |b| {
                b.iter(|| {
                    let mut line_buf: Vec<flatten::Line> = vec![];
                    let mut temp_buf: Vec<flatten::Line> = vec![];

                    for path in &$item.fills {
                        flatten::fill(&path.path, path.transform, &mut temp_buf);
                        line_buf.extend(&temp_buf);
                    }

                    for stroke in &expanded_strokes {
                        flatten::fill(stroke, Affine::IDENTITY, &mut temp_buf);
                        line_buf.extend(&temp_buf);
                    }

                    std::hint::black_box(&line_buf);
                })
            });
        };
    }

    for item in get_data_items() {
        flatten_single!(item);
    }
}

pub fn strokes(c: &mut Criterion) {
    let mut g = c.benchmark_group("strokes");

    macro_rules! expand_single {
        ($item:expr) => {
            g.bench_function($item.name.clone(), |b| {
                b.iter(|| {
                    let mut paths = vec![];

                    for path in &$item.strokes {
                        let stroke = Stroke {
                            width: path.stroke_width as f64,
                            ..Default::default()
                        };
                        paths.push(flatten::expand_stroke(path.path.iter(), &stroke, 0.25));
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
