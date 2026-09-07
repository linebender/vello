// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::data::get_data_items;
use crate::harness::Registry;
use vello_common::flatten;
use vello_common::flatten::FlattenCtx;
use vello_common::geometry::RectU16;
use vello_common::kurbo::Stroke;
use vello_common::kurbo::StrokeCtx;
use vello_cpu::Level;
use vello_cpu::kurbo::Affine;

pub fn register(registry: &mut Registry) {
    for item in get_data_items() {
        let expanded_strokes = item.expanded_strokes();
        registry.add(format!("flatten/{}", item.name), move |b| {
            // Reuse allocations to better simulate real-world use.
            let mut line_buf: Vec<flatten::Line> = vec![];
            let mut temp_buf: Vec<flatten::Line> = vec![];
            let mut flatten_ctx = FlattenCtx::default();

            b.iter(|| {
                line_buf.clear();

                for path in &item.fills {
                    flatten::fill(
                        Level::new(),
                        &path.path,
                        path.transform,
                        &mut temp_buf,
                        &mut flatten_ctx,
                        RectU16::new(0, 0, item.width, item.height),
                    );
                    line_buf.extend(&temp_buf);
                }

                for stroke in &expanded_strokes {
                    flatten::fill(
                        Level::new(),
                        stroke,
                        Affine::IDENTITY,
                        &mut temp_buf,
                        &mut flatten_ctx,
                        RectU16::new(0, 0, item.width, item.height),
                    );
                    line_buf.extend(&temp_buf);
                }

                std::hint::black_box(&line_buf);
            });
        });
    }

    for item in get_data_items() {
        registry.add(format!("strokes/{}", item.name), move |b| {
            let mut stroke_ctx = StrokeCtx::default();

            b.iter(|| {
                let mut paths = vec![];

                for path in &item.strokes {
                    let stroke = Stroke {
                        width: path.stroke_width as f64,
                        ..Default::default()
                    };
                    flatten::expand_stroke(path.path.iter(), &stroke, 0.25, &mut stroke_ctx);
                    paths.push(stroke_ctx.output().clone());
                }

                std::hint::black_box(&paths);
            });
        });
    }
}
