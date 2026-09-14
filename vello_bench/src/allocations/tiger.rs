// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::hint::black_box;

use super::{AllocationBenchmark, AllocationLimits, AllocationStats, measure};
use crate::data::{DataItem, get_data_items};
use vello_common::fearless_simd::Level;
use vello_common::kurbo::Stroke;
use vello_common::pixmap::Pixmap;
use vello_cpu::{RenderContext, RenderSettings, Resources};

pub const HOT_SCENE_BUILD: AllocationBenchmark = AllocationBenchmark::new(
    "tiger/hot_scene_build",
    measure_hot_scene_build,
    AllocationLimits::ZERO,
);

pub const HOT_RASTERIZE: AllocationBenchmark = AllocationBenchmark::new(
    "tiger/hot_rasterize",
    measure_hot_rasterize,
    AllocationLimits::new(4, 0, 7_232, 7_232, 0),
);

pub const HOT_FULL_FRAME: AllocationBenchmark = AllocationBenchmark::new(
    "tiger/hot_full_frame",
    measure_hot_full_frame,
    AllocationLimits::new(4, 0, 7_232, 7_232, 0),
);

fn measure_hot_scene_build(measured_frames: usize) -> AllocationStats {
    let item = tiger();
    let mut context = new_context(item);

    populate_scene(&mut context, item);
    context.flush();

    let (_, stats) = measure(|| {
        for _ in 0..measured_frames {
            context.reset();
            populate_scene(&mut context, item);
            context.flush();
            black_box(&context);
        }
    });
    stats
}

fn measure_hot_rasterize(measured_frames: usize) -> AllocationStats {
    let item = tiger();
    let mut context = new_context(item);
    populate_scene(&mut context, item);
    context.flush();

    let mut resources = Resources::new();
    let mut target = Pixmap::new(item.width, item.height);
    context.render(&mut target, &mut resources);

    let (_, stats) = measure(|| {
        for _ in 0..measured_frames {
            context.render(&mut target, &mut resources);
            black_box(&target);
        }
    });
    stats
}

fn measure_hot_full_frame(measured_frames: usize) -> AllocationStats {
    let item = tiger();
    let mut context = new_context(item);
    let mut resources = Resources::new();
    let mut target = Pixmap::new(item.width, item.height);

    render_frame(&mut context, &mut resources, &mut target, item);

    let (_, stats) = measure(|| {
        for _ in 0..measured_frames {
            render_frame(&mut context, &mut resources, &mut target, item);
            black_box(&target);
        }
    });
    stats
}

fn tiger() -> &'static DataItem {
    get_data_items()
        .first()
        .expect("the benchmark data always contains the Ghostscript tiger")
}

fn new_context(item: &DataItem) -> RenderContext {
    RenderContext::new_with(
        item.width,
        item.height,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    )
}

fn render_frame(
    context: &mut RenderContext,
    resources: &mut Resources,
    target: &mut Pixmap,
    item: &DataItem,
) {
    context.reset();
    populate_scene(context, item);
    context.flush();
    context.render(target, resources);
}

fn populate_scene(context: &mut RenderContext, item: &DataItem) {
    for fill in &item.fills {
        context.set_transform(fill.transform);
        context.fill_path(&fill.path);
    }

    for stroke in &item.strokes {
        context.set_transform(stroke.transform);
        context.set_stroke(Stroke::new(f64::from(stroke.stroke_width)));
        context.stroke_path(&stroke.path);
    }
}
