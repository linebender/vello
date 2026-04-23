// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Regenerate the probe reference assets in `vello_common/assets`.

use bytemuck::cast_slice;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use oxipng::Options;
use vello_common::{
    filter_effects::Filter,
    kurbo::{Affine, BezPath, Rect},
    paint::{ImageSource, PaintType},
    peniko::BlendMode,
    probe::{self, ProbeRenderer},
};
use vello_cpu::{Level, RenderContext, RenderMode, RenderSettings, Resources};

struct CpuProbeContext<'a>(&'a mut RenderContext);

impl ProbeRenderer for CpuProbeContext<'_> {
    fn set_transform(&mut self, transform: Affine) {
        self.0.set_transform(transform);
    }

    fn set_paint(&mut self, paint: PaintType) {
        self.0.set_paint(paint);
    }

    fn fill_path(&mut self, path: &BezPath) {
        self.0.fill_path(path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.0.fill_rect(rect);
    }

    fn push_layer(&mut self, blend_mode: Option<BlendMode>, opacity: Option<f32>) {
        self.0.push_layer(None, blend_mode, opacity, None, None);
    }

    fn push_filter_layer(&mut self, filter: Filter) {
        self.0.push_filter_layer(filter);
    }

    fn pop_layer(&mut self) {
        self.0.pop_layer();
    }

    fn set_paint_transform(&mut self, paint_transform: Affine) {
        self.0.set_paint_transform(paint_transform);
    }

    fn reset_paint_transform(&mut self) {
        self.0.reset_paint_transform();
    }
}

fn main() {
    // The PNG output is only for easier inspection. The RGBA binary blob is what will actually
    // be included in the probe result.
    let png_output = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_common/assets/probe.png");
    let rgba_output =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_common/assets/probe.rgba");

    let (width, height) = probe::canvas_size();
    let settings = RenderSettings {
        level: Level::fallback(),
        num_threads: 0,
        render_mode: RenderMode::OptimizeQuality,
    };
    let mut ctx = RenderContext::new_with(width, height, settings);

    probe::draw_scene(
        &mut CpuProbeContext(&mut ctx),
        ImageSource::Pixmap(Arc::new(probe::probe_image_pixmap())),
    );
    ctx.flush();

    let mut resources = Resources::new();
    let mut pixmap = vello_common::pixmap::Pixmap::new(width, height);
    ctx.render_to_pixmap(&mut resources, &mut pixmap);

    let rgba = pixmap.clone().take_unpremultiplied();
    std::fs::write(rgba_output, cast_slice(&rgba)).unwrap();
    let png = pixmap.into_png().unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    let png = oxipng::optimize_from_memory(&png, &Options::max_compression()).unwrap();
    std::fs::write(png_output, png).unwrap();
}
