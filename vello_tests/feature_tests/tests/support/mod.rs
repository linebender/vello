// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello_common::pixmap::Pixmap;
use vello_common::probe::{self, ProbeFeature};
#[cfg(not(target_arch = "wasm32"))]
use vello_common::{
    filter_effects::Filter,
    kurbo::{Affine, BezPath, Rect},
    paint::PaintType,
    peniko::BlendMode,
    probe::ProbeRenderer,
};

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
pub(crate) mod webgl;
#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
pub(crate) mod wgpu;

pub(crate) fn elements() -> Vec<ProbeFeature> {
    let mut universal = Vec::new();
    let mut optional = Vec::new();
    for &feature in probe::ALL_PROBE_ELEMENTS {
        let enabled = match feature {
            ProbeFeature::BlurredRoundedRect => Some(cfg!(feature = "blurred_rounded_rect")),
            ProbeFeature::ImageBicubic => Some(cfg!(feature = "image_bicubic")),
            ProbeFeature::SweepGradient => Some(cfg!(feature = "gradient_sweep")),
            _ => None,
        };
        match enabled {
            None => universal.push(feature),
            Some(true) => optional.push(feature),
            Some(false) => (),
        }
    }
    universal.extend(optional);
    universal
}

pub(crate) fn check_snapshot(actual: Pixmap, backend: &str, threshold: u8, is_reference: bool) {
    let reference_name = env!("PROBE_SNAPSHOT_NAME");
    let run_name = format!("{reference_name}_{backend}");
    #[cfg(not(target_arch = "wasm32"))]
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    vello_test_support::Snapshot {
        reference_name,
        run_name: &run_name,
        threshold,
        diff_pixels: 0,
        is_reference,
        #[cfg(not(target_arch = "wasm32"))]
        snapshots_dir: &root.join("../snapshots"),
        #[cfg(not(target_arch = "wasm32"))]
        diffs_dir: &root.join("../diffs"),
        #[cfg(target_arch = "wasm32")]
        reference_png: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../snapshots/",
            env!("PROBE_SNAPSHOT_NAME"),
            ".png"
        )),
    }
    .check(&actual.into_png().unwrap());
}

#[cfg(not(target_arch = "wasm32"))]
struct CpuProbeContext<'a>(&'a mut vello_cpu::RenderContext);

#[cfg(not(target_arch = "wasm32"))]
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

    fn fill_blurred_rounded_rect(&mut self, rect: &Rect, radius: f32, std_dev: f32, invert: bool) {
        self.0
            .fill_blurred_rounded_rect(rect, radius, std_dev, invert);
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

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn render_cpu(elements: &[ProbeFeature]) -> Pixmap {
    use std::sync::Arc;
    use vello_common::{TargetInit, color::palette::css, paint::ImageSource};
    use vello_cpu::{
        Level, RasterizerSettings, RenderContext, RenderMode, RenderSettings, Resources,
    };

    let (width, height) = probe::canvas_size(elements);
    let mut ctx = RenderContext::new_with(
        width,
        height,
        RenderSettings {
            level: Level::baseline(),
            num_threads: 0,
        },
    );

    probe::draw_scene(
        &mut CpuProbeContext(&mut ctx),
        ImageSource::Pixmap(Arc::new(probe::probe_image_pixmap())),
        elements,
    );
    ctx.flush();

    let mut pixmap = Pixmap::new(width, height);

    ctx.render_with(
        &mut pixmap,
        &mut Resources::new(),
        RasterizerSettings {
            render_mode: RenderMode::OptimizeQuality,
            target_init: TargetInit::Clear(css::WHITE),
            ..Default::default()
        },
    );
    pixmap
}
