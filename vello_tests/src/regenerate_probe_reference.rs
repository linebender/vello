// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Regenerate the probe reference assets in `vello_common/assets`.

use std::{path::PathBuf, sync::Arc};

#[cfg(not(target_arch = "wasm32"))]
use oxipng::Options;
use vello_common::{
    TargetInit,
    color::palette::css,
    filter_effects::Filter,
    kurbo::{Affine, BezPath, Rect},
    paint::{ImageSource, PaintType},
    peniko::{BlendMode, ImageAlphaType},
    pixmap::Pixmap,
    probe::{self, ProbeRenderer},
};
use vello_cpu::{Level, RasterizerSettings, RenderContext, RenderMode, RenderSettings, Resources};

fn reference_path(element: probe::ProbeFeature, extension: &str) -> PathBuf {
    let name = match element {
        probe::ProbeFeature::SolidRect => "probe_solid_rect",
        probe::ProbeFeature::AlphaBlending => "probe_alpha_blending",
        probe::ProbeFeature::Gradient => "probe_gradient",
        probe::ProbeFeature::ImageNearest => "probe_image_nearest",
        probe::ProbeFeature::Filter => "probe_filter",
        probe::ProbeFeature::ImageBilinear => "probe_image_bilinear",
        probe::ProbeFeature::OpacityLayer => "probe_opacity_layer",
        probe::ProbeFeature::Blending => "probe_blending",
        probe::ProbeFeature::Transformed => "probe_transformed",
        probe::ProbeFeature::DepthBuffer => "probe_depth_buffer",
    };
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../vello_common/assets")
        .join(format!("{name}.{extension}"))
}

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

fn render_probe_pixmap(elements: &[probe::ProbeFeature]) -> Pixmap {
    let (width, height) = probe::canvas_size(elements);
    let settings = RenderSettings {
        level: Level::fallback(),
        num_threads: 0,
    };
    let mut ctx = RenderContext::new_with(width, height, settings);

    probe::draw_scene(
        &mut CpuProbeContext(&mut ctx),
        ImageSource::Pixmap(Arc::new(probe::probe_image_pixmap())),
        elements,
    );
    ctx.flush();

    let mut resources = Resources::new();
    let mut pixmap = Pixmap::new(width, height);
    ctx.render_with(
        &mut pixmap,
        &mut resources,
        RasterizerSettings {
            render_mode: RenderMode::OptimizeQuality,
            target_init: TargetInit::Clear(css::WHITE),
            ..Default::default()
        },
    );
    pixmap
}

fn encode_png(pixmap: Pixmap) -> Vec<u8> {
    let png = pixmap.into_png().unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    let png = oxipng::optimize_from_memory(&png, &Options::max_compression()).unwrap();
    png
}

fn main() {
    for element in probe::ALL_PROBE_ELEMENTS {
        let pixmap = render_probe_pixmap(&[*element]);
        let rgba = pixmap.clone().take_rgba8(ImageAlphaType::Alpha);
        std::fs::write(reference_path(*element, "rgba"), rgba).unwrap();
        std::fs::write(reference_path(*element, "png"), encode_png(pixmap)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_elements_match_reference(elements: &[probe::ProbeFeature]) {
        assert!(
            probe::Probe::<()>::from_actual(render_probe_pixmap(elements), elements).is_success(),
            "rendered probe elements did not match their reference cells: {elements:?}"
        );
    }

    #[test]
    fn probe_reference_is_up_to_date() {
        for element in probe::ALL_PROBE_ELEMENTS {
            let pixmap = render_probe_pixmap(&[*element]);
            assert_eq!(
                std::fs::read(reference_path(*element, "rgba")).unwrap(),
                pixmap.clone().take_rgba8(ImageAlphaType::Alpha),
                "RGBA reference for {element:?} is out of date; run `cargo run -p vello_tests --bin regenerate_probe_reference`",
            );
            assert_eq!(
                std::fs::read(reference_path(*element, "png")).unwrap(),
                encode_png(pixmap),
                "PNG reference for {element:?} is out of date; run `cargo run -p vello_tests --bin regenerate_probe_reference`",
            );
        }
    }

    #[test]
    fn individual_probe_elements_match_reference() {
        for element in probe::ALL_PROBE_ELEMENTS {
            assert_elements_match_reference(&[*element]);
        }
    }

    #[test]
    fn probe_failure_reports_statistics_per_cell() {
        let elements = [
            probe::ProbeFeature::SolidRect,
            probe::ProbeFeature::Gradient,
        ];
        let mut actual = render_probe_pixmap(&elements);
        actual.set_pixel(0, 0, css::BLACK.premultiply().to_rgba8());

        let probe::Probe::Error(result) = probe::Probe::<()>::from_actual(actual, &elements) else {
            panic!("modified probe unexpectedly matched its references");
        };
        let statistics = &result.statistics;
        assert_eq!(statistics.len(), 2);
        assert_eq!(statistics[0].feature, probe::ProbeFeature::SolidRect);
        assert_eq!(statistics[0].different_pixel_count, 1);
        assert_eq!(statistics[1].feature, probe::ProbeFeature::Gradient);
        assert_eq!(statistics[1].different_pixel_count, 0);
    }

    #[test]
    fn reversed_probe_elements_match_reference() {
        let reversed = probe::ALL_PROBE_ELEMENTS
            .iter()
            .rev()
            .copied()
            .collect::<Vec<_>>();

        assert_elements_match_reference(&reversed);
    }

    #[test]
    fn full_probe_row_matches_reference() {
        assert_elements_match_reference(&[
            probe::ProbeFeature::SolidRect,
            probe::ProbeFeature::Gradient,
            probe::ProbeFeature::DepthBuffer,
        ]);
    }

    #[test]
    fn partial_second_probe_row_matches_reference() {
        assert_elements_match_reference(&[
            probe::ProbeFeature::ImageBilinear,
            probe::ProbeFeature::DepthBuffer,
            probe::ProbeFeature::SolidRect,
            probe::ProbeFeature::Filter,
        ]);
    }
}
