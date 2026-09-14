// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Regenerate the probe reference assets in `vello_common/assets`.

use std::{
    path::PathBuf,
    sync::{Arc, LazyLock},
};

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

static PROBE_PNG_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_common/assets/probe.png")
});
static PROBE_RGBA_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_common/assets/probe.rgba")
});
static THREE_ELEMENTS_PNG_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_common/assets/probe_3_elements.png")
});
static FOUR_ELEMENTS_PNG_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vello_common/assets/probe_4_elements.png")
});
static WITHOUT_FILTER_PNG_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../vello_common/assets/probe_without_filter.png")
});

const THREE_ELEMENTS: &[probe::ProbeFeature] = &[
    probe::ProbeFeature::SolidRect,
    probe::ProbeFeature::Gradient,
    probe::ProbeFeature::DepthBuffer,
];
const FOUR_ELEMENTS: &[probe::ProbeFeature] = &[
    probe::ProbeFeature::ImageBilinear,
    probe::ProbeFeature::DepthBuffer,
    probe::ProbeFeature::SolidRect,
    probe::ProbeFeature::Filter,
];
const WITHOUT_FILTER: &[probe::ProbeFeature] = &[
    probe::ProbeFeature::DepthBuffer,
    probe::ProbeFeature::Transformed,
    probe::ProbeFeature::Blending,
    probe::ProbeFeature::OpacityLayer,
    probe::ProbeFeature::ImageBilinear,
    probe::ProbeFeature::ImageNearest,
    probe::ProbeFeature::Gradient,
    probe::ProbeFeature::AlphaBlending,
    probe::ProbeFeature::SolidRect,
];

struct ProbeReferenceData {
    png: Vec<u8>,
    rgba: Vec<u8>,
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

fn build_probe_reference_data() -> ProbeReferenceData {
    let pixmap = render_probe_pixmap(probe::ALL_PROBE_ELEMENTS);
    let rgba = pixmap.clone().take_rgba8(ImageAlphaType::Alpha);
    let png = encode_png(pixmap);
    ProbeReferenceData { png, rgba }
}

fn build_selected_reference_png(elements: &[probe::ProbeFeature]) -> Vec<u8> {
    encode_png(render_probe_pixmap(elements))
}

fn encode_png(pixmap: Pixmap) -> Vec<u8> {
    let png = pixmap.into_png().unwrap();
    #[cfg(not(target_arch = "wasm32"))]
    let png = oxipng::optimize_from_memory(&png, &Options::max_compression()).unwrap();
    png
}

fn main() {
    let reference = build_probe_reference_data();
    std::fs::write(&*PROBE_RGBA_PATH, reference.rgba).unwrap();
    std::fs::write(&*PROBE_PNG_PATH, reference.png).unwrap();
    std::fs::write(
        &*THREE_ELEMENTS_PNG_PATH,
        build_selected_reference_png(THREE_ELEMENTS),
    )
    .unwrap();
    std::fs::write(
        &*FOUR_ELEMENTS_PNG_PATH,
        build_selected_reference_png(FOUR_ELEMENTS),
    )
    .unwrap();
    std::fs::write(
        &*WITHOUT_FILTER_PNG_PATH,
        build_selected_reference_png(WITHOUT_FILTER),
    )
    .unwrap();
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

    fn assert_png_is_up_to_date(path: &std::path::Path, elements: &[probe::ProbeFeature]) {
        assert_eq!(
            std::fs::read(path).unwrap(),
            build_selected_reference_png(elements),
            "{} is out of date; run `cargo run -p vello_tests --bin regenerate_probe_reference`",
            path.display()
        );
    }

    #[test]
    fn probe_reference_is_up_to_date() {
        let reference = build_probe_reference_data();
        let committed_rgba = std::fs::read(&*PROBE_RGBA_PATH).unwrap();
        let committed_png = std::fs::read(&*PROBE_PNG_PATH).unwrap();

        assert_eq!(
            committed_rgba, reference.rgba,
            "probe.rgba is out of date; run `cargo run -p vello_tests --bin regenerate_probe_reference`",
        );
        assert_eq!(
            committed_png, reference.png,
            "probe.png is out of date; run `cargo run -p vello_tests --bin regenerate_probe_reference`",
        );
    }

    #[test]
    fn individual_probe_elements_match_reference() {
        for element in probe::ALL_PROBE_ELEMENTS {
            assert_elements_match_reference(&[*element]);
        }
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
        assert_elements_match_reference(THREE_ELEMENTS);
        assert_png_is_up_to_date(&THREE_ELEMENTS_PNG_PATH, THREE_ELEMENTS);
    }

    #[test]
    fn partial_second_probe_row_matches_reference() {
        assert_elements_match_reference(FOUR_ELEMENTS);
        assert_png_is_up_to_date(&FOUR_ELEMENTS_PNG_PATH, FOUR_ELEMENTS);
    }

    #[test]
    fn rearranged_probe_without_filter_matches_reference() {
        assert_elements_match_reference(WITHOUT_FILTER);
        assert_png_is_up_to_date(&WITHOUT_FILTER_PNG_PATH, WITHOUT_FILTER);
    }

    #[test]
    fn reordered_and_repeated_probe_elements_match_reference() {
        let elements = [
            probe::ProbeFeature::DepthBuffer,
            probe::ProbeFeature::Filter,
            probe::ProbeFeature::SolidRect,
            probe::ProbeFeature::DepthBuffer,
            probe::ProbeFeature::Gradient,
        ];

        assert_elements_match_reference(&elements);
    }
}
