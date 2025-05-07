// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reproductions for known bugs, to allow test driven development

// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(
    clippy::missing_assert_message,
    clippy::should_panic_without_expect,
    clippy::allow_attributes_without_reason
)]

use scenes::ImageCache;
use vello::{
    AaConfig, Scene,
    kurbo::{Affine, Rect},
    peniko::{ImageFormat, ImageQuality, color::palette},
};
use vello_tests::{TestParams, smoke_snapshot_test_sync};

/// A reproduction of <https://github.com/linebender/vello/issues/680>
fn many_bins(use_cpu: bool) {
    let mut scene = Scene::new();
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        palette::css::RED,
        None,
        &Rect::new(-5., -5., 256. * 20., 256. * 20.),
    );
    let params = TestParams {
        use_cpu,
        ..TestParams::new("many_bins", 256 * 17, 256 * 17)
    };
    // To view, use VELLO_DEBUG_TEST=many_bins
    let image = vello_tests::render_then_debug_sync(&scene, &params).unwrap();
    assert_eq!(image.format, ImageFormat::Rgba8);
    let mut red_count = 0;
    let mut black_count = 0;
    for pixel in image.data.data().chunks_exact(4) {
        let &[r, g, b, a] = pixel else { unreachable!() };
        let is_red = r == 255 && g == 0 && b == 0 && a == 255;
        let is_black = r == 0 && g == 0 && b == 0 && a == 255;
        if !is_red && !is_black {
            panic!("{pixel:?}");
        }
        match (is_red, is_black) {
            (true, true) => unreachable!(),
            (true, false) => red_count += 1,
            (false, true) => black_count += 1,
            (false, false) => panic!("Got unexpected pixel {pixel:?}"),
        }
    }
    // When #680 is fixed, this will become:
    // let drawn_bins = 17 /* x bins */ * 17 /* y bins*/;

    // The current maximum number of bins.
    let drawn_bins = 256;
    let expected_red_count = drawn_bins * 256 /* tiles per bin */ * 256 /* Pixels per tile */;
    assert_eq!(red_count, expected_red_count);
    assert!(black_count > 0);
}

// With wgpu 23, this started mysteriously working on macOS (and only on macOS).
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
#[cfg_attr(target_os = "macos", should_panic)]
fn many_bins_gpu() {
    many_bins(false);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
#[should_panic]
fn many_bins_cpu() {
    many_bins(true);
}

const DATA_IMAGE_PNG: &[u8] = include_bytes!("../snapshots/smoke/data_image_roundtrip.png");

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
// Note that this isn't exactly 0.3, it's a number starting with "0.3"
#[should_panic(expected = "Expected mean to be less than 0.001, got 0.3")]
fn test_data_image_roundtrip() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new("data_image_roundtrip", image.width, image.height);
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}
