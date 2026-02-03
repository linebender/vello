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
    kurbo::{Affine, Rect, Triangle},
    peniko::{Color, ColorStop, Extend, Gradient, ImageFormat, ImageQuality, Mix, color::palette},
};
use vello_tests::{TestParams, smoke_snapshot_test_sync, snapshot_test_sync};

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

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn many_bins_gpu() {
    many_bins(false);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
#[should_panic]
fn many_bins_cpu() {
    many_bins(true);
}

/// Test for <https://github.com/linebender/vello/issues/1061>
#[test]
#[should_panic]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_layer_size() {
    let mut scene = Scene::new();
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        vello::peniko::color::AlphaColor::from_rgb8(0, 255, 0),
        None,
        &Rect::from_origin_size((0.0, 0.0), (60., 60.)),
    );
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        vello::peniko::color::AlphaColor::from_rgb8(255, 0, 0),
        None,
        &Rect::from_origin_size((20.0, 20.0), (20., 20.)),
    );
    scene.push_layer(
        vello::peniko::Fill::NonZero,
        vello::peniko::Compose::Clear,
        1.0,
        Affine::IDENTITY,
        &Rect::from_origin_size((20.0, 20.0), (20., 20.)),
    );
    scene.pop_layer();
    let params = TestParams::new("layer_size", 60, 60);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

const DATA_IMAGE_PNG: &[u8] = include_bytes!("../snapshots/smoke/data_image_roundtrip.png");

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
#[ignore = "CI runs these tests on a CPU, leading to them having unrealistic precision"] // Uncomment below line when removing this.
// #[cfg_attr(skip_gpu_tests, ignore)]
#[should_panic]
fn test_data_image_roundtrip_extend_reflect() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Reflect);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new(
        "data_image_roundtrip",
        image.image.width,
        image.image.height,
    );
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
#[ignore = "CI runs these tests on a CPU, leading to them having unrealistic precision"] // Uncomment below line when removing this.
// #[cfg_attr(skip_gpu_tests, ignore)]
#[should_panic]
fn test_data_image_roundtrip_extend_repeat() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Repeat);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new(
        "data_image_roundtrip",
        image.image.width,
        image.image.height,
    );
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// <https://github.com/web-platform-tests/wpt/blob/18c64a74b1/html/canvas/element/fill-and-stroke-styles/2d.gradient.interpolate.coloralpha.html>
/// See <https://github.com/linebender/vello/issues/1056>.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_gradient_color_alpha() {
    let mut scene = Scene::new();
    let viewport = Rect::new(0., 0., 100., 50.);
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Gradient::new_linear((0., 0.), (100., 0.)).with_stops([
            ColorStop {
                offset: 0.,
                color: Color::from_rgba8(255, 255, 0, 0).into(),
            },
            ColorStop {
                offset: 1.,
                color: Color::from_rgba8(0, 0, 255, 255).into(),
            },
        ]),
        None,
        &viewport,
    );
    let mut params = TestParams::new("gradient_color_alpha", 100, 50);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// See <https://github.com/linebender/vello/issues/1198>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn clip_blends() {
    let mut scene = Scene::new();

    scene.fill(
        vello::peniko::Fill::EvenOdd,
        Affine::IDENTITY,
        palette::css::BLUE,
        None,
        &Rect::from_origin_size((0., 0.), (100., 100.)),
    );
    let layer_shape = Triangle::from_coords((50., 0.), (0., 100.), (100., 100.));
    scene.push_clip_layer(vello::peniko::Fill::NonZero, Affine::IDENTITY, &layer_shape);
    scene.push_layer(
        vello::peniko::Fill::NonZero,
        Mix::Multiply,
        1.0,
        Affine::IDENTITY,
        &layer_shape,
    );
    scene.fill(
        vello::peniko::Fill::EvenOdd,
        Affine::IDENTITY,
        palette::css::AQUAMARINE,
        None,
        &Rect::from_origin_size((0., 0.), (100., 100.)),
    );
    scene.pop_layer();
    scene.pop_layer();

    let params = TestParams::new("clip_blends", 100, 100);
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}
