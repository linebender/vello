// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests to ensure that certain issues which don't deserve a test scene don't regress

use scenes::ImageCache;
use vello::{
    AaConfig, Scene,
    kurbo::{Affine, RoundedRect, Stroke},
    peniko::{ImageQuality, color::palette},
};
use vello_tests::{TestParams, snapshot_test_sync};

/// Test created from <https://github.com/linebender/vello/issues/616>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn rounded_rectangle_watertight() {
    let mut scene = Scene::new();
    let rect = RoundedRect::new(60.0, 10.0, 80.0, 30.0, 10.0);
    let stroke = Stroke::new(2.0);
    scene.stroke(&stroke, Affine::IDENTITY, palette::css::WHITE, None, &rect);
    let mut params = TestParams::new("rounded_rectangle_watertight", 70, 30);
    params.anti_aliasing = AaConfig::Msaa16;
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

const DATA_IMAGE_PNG: &[u8] = include_bytes!("../snapshots/data_image_roundtrip.png");

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
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
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}
