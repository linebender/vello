// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello::{
    kurbo::{Affine, RoundedRect, Stroke},
    peniko::color::palette,
    AaConfig, Scene,
};
use vello_tests::{snapshot_test_sync, TestParams};

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
