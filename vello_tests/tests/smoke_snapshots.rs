// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests to validate our snapshot testing ability

use scenes::SimpleText;
use vello::{
    kurbo::{Affine, Circle, Rect},
    peniko::{color::palette, Brush, Fill},
    Scene,
};
use vello_tests::{smoke_snapshot_test_sync, TestParams};

fn filled_square(use_cpu: bool) {
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::BLUE),
        None,
        &Rect::from_center_size((10., 10.), (6., 6.)),
    );
    let params = TestParams {
        use_cpu,
        ..TestParams::new("filled_square", 20, 20)
    };
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

fn filled_circle(use_cpu: bool) {
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::BLUE),
        None,
        &Circle::new((10., 10.), 7.),
    );
    let params = TestParams {
        use_cpu,
        ..TestParams::new("filled_circle", 20, 20)
    };
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

fn two_emoji(use_cpu: bool) {
    let mut scene = Scene::new();
    let mut text = SimpleText::new();
    text.add_colr_emoji_run(
        &mut scene,
        24.,
        Affine::translate((0., 24.)),
        None,
        Fill::NonZero,
        "🤠",
    );
    text.add_bitmap_emoji_run(
        &mut scene,
        24.,
        Affine::translate((30., 24.)),
        None,
        Fill::NonZero,
        "🤠",
    );
    let params = TestParams {
        use_cpu,
        ..TestParams::new("two_emoji", 60, 30)
    };
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn filled_square_gpu() {
    filled_square(false);
}

#[test]
// The fine shader still requires a GPU, and so we still get a wgpu device
// skip this for now
#[cfg_attr(skip_gpu_tests, ignore)]
fn filled_square_cpu() {
    filled_square(true);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn filled_circle_gpu() {
    filled_circle(false);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn filled_circle_cpu() {
    filled_circle(true);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn two_emoji_gpu() {
    two_emoji(false);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn two_emoji_cpu() {
    two_emoji(true);
}
