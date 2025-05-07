// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests for text hinting

// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(
    clippy::cast_possible_truncation,
    clippy::allow_attributes_without_reason
)]

use scenes::SimpleText;
use vello::{
    Scene,
    kurbo::Affine,
    peniko::{Brush, Fill, color::palette},
};
use vello_tests::{TestParams, snapshot_test_sync};

fn encode_hinted_text(text: &str, font_size: f32) -> Scene {
    let mut scene = Scene::new();
    let mut simple_text = SimpleText::new();

    let transform = Affine::translate((0., f64::from(font_size)));
    simple_text.add_var_run(
        &mut scene,
        None,
        font_size,
        &[],
        &Brush::Solid(palette::css::WHITE),
        transform,
        None,
        Fill::EvenOdd,
        text,
        true,
    );

    scene
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn simple_hinted() {
    let font_size = 12.;
    let scene = encode_hinted_text("The quick brown fox", font_size);
    let params = TestParams::new(
        "simple_hinted",
        (font_size * 10.) as _,
        (font_size * 1.1).ceil() as _,
    );
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.02);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn scaled_hinted() {
    let font_size = 12.;
    let text_scene = encode_hinted_text("The quick brown fox", font_size);
    let mut scene = Scene::new();
    scene.append(&text_scene, Some(Affine::scale(1.5)));

    let params = TestParams::new(
        "scaled_hinted",
        (font_size * 15.) as _,
        (font_size * 1.65).ceil() as _,
    );
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.02);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn integer_translation() {
    let font_size = 12.;
    let text_scene = encode_hinted_text("The quick brown fox", font_size);
    let mut scene = Scene::new();
    scene.append(&text_scene, Some(Affine::translate((0., 5.))));

    let params = TestParams::new(
        "integer_translation",
        (font_size * 10.) as _,
        (font_size * 1.1 + 10.).ceil() as _,
    );
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.02);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn non_integer_translation() {
    let font_size = 12.;
    let text_scene = encode_hinted_text("The quick brown fox", font_size);
    let mut scene = Scene::new();
    scene.append(&text_scene, Some(Affine::translate((0., 5.5))));

    let params = TestParams::new(
        "non_integer_translation",
        (font_size * 10.) as _,
        (font_size * 1.1 + 10.).ceil() as _,
    );
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.02);
}
