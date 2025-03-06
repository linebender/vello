// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests which ensure that the GPU and CPU renderers give the same results across
//! a range of our test scenes.
//!
//! The long-term intention is for our CPU renderer to provide an independent implementation
//! (currently `fine` is shared), so that this can be a robust first-line of defence.
//!
//! This type of test is useful, as it avoids committing large snapshots to the repository, which are
//! not handled very well by git.

use scenes::{ExampleScene, test_scenes};
use vello_tests::{TestParams, compare_gpu_cpu_sync, encode_test_scene};

/// Make sure the CPU and GPU renderers match on the test scenes
fn compare_test_scene(test_scene: ExampleScene, mut params: TestParams) {
    let scene = encode_test_scene(test_scene, &mut params);
    compare_gpu_cpu_sync(scene, params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_splash() {
    let test_scene = test_scenes::splash_with_tiger();
    let params = TestParams::new("compare_splash", 600, 600);

    compare_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_funky_paths() {
    let test_scene = test_scenes::funky_paths();
    let params = TestParams::new("compare_funky_paths", 600, 600);

    compare_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_stroke_styles() {
    let test_scene = test_scenes::stroke_styles();
    let params = TestParams::new("compare_stroke_styles", 1200, 850);

    compare_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_stroke_styles_non_uniform() {
    let test_scene = test_scenes::stroke_styles_non_uniform();
    let params = TestParams::new("compare_stroke_styles_non_uniform", 1200, 850);

    compare_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_stroke_styles_skew() {
    let test_scene = test_scenes::stroke_styles_skew();
    let params = TestParams::new("compare_stroke_styles_skew", 1200, 850);
    compare_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_tricky_strokes() {
    let test_scene = test_scenes::tricky_strokes();
    let params = TestParams::new("compare_tricky_strokes", 1200, 850);
    compare_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_fill_types() {
    let test_scene = test_scenes::fill_types();
    let params = TestParams::new("compare_fill_types", 1400, 700);
    compare_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_deep_blend() {
    let test_scene = test_scenes::deep_blend();
    assert_eq!(test_scene.config.name, "deep_blend");
    let params = TestParams::new("compare_deep_blend", 150, 150);
    compare_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_blurred_rounded_rect() {
    let test_scene = test_scenes::blurred_rounded_rect();
    assert_eq!(test_scene.config.name, "blurred_rounded_rect");
    let params = TestParams::new("compare_blurred_rounded_rect", 1200, 1200);
    compare_test_scene(test_scene, params);
}
