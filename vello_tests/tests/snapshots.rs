// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use scenes::{test_scenes, ExampleScene};
use vello_tests::{encode_test_scene, snapshot_test_sync, TestParams};

/// Make sure the CPU and GPU renderers match on the test scenes
fn snapshot_test_scene(test_scene: ExampleScene, mut params: TestParams) {
    let scene = encode_test_scene(test_scene, &mut params);
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_splash() {
    let test_scene = test_scenes::splash_with_tiger();
    let params = TestParams::new("splash", 300, 300);

    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_funky_paths() {
    let test_scene = test_scenes::funky_paths();
    let params = TestParams::new("funky_paths", 600, 600);

    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_stroke_styles() {
    let test_scene = test_scenes::stroke_styles();
    let params = TestParams::new("stroke_styles", 600, 425);

    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_stroke_styles_non_uniform() {
    let test_scene = test_scenes::stroke_styles_non_uniform();
    let params = TestParams::new("stroke_styles_non_uniform", 600, 425);

    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_stroke_styles_skew() {
    let test_scene = test_scenes::stroke_styles_skew();
    let params = TestParams::new("stroke_styles_skew", 600, 425);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_tricky_strokes() {
    let test_scene = test_scenes::tricky_strokes();
    let params = TestParams::new("tricky_strokes", 600, 425);
    snapshot_test_scene(test_scene, params);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn snapshot_fill_types() {
    let test_scene = test_scenes::fill_types();
    assert_eq!(test_scene.config.name, "fill_types");
    let params = TestParams::new("fill_types", 700, 350);
    snapshot_test_scene(test_scene, params);
}
