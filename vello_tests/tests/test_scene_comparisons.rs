// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use scenes::{ExampleScene, ImageCache, SceneParams, SimpleText};
use vello::{
    kurbo::{Affine, Vec2},
    Scene,
};
use vello_tests::{compare_test_sync, TestParams};

/// Make sure the CPU and GPU renderers match on the test scenes
fn encode_test_scene(mut test_scene: ExampleScene, test_params: &mut TestParams) -> Scene {
    let mut inner_scene = Scene::new();
    let mut image_cache = ImageCache::new();
    let mut text = SimpleText::new();
    let mut scene_params = SceneParams {
        base_color: None,
        complexity: 100,
        time: 0.,
        images: &mut image_cache,
        interactive: false,
        resolution: None,
        text: &mut text,
    };
    test_scene
        .function
        .render(&mut inner_scene, &mut scene_params);
    if test_params.base_colour.is_none() {
        test_params.base_colour = scene_params.base_color;
    }
    if let Some(resolution) = scene_params.resolution {
        // Automatically scale the rendering to fill as much of the window as possible
        let factor = Vec2::new(test_params.width as f64, test_params.height as f64);
        let scale_factor = (factor.x / resolution.x).min(factor.y / resolution.y);
        let mut outer_scene = Scene::new();
        outer_scene.append(&inner_scene, Some(Affine::scale(scale_factor)));
        outer_scene
    } else {
        inner_scene
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_splash() {
    let test_scene = scenes::test_scenes().scenes.swap_remove(0);
    assert_eq!(test_scene.config.name, "splash_with_tiger");
    let mut params = TestParams::new("compare_splash", 600, 600);
    let scene = encode_test_scene(test_scene, &mut params);
    match compare_test_sync(scene, params)
        .and_then(|mut snapshot| snapshot.assert_mean_less_than(0.01))
    {
        Ok(()) => (),
        Err(e) => panic!("{:#}", e),
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_funky_paths() {
    let test_scene = scenes::test_scenes().scenes.swap_remove(1);
    assert_eq!(test_scene.config.name, "funky_paths");
    let mut params = TestParams::new("compare_funky_paths", 600, 600);
    let scene = encode_test_scene(test_scene, &mut params);
    match compare_test_sync(scene, params)
        .and_then(|mut snapshot| snapshot.assert_mean_less_than(0.01))
    {
        Ok(()) => (),
        Err(e) => panic!("{:#}", e),
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_stroke_styles() {
    let test_scene = scenes::test_scenes().scenes.swap_remove(2);
    assert_eq!(test_scene.config.name, "stroke_styles");
    let mut params = TestParams::new("compare_stroke_styles", 1200, 850);
    let scene = encode_test_scene(test_scene, &mut params);
    match compare_test_sync(scene, params)
        .and_then(|mut snapshot| snapshot.assert_mean_less_than(0.01))
    {
        Ok(()) => (),
        Err(e) => panic!("{:#}", e),
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_stroke_styles_non_uniform() {
    let test_scene = scenes::test_scenes().scenes.swap_remove(3);
    assert_eq!(test_scene.config.name, "stroke_styles (non-uniform scale)");
    let mut params = TestParams::new("compare_stroke_styles_non_uniform", 1200, 850);
    let scene = encode_test_scene(test_scene, &mut params);
    match compare_test_sync(scene, params)
        .and_then(|mut snapshot| snapshot.assert_mean_less_than(0.01))
    {
        Ok(()) => (),
        Err(e) => panic!("{:#}", e),
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_stroke_styles_skew() {
    let test_scene = scenes::test_scenes().scenes.swap_remove(4);
    assert_eq!(test_scene.config.name, "stroke_styles (skew)");
    let mut params = TestParams::new("compare_stroke_styles_skew", 1200, 850);
    let scene = encode_test_scene(test_scene, &mut params);
    match compare_test_sync(scene, params)
        .and_then(|mut snapshot| snapshot.assert_mean_less_than(0.01))
    {
        Ok(()) => (),
        Err(e) => panic!("{:#}", e),
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_tricky_strokes() {
    let test_scene = scenes::test_scenes().scenes.swap_remove(5);
    assert_eq!(test_scene.config.name, "tricky_strokes");
    let mut params = TestParams::new("compare_tricky_strokes", 1200, 850);
    let scene = encode_test_scene(test_scene, &mut params);
    match compare_test_sync(scene, params)
        .and_then(|mut snapshot| snapshot.assert_mean_less_than(0.01))
    {
        Ok(()) => (),
        Err(e) => panic!("{:#}", e),
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn compare_fill_types() {
    let test_scene = scenes::test_scenes().scenes.swap_remove(6);
    assert_eq!(test_scene.config.name, "fill_types");
    let mut params = TestParams::new("compare_fill_types", 1400, 700);
    let scene = encode_test_scene(test_scene, &mut params);
    match compare_test_sync(scene, params)
        .and_then(|mut snapshot| snapshot.assert_mean_less_than(0.01))
    {
        Ok(()) => (),
        Err(e) => panic!("{:#}", e),
    }
}
