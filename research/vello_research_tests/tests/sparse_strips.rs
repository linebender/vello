// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! End-to-end tests for [`AaConfig::SparseMsaa16`]: the CPU sparse strips (a port of Skia's
//! MSAA sparse strips) feeding the GPU draw, clip, coarse and fine stages.
//!
//! Each scene is rendered with the sparse strips and with the GPU `Area` and `Msaa16`
//! pipelines. The pipelines only differ in how they anti-alias edges, so the sparse strips
//! must be about as close to `Area` as the GPU's own 16x MSAA is. Missing tiles, wrong
//! winding or leaking clips show up as much larger differences.

use std::path::Path;

use nv_flip::{FlipImageRgb8, FlipPool};
use scenes::{ExampleScene, test_scenes};
use vello::peniko::ImageData;
use vello::{AaConfig, Scene};
use vello_research_tests::{
    TestParams, encode_test_scene, render_then_debug_sync, write_png_to_file,
};

fn flip_image(image: &ImageData) -> FlipImageRgb8 {
    let rgb: Vec<u8> = image
        .data
        .data()
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    FlipImageRgb8::with_data(image.width, image.height, &rgb)
}

/// The nv-flip mean error between two renders, and their largest channel difference.
fn diff(reference: &ImageData, test: &ImageData) -> (f32, u8) {
    let error_map = nv_flip::flip(
        flip_image(reference),
        flip_image(test),
        nv_flip::DEFAULT_PIXELS_PER_DEGREE,
    );
    let mean = FlipPool::from_image(&error_map).mean();
    let max = reference
        .data
        .data()
        .iter()
        .zip(test.data.data())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    (mean, max)
}

fn render(scene: &Scene, params: &mut TestParams, aa: AaConfig) -> ImageData {
    params.anti_aliasing = aa;
    render_then_debug_sync(scene, params).unwrap()
}

fn check_scene(test_scene: ExampleScene, name: &str, width: u32, height: u32) {
    let mut params = TestParams::new(format!("sparse_strips_{name}"), width, height);
    let scene = encode_test_scene(test_scene, &mut params);
    let sparse = render(&scene, &mut params, AaConfig::SparseMsaa16);
    let area = render(&scene, &mut params, AaConfig::Area);
    let msaa16 = render(&scene, &mut params, AaConfig::Msaa16);

    let (sparse_vs_area, sparse_max) = diff(&area, &sparse);
    let (msaa16_vs_area, msaa16_max) = diff(&area, &msaa16);
    let (sparse_vs_msaa16, _) = diff(&msaa16, &sparse);
    println!(
        "sparse_strips {name}: flip vs Area: SparseMsaa16 {sparse_vs_area:.5} (max channel \
         diff {sparse_max}), Msaa16 {msaa16_vs_area:.5} (max {msaa16_max}); \
         SparseMsaa16 vs Msaa16 {sparse_vs_msaa16:.5}"
    );

    let limit = 1.5 * msaa16_vs_area + 0.003;
    let failed = sparse_vs_area > limit;
    // Set `VELLO_SPARSE_STRIPS_DUMP` to write the renders for visual inspection.
    if failed || std::env::var_os("VELLO_SPARSE_STRIPS_DUMP").is_some() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("debug_outputs");
        std::fs::create_dir_all(&dir).unwrap();
        for (image, suffix) in [(&sparse, "sparse"), (&area, "area"), (&msaa16, "msaa16")] {
            let path = dir.join(format!("sparse_strips_{name}_{suffix}.png"));
            write_png_to_file(&params, &path, image, None, false).unwrap();
        }
    }
    assert!(
        !failed,
        "{name}: SparseMsaa16 is too far from Area: flip mean {sparse_vs_area} > {limit}. \
         Wrote the renders to debug_outputs/"
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_splash() {
    check_scene(test_scenes::splash_with_tiger(), "splash", 300, 300);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_funky_paths() {
    check_scene(test_scenes::funky_paths(), "funky_paths", 600, 600);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_stroke_styles() {
    check_scene(test_scenes::stroke_styles(), "stroke_styles", 600, 425);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_tricky_strokes() {
    check_scene(test_scenes::tricky_strokes(), "tricky_strokes", 600, 425);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_fill_types() {
    check_scene(test_scenes::fill_types(), "fill_types", 700, 350);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_deep_blend() {
    check_scene(test_scenes::deep_blend(), "deep_blend", 200, 200);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_many_clips() {
    check_scene(test_scenes::many_clips(), "many_clips", 200, 200);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_clip_test() {
    check_scene(test_scenes::clip_test(), "clip_test", 512, 768);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_longpathdash_butt() {
    // Thousands of tiny dashes. The sparse strips use Vello's CPU flattener, so also measure how
    // far Vello's CPU shaders (`use_cpu`, flatten included) are from the GPU ones, both rendered
    // with `Area` by the GPU fine stage.
    //
    // TODO: The sparse render has one spurious solid tile near the bottom right (pixels about
    // 416..432 x 64..80) that the GPU pipelines don't. The flip mean is too coarse to catch it.
    let mut params = TestParams::new("sparse_strips_longpathdash_butt_flatten", 440, 80);
    let scene = encode_test_scene(test_scenes::longpathdash_butt(), &mut params);
    let gpu_flatten = render(&scene, &mut params, AaConfig::Area);
    params.use_cpu = true;
    let cpu_flatten = render(&scene, &mut params, AaConfig::Area);
    let (mean, max) = diff(&gpu_flatten, &cpu_flatten);
    println!("sparse_strips longpathdash_butt: Area, CPU vs GPU flatten: {mean:.5} (max {max})");
    check_scene(
        test_scenes::longpathdash_butt(),
        "longpathdash_butt",
        440,
        80,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_blurred_rounded_rect() {
    check_scene(
        test_scenes::blurred_rounded_rect(),
        "blurred_rounded_rect",
        400,
        400,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_gradient_extend() {
    check_scene(test_scenes::gradient_extend(), "gradient_extend", 200, 200);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_luminance_mask() {
    check_scene(test_scenes::luminance_mask(), "luminance_mask", 55, 55);
}

/// The sparse strips run on the CPU and feed GPU-only shaders, so they can't be combined with
/// `use_cpu`.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn sparse_strips_reject_use_cpu() {
    let mut params = TestParams::new("sparse_strips_reject_use_cpu", 64, 64);
    let scene = encode_test_scene(test_scenes::fill_types(), &mut params);
    params.anti_aliasing = AaConfig::SparseMsaa16;
    params.use_cpu = true;
    assert!(render_then_debug_sync(&scene, &params).is_err());
}
