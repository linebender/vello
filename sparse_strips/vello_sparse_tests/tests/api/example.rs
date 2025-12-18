// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! An example expansion of `#[vello_api_test]`, which makes editing the code of that macro easier.

use vello_api::{PaintScene, Renderer, peniko::Fill};
use vello_cpu::{
    color::palette::css,
    kurbo::{Affine, Circle},
};

use crate::api::infra::{self, TestParams};

// #[vello_api_test] on this function expands to everything below
fn my_test(scene: &mut impl PaintScene, _renderer: &dyn Renderer) {
    scene.set_solid_brush(css::BLUE_VIOLET);
    scene.fill_path(
        Affine::IDENTITY,
        Fill::EvenOdd,
        Circle {
            center: (50., 50.).into(),
            radius: 25.,
        },
    );
}

const MY_TEST_PARAMS: TestParams = TestParams {
    name: "my_test",
    cpu_u8_scalar_tolerance: 2,
    cpu_u8_simd_tolerance: 2,
    cpu_f32_scalar_tolerance: 0,
    cpu_f32_simd_tolerance: 1,
    hybrid_tolerance: 1,
    diff_pixels: 0,
    width: 100,
    height: 100,
    transparent: false,
    #[cfg(target_arch = "wasm32")]
    reference_image_bytes: Some(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/snapshots/",
        "my_test",
        ".png"
    ))),
    #[cfg(not(target_arch = "wasm32"))]
    reference_image_bytes: None,
    no_ref: false,
};

#[test]
fn my_test_cpu_u8_scalar() {
    let Some(level) = infra::parse_level("fallback") else {
        // We pass tests when the SIMD level isn't supported on this machine.
        // This is a trade-off for better cross-compilation.
        return;
    };
    infra::run_test_cpu(
        my_test,
        MY_TEST_PARAMS,
        vello_cpu::RenderSettings {
            level,
            num_threads: 0,
            render_mode: vello_cpu::RenderMode::OptimizeSpeed,
        },
        "my_test_cpu_u8_scalar",
        MY_TEST_PARAMS.cpu_u8_scalar_tolerance,
        true,
    );
}

#[test]
fn my_test_cpu_f32_scalar() {
    let Some(level) = infra::parse_level("fallback") else {
        // We pass tests when the SIMD level isn't supported on this machine.
        // This is a trade-off for better cross-compilation.
        return;
    };
    infra::run_test_cpu(
        my_test,
        MY_TEST_PARAMS,
        vello_cpu::RenderSettings {
            level,
            num_threads: 0,
            render_mode: vello_cpu::RenderMode::OptimizeQuality,
        },
        "my_test_cpu_f32_scalar",
        MY_TEST_PARAMS.cpu_f32_scalar_tolerance,
        false,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn my_test_cpu_u8_sse4() {
    let Some(level) = infra::parse_level("sse42") else {
        // We pass tests when the SIMD level isn't supported on this machine.
        // This is a trade-off for better cross-compilation.
        return;
    };
    infra::run_test_cpu(
        my_test,
        MY_TEST_PARAMS,
        vello_cpu::RenderSettings {
            level,
            num_threads: 0,
            render_mode: vello_cpu::RenderMode::OptimizeSpeed,
        },
        "my_test_cpu_u8_sse4",
        MY_TEST_PARAMS.cpu_u8_simd_tolerance,
        false,
    );
}
// Etc.

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn my_test_hybrid_wgpu() {
    // Currently we only test Vello Hybrid with the fallback level, to avoid making too many variations
    let Some(level) = infra::parse_level("fallback") else {
        // We pass tests when the SIMD level isn't supported on this machine.
        // This is a trade-off for better cross-compilation.
        return;
    };
    infra::run_test_hybrid_wgpu(
        my_test,
        MY_TEST_PARAMS,
        vello_hybrid::RenderSettings {
            level,
            atlas_config: vello_hybrid::AtlasConfig::default(),
        },
        "my_test_hybrid_wgpu",
        MY_TEST_PARAMS.hybrid_tolerance,
        false,
    );
}

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn my_test_hybrid_webgl() {
    // Currently we only test Vello Hybrid with the fallback level, to avoid making too many variations
    let Some(level) = infra::parse_level("fallback") else {
        // We pass tests when the SIMD level isn't supported on this machine.
        // This is a trade-off for better cross-compilation and support for weak CI machines.
        return;
    };
    infra::run_test_hybrid_webgl(
        my_test,
        MY_TEST_PARAMS,
        vello_hybrid::RenderSettings {
            level,
            atlas_config: vello_hybrid::AtlasConfig::default(),
        },
        "my_test_hybrid_webgl",
        MY_TEST_PARAMS.hybrid_tolerance,
    );
}
