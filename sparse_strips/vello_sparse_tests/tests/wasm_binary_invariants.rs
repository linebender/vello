// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use wasm_bindgen_test::*;

#[cfg(not(target_feature = "simd128"))]
#[wasm_bindgen_test]
async fn no_simd_instruction_inclusion() {
    // Unless the WASM binary is explicitly built with `RUSTFLAGS=-Ctarget-feature=+simd128` then it
    // is imperative that there isn't a single SIMD instruction in the resulting binary. These can
    // accidentally creep into the binary due to usage of `#![cfg(target_feature = "simd128")]`. Any
    // inclusion of a SIMD instruction in a non-SIMD WASM binary can invalidate the whole binary for
    // browsers (or WebAssembly runtimes) that do not have SIMD support.
    //
    // This test runs when simd128 is not enabled, and self-introspects the binary to ensure no SIMD
    // instructions are included.

    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use wasmparser::{Validator, WasmFeatures};

    let window = web_sys::window().unwrap();
    let url = "/wasm-bindgen-test_bg.wasm";
    let response = JsFuture::from(window.fetch_with_str(url)).await.unwrap();
    let response: web_sys::Response = response.dyn_into().unwrap();
    assert!(response.ok(), "binary could not be fetched");
    let buffer = JsFuture::from(response.array_buffer().unwrap())
        .await
        .unwrap();

    let wasm_module_bytes = web_sys::js_sys::Uint8Array::new(&buffer).to_vec();
    let mut wasm_validator_without_simd =
        Validator::new_with_features(WasmFeatures::all().difference(WasmFeatures::SIMD));

    assert!(
        wasm_validator_without_simd
            .validate_all(&wasm_module_bytes)
            .is_ok(),
        "WebAssembly module contains unexpected SIMD instructions"
    );
}

#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
async fn webgl_probe_succeeds() {
    use vello_hybrid::WebGlProbeStatus;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::HtmlCanvasElement;

    async fn wait_for_animation_frame() {
        let promise = web_sys::js_sys::Promise::new(&mut |resolve, _reject| {
            web_sys::window()
                .unwrap()
                .request_animation_frame(&resolve)
                .unwrap();
        });
        JsFuture::from(promise).await.unwrap();
    }

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(200);
    canvas.set_height(200);

    let mut renderer = vello_hybrid::WebGlRenderer::new(&canvas);
    let mut pending = renderer
        .probe()
        .unwrap_or_else(|error| panic!("WebGlRenderer::probe() failed to render: {error:?}"));

    const MAX_FRAMES: u32 = 600;

    for _ in 0..MAX_FRAMES {
        match pending.try_finish() {
            Ok(WebGlProbeStatus::Complete(result)) => {
                assert!(result.is_success());
                return;
            }
            Ok(WebGlProbeStatus::Pending(next_pending)) => {
                pending = next_pending;
                wait_for_animation_frame().await;
            }
            Err(error) => panic!("WebGlRenderer::probe() readback failed: {error:?}"),
        }
    }

    panic!(
        "WebGlRenderer::probe() did not finish within {} animation frames",
        MAX_FRAMES
    );
}

// This test reproduces a bug where creating a renderer would leave a non-default framebuffer without
// depth attachment bound, as a result of which `DEPTH_BITS` would return 0 when creating a second
// renderer.
#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
fn webgl_create_renderer_twice() {
    use vello_hybrid::WebGlRenderer;
    use wasm_bindgen::JsCast;
    use web_sys::HtmlCanvasElement;

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(16);
    canvas.set_height(16);

    let _ = WebGlRenderer::new(&canvas);
    let _ = WebGlRenderer::new(&canvas);
}
