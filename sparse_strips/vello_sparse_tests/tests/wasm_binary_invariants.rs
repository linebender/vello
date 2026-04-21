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
fn webgl_probe_succeeds() {
    use vello_hybrid::Probe;
    use wasm_bindgen::JsCast;
    use web_sys::HtmlCanvasElement;

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(200);
    canvas.set_height(200);

    let mut renderer = vello_hybrid::WebGlRenderer::new(&canvas);
    match renderer.probe() {
        Probe::Success => {}
        Probe::Error(result) => {
            let expected = png_data_url(&result.expected);
            let actual = png_data_url(&result.actual);
            panic!(
                "WebGlRenderer::probe() unexpectedly failed with a pixel mismatch\nexpected_png={expected}\nactual_png={actual}"
            );
        }
        Probe::RenderError(error) => {
            panic!("WebGlRenderer::probe() failed to render: {error:?}");
        }
    }
}

#[cfg(feature = "webgl")]
fn png_data_url(png: &[u8]) -> String {
    let base64 = base64_encode(png);
    format!("data:image/png;base64,{base64}")
}

#[cfg(feature = "webgl")]
fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let encoded_len = bytes.len().div_ceil(3) * 4;
    let mut encoded = String::with_capacity(encoded_len);

    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let word = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);

        encoded.push(TABLE[((word >> 18) & 0x3f) as usize] as char);
        encoded.push(TABLE[((word >> 12) & 0x3f) as usize] as char);
        encoded.push(if chunk.len() > 1 {
            TABLE[((word >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            TABLE[(word & 0x3f) as usize] as char
        } else {
            '='
        });
    }

    encoded
}
