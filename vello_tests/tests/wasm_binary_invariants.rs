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
async fn webgl_probe_succeeds_with_depth_buffer() {
    run_webgl_probe(true).await;
}

#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
async fn webgl_probe_succeeds_without_depth_buffer() {
    run_webgl_probe(false).await;
}

#[cfg(feature = "webgl")]
async fn run_webgl_probe(use_depth_buffer: bool) {
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

    let (mut renderer, _) = vello_gpu::WebGlRenderer::new_with(
        &canvas,
        vello_gpu::RenderSettings::default(),
        use_depth_buffer,
    )
    .unwrap();
    assert_probe_succeeds(&mut renderer).await;
}

/// Renders all probe elements with `renderer` and asserts that the output matches the reference.
#[cfg(feature = "webgl")]
async fn assert_probe_succeeds(renderer: &mut vello_gpu::WebGlRenderer) {
    use vello_gpu::WebGlProbeStatus;
    use wasm_bindgen_futures::JsFuture;

    async fn wait_for_animation_frame() {
        let promise = web_sys::js_sys::Promise::new(&mut |resolve, _reject| {
            web_sys::window()
                .unwrap()
                .request_animation_frame(&resolve)
                .unwrap();
        });
        JsFuture::from(promise).await.unwrap();
    }

    let mut pending = renderer
        .probe(vello_common::probe::ALL_PROBE_ELEMENTS)
        .unwrap_or_else(|error| panic!("WebGlRenderer::probe() failed to render: {error:?}"));

    const MAX_FRAMES: u32 = 600;

    for _ in 0..MAX_FRAMES {
        match pending.try_finish() {
            Ok(WebGlProbeStatus::Complete(result)) => {
                assert!(result.is_success(), "probe failed unexpectedly");
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

#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
async fn webgl_pending_renderer_init_completes() {
    use vello_gpu::WebGlRendererInitStatus;
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
    canvas.set_width(16);
    canvas.set_height(16);

    let (mut init, _) = vello_gpu::WebGlRenderer::begin(&canvas).unwrap();
    const MAX_FRAMES: u32 = 600;

    for _ in 0..MAX_FRAMES {
        match init.try_finish().unwrap() {
            WebGlRendererInitStatus::Complete(_) => return,
            WebGlRendererInitStatus::Pending(next_init) => {
                init = next_init;
                wait_for_animation_frame().await;
            }
        }
    }

    panic!(
        "WebGlRenderer initialization did not finish within {} animation frames",
        MAX_FRAMES
    );
}

/// Loads `vello_gpu_shaders`' JavaScript module the way a page would before the WASM binary, then
/// calls its `compileVelloShaders` on `canvas` with `context_attributes`.
#[cfg(feature = "webgl")]
async fn compile_shaders_from_javascript(
    canvas: &web_sys::HtmlCanvasElement,
    context_attributes: &wasm_bindgen::JsValue,
) -> wasm_bindgen::JsValue {
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_futures::JsFuture;
    use web_sys::js_sys::{Function, Promise, Reflect};

    let import = Function::new_with_args(
        "source",
        "return import('data:text/javascript;charset=utf-8,' + encodeURIComponent(source));",
    );
    let module = import
        .call1(
            &JsValue::NULL,
            &vello_gpu_shaders::webgl_bootstrap::JS_MODULE.into(),
        )
        .unwrap()
        .dyn_into::<Promise>()
        .unwrap();
    let module = JsFuture::from(module).await.unwrap();
    Reflect::get(&module, &"compileVelloShaders".into())
        .unwrap()
        .dyn_into::<Function>()
        .unwrap()
        .call2(&JsValue::NULL, canvas, context_attributes)
        .unwrap()
}

/// Creates a renderer from `precompiled` and runs the probe with it.
#[cfg(feature = "webgl")]
async fn render_with_precompiled(precompiled: &wasm_bindgen::JsValue) {
    use vello_gpu::{RenderSettings, WebGlRenderer};
    use wasm_bindgen_futures::JsFuture;

    let (init, _) =
        WebGlRenderer::begin_with_precompiled(precompiled, RenderSettings::default()).unwrap();

    const MAX_FRAMES: u32 = 600;
    let mut frames = 0;
    while !init.is_ready() {
        assert!(
            frames < MAX_FRAMES,
            "shader compilation did not finish within {MAX_FRAMES} animation frames"
        );
        frames += 1;
        let promise = web_sys::js_sys::Promise::new(&mut |resolve, _reject| {
            web_sys::window()
                .unwrap()
                .request_animation_frame(&resolve)
                .unwrap();
        });
        JsFuture::from(promise).await.unwrap();
    }
    assert!(
        init.is_ready(),
        "readiness must persist after it is reached"
    );

    let mut renderer = init.finish().unwrap();
    assert!(
        renderer.shader_compilation_stats().poll_count >= 1,
        "readiness checks must be counted as polls"
    );
    assert_probe_succeeds(&mut renderer).await;
}

#[cfg(feature = "webgl")]
fn create_test_canvas() -> web_sys::HtmlCanvasElement {
    use wasm_bindgen::JsCast;

    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(16);
    canvas.set_height(16);
    canvas
}

#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
async fn webgl_shaders_precompiled_by_javascript_are_adopted() {
    use wasm_bindgen::{JsCast, JsValue};
    use web_sys::js_sys::{Object, Reflect};

    let context_attributes = Object::new();
    Reflect::set(&context_attributes, &"alpha".into(), &JsValue::FALSE).unwrap();
    Reflect::set(&context_attributes, &"antialias".into(), &JsValue::TRUE).unwrap();
    let precompiled =
        compile_shaders_from_javascript(&create_test_canvas(), &context_attributes).await;

    let gl = Reflect::get(&precompiled, &"gl".into())
        .unwrap()
        .unchecked_into::<web_sys::WebGl2RenderingContext>();
    let created_attributes = gl.get_context_attributes().unwrap();
    let created_attribute = |name: &str| Reflect::get(&created_attributes, &name.into()).unwrap();
    assert_eq!(
        created_attribute("alpha"),
        JsValue::FALSE,
        "application context attributes must reach the created context"
    );
    assert_eq!(
        created_attribute("antialias"),
        JsValue::FALSE,
        "applications must not be able to enable antialiasing"
    );

    render_with_precompiled(&precompiled).await;
}

#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
async fn webgl_shaders_precompiled_without_depth_buffer_are_adopted() {
    use wasm_bindgen::JsValue;
    use web_sys::js_sys::{Object, Reflect};

    let context_attributes = Object::new();
    Reflect::set(&context_attributes, &"depth".into(), &JsValue::FALSE).unwrap();
    let precompiled =
        compile_shaders_from_javascript(&create_test_canvas(), &context_attributes).await;

    render_with_precompiled(&precompiled).await;
}

#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
async fn webgl_contexts_created_by_javascript_and_rust_match() {
    use vello_gpu::{RenderSettings, WebGlRenderer};
    use wasm_bindgen::{JsCast, JsValue};
    use web_sys::WebGl2RenderingContext;
    use web_sys::js_sys::{JSON, Object, Reflect};

    /// Serializes all of `gl`'s context attributes with sorted keys, so they can be compared.
    fn attributes_json(gl: &WebGl2RenderingContext) -> String {
        let attributes = gl.get_context_attributes().unwrap();
        let keys = Object::keys(&attributes).sort();
        JSON::stringify_with_replacer(&attributes, &keys)
            .unwrap()
            .into()
    }

    let precompiled =
        compile_shaders_from_javascript(&create_test_canvas(), &JsValue::UNDEFINED).await;
    let javascript_gl = Reflect::get(&precompiled, &"gl".into())
        .unwrap()
        .unchecked_into::<WebGl2RenderingContext>();

    let canvas = create_test_canvas();
    let _renderer = WebGlRenderer::begin_with(&canvas, RenderSettings::default(), true).unwrap();
    let rust_gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .unchecked_into::<WebGl2RenderingContext>();

    assert_eq!(
        attributes_json(&javascript_gl),
        attributes_json(&rust_gl),
        "`compileVelloShaders` and `WebGlRenderer` must create contexts with the same attributes"
    );
}

/// Returns why creating a renderer from `precompiled` was rejected.
#[cfg(feature = "webgl")]
fn precompiled_rejection(
    precompiled: &wasm_bindgen::JsValue,
) -> vello_gpu::IncompatiblePrecompiledShadersReason {
    use vello_gpu::{RenderSettings, WebGlError, WebGlRenderer};

    match WebGlRenderer::begin_with_precompiled(precompiled, RenderSettings::default()) {
        Err(WebGlError::IncompatiblePrecompiledShaders(reason)) => reason,
        Err(error) => panic!("expected incompatible precompiled shaders, got {error:?}"),
        Ok(_) => panic!("incompatible precompiled shaders must be rejected"),
    }
}

#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
fn webgl_null_precompiled_shaders_are_rejected() {
    assert_eq!(
        precompiled_rejection(&wasm_bindgen::JsValue::NULL),
        vello_gpu::IncompatiblePrecompiledShadersReason::NotCompileVelloShadersResult,
        "the `null` that `compileVelloShaders` returns without WebGL2 must be rejected"
    );
}

#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
async fn webgl_shaders_precompiled_from_another_module_are_rejected() {
    use vello_gpu::{RenderSettings, WebGlRenderer};

    let canvas = create_test_canvas();
    let precompiled =
        compile_shaders_from_javascript(&canvas, &wasm_bindgen::JsValue::UNDEFINED).await;
    web_sys::js_sys::Reflect::set(&precompiled, &"moduleHash".into(), &"stale".into()).unwrap();

    assert_eq!(
        precompiled_rejection(&precompiled),
        vello_gpu::IncompatiblePrecompiledShadersReason::DifferentModule,
        "programs compiled by a different module build must be rejected"
    );

    // Falling back after a rejection must work on the canvas that already has a context.
    let (init, _) = WebGlRenderer::begin_with(&canvas, RenderSettings::default(), true).unwrap();
    assert_probe_succeeds(&mut init.finish().unwrap()).await;
}

// This test reproduces a bug where creating a renderer would leave a non-default framebuffer without
// depth attachment bound, as a result of which `DEPTH_BITS` would return 0 when creating a second
// renderer.
#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
fn webgl_create_renderer_twice() {
    use vello_gpu::WebGlRenderer;
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

    let _ = WebGlRenderer::new(&canvas).unwrap();
    let _ = WebGlRenderer::new(&canvas).unwrap();
}
