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
    use vello_gpu::WebGlProbeStatus;
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

    let (mut renderer, _) = vello_gpu::WebGlRenderer::new_with(
        &canvas,
        vello_gpu::RenderSettings::default(),
        use_depth_buffer,
    )
    .unwrap();
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

#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
async fn webgl_shaders_compiled_ahead_create_renderer() {
    use vello_gpu::{RenderSettings, WebGlRenderer, WebGlShaders};
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

    let shaders = WebGlShaders::compile(&canvas, true).unwrap();
    const MAX_FRAMES: u32 = 600;
    let mut frames = 0;
    while !shaders.is_ready() {
        assert!(
            frames < MAX_FRAMES,
            "shader compilation did not finish within {MAX_FRAMES} animation frames"
        );
        frames += 1;
        wait_for_animation_frame().await;
    }

    let (init, mut resources) =
        WebGlRenderer::begin_with_shaders(shaders, RenderSettings::default()).unwrap();
    assert!(
        init.is_ready(),
        "readiness must persist after it is reached"
    );
    let mut renderer = init.finish().unwrap();
    assert!(
        renderer.shader_compilation_stats().poll_count >= 1,
        "readiness checks must be counted as polls"
    );

    let scene = vello_gpu::Scene::new(16, 16);
    renderer
        .render(
            &scene,
            &mut resources,
            &vello_gpu::RenderSize {
                width: 16,
                height: 16,
            },
            &vello_gpu::WebGlTextureBindings::new(),
            vello_common::color::AlphaColor::TRANSPARENT,
        )
        .unwrap();
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

#[cfg(feature = "webgl")]
async fn render_with_shaders(shaders: vello_gpu::WebGlShaders) {
    use vello_gpu::{RenderSettings, WebGlRenderer};
    use wasm_bindgen_futures::JsFuture;

    const MAX_FRAMES: u32 = 600;
    let mut frames = 0;
    while !shaders.is_ready() {
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

    let (init, mut resources) =
        WebGlRenderer::begin_with_shaders(shaders, RenderSettings::default()).unwrap();
    let mut renderer = init.finish().unwrap();
    renderer
        .render(
            &vello_gpu::Scene::new(16, 16),
            &mut resources,
            &vello_gpu::RenderSize {
                width: 16,
                height: 16,
            },
            &vello_gpu::WebGlTextureBindings::new(),
            vello_common::color::AlphaColor::TRANSPARENT,
        )
        .unwrap();
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

    let canvas = create_test_canvas();
    let context_attributes = Object::new();
    Reflect::set(&context_attributes, &"alpha".into(), &JsValue::FALSE).unwrap();
    let precompiled = compile_shaders_from_javascript(&canvas, &context_attributes).await;

    let gl = Reflect::get(&precompiled, &"gl".into())
        .unwrap()
        .unchecked_into::<web_sys::WebGl2RenderingContext>();
    let alpha = Reflect::get(&gl.get_context_attributes().unwrap(), &"alpha".into()).unwrap();
    assert_eq!(
        alpha,
        JsValue::FALSE,
        "application context attributes must reach the created context"
    );

    let shaders = vello_gpu::WebGlShaders::from_precompiled(&canvas, true, &precompiled).unwrap();
    assert!(
        shaders.is_precompiled(),
        "programs compiled by the generated JavaScript must be adopted"
    );
    render_with_shaders(shaders).await;
}

#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
async fn webgl_shaders_precompiled_for_another_canvas_are_recompiled() {
    let precompiled =
        compile_shaders_from_javascript(&create_test_canvas(), &wasm_bindgen::JsValue::UNDEFINED)
            .await;

    let canvas = create_test_canvas();
    let shaders = vello_gpu::WebGlShaders::from_precompiled(&canvas, true, &precompiled).unwrap();
    assert!(
        !shaders.is_precompiled(),
        "programs from another context must not be adopted"
    );
    render_with_shaders(shaders).await;
}

#[cfg(feature = "webgl")]
#[wasm_bindgen_test]
async fn webgl_shaders_precompiled_from_other_sources_are_recompiled() {
    let canvas = create_test_canvas();
    let precompiled =
        compile_shaders_from_javascript(&canvas, &wasm_bindgen::JsValue::UNDEFINED).await;
    web_sys::js_sys::Reflect::set(&precompiled, &"sourcesHash".into(), &"stale".into()).unwrap();

    let shaders = vello_gpu::WebGlShaders::from_precompiled(&canvas, true, &precompiled).unwrap();
    assert!(
        !shaders.is_precompiled(),
        "programs compiled from different sources must not be adopted"
    );
    render_with_shaders(shaders).await;
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
