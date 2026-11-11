// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests that errors caused by a lost WebGL context are reported as `WebGlError::ContextLost`.

use crate::wasm_binary_invariants::compile_shaders_from_javascript;
use vello_common::color::palette::css::RED;
use vello_common::kurbo::Rect;
use vello_gpu::{
    RenderSettings, Resources, Scene, WebGlError, WebGlProbeError, WebGlRenderOptions,
    WebGlRenderTarget, WebGlRenderer, WebGlTextureBindings,
};
use vello_tests::renderer::create_webgl_depth_buffer;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::*;
use web_sys::js_sys::{Function, Reflect};
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

const SIZE: u16 = 8;

fn canvas() -> HtmlCanvasElement {
    let canvas: HtmlCanvasElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap();
    canvas.set_width(SIZE.into());
    canvas.set_height(SIZE.into());
    canvas
}

fn renderer() -> (WebGlRenderer, Resources) {
    renderer_for(&canvas())
}

fn renderer_for(canvas: &HtmlCanvasElement) -> (WebGlRenderer, Resources) {
    WebGlRenderer::new_with(canvas, RenderSettings::default(), true).unwrap()
}

fn context(canvas: &HtmlCanvasElement) -> WebGl2RenderingContext {
    canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into()
        .unwrap()
}

fn lose_context(gl: &WebGl2RenderingContext) {
    let extension = gl.get_extension("WEBGL_lose_context").unwrap().unwrap();
    Reflect::get(&extension, &"loseContext".into())
        .unwrap()
        .unchecked_into::<Function>()
        .call0(&extension)
        .unwrap();
    assert!(gl.is_context_lost(), "context should be lost");
}

fn assert_context_lost<T: core::fmt::Debug>(result: Result<T, WebGlError>) {
    assert!(
        matches!(result, Err(WebGlError::ContextLost)),
        "expected `WebGlError::ContextLost`, got {result:?}"
    );
}

fn assert_ok_or_context_lost<T: core::fmt::Debug>(result: Result<T, WebGlError>) {
    assert!(
        matches!(result, Ok(_) | Err(WebGlError::ContextLost)),
        "expected success or `WebGlError::ContextLost`, got {result:?}"
    );
}

#[wasm_bindgen_test]
fn webgl_context_lost_before_renderer_recreation() {
    let canvas = canvas();
    let (renderer, _) = renderer_for(&canvas);
    lose_context(renderer.gl_context());

    assert_context_lost(WebGlRenderer::new_with(
        &canvas,
        RenderSettings::default(),
        true,
    ));
}

#[wasm_bindgen_test]
async fn webgl_context_lost_before_adopting_precompiled_shaders() {
    let canvas = canvas();
    let precompiled = compile_shaders_from_javascript(&canvas, &JsValue::UNDEFINED).await;
    lose_context(&context(&canvas));

    assert_context_lost(WebGlRenderer::begin_with_precompiled(
        &precompiled,
        RenderSettings::default(),
    ));
}

#[wasm_bindgen_test]
fn webgl_context_lost_during_shader_compilation() {
    let canvas = canvas();
    let (init, _) = WebGlRenderer::begin_with(&canvas, RenderSettings::default(), true).unwrap();
    lose_context(&context(&canvas));

    assert_context_lost(init.finish());
}

#[wasm_bindgen_test]
fn webgl_context_lost_while_polling_shader_compilation() {
    let canvas = canvas();
    let (init, _) = WebGlRenderer::begin_with(&canvas, RenderSettings::default(), true).unwrap();
    lose_context(&context(&canvas));

    assert_context_lost(init.try_finish());
}

#[wasm_bindgen_test]
fn webgl_context_lost_before_texture_target_creation() {
    let (renderer, _) = renderer();
    let gl = renderer.gl_context().clone();
    let texture = gl.create_texture().unwrap();
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
    gl.tex_storage_2d(
        WebGl2RenderingContext::TEXTURE_2D,
        1,
        WebGl2RenderingContext::RGBA8,
        SIZE.into(),
        SIZE.into(),
    );
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, None);
    let depth = create_webgl_depth_buffer(&gl, SIZE, SIZE);
    lose_context(&gl);

    assert_context_lost(renderer.create_texture_target(&texture, Some(&depth), SIZE, SIZE));
}

#[wasm_bindgen_test]
fn webgl_context_lost_before_render() {
    let (mut renderer, mut resources) = renderer();
    lose_context(renderer.gl_context());
    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_paint(RED);
    scene.fill_rect(&Rect::new(0.0, 0.0, f64::from(SIZE), f64::from(SIZE)));

    assert_ok_or_context_lost(renderer.render(
        &scene,
        &mut resources,
        WebGlRenderTarget::DrawingBuffer,
        &WebGlTextureBindings::new(),
        WebGlRenderOptions::default(),
    ));
    assert_ok_or_context_lost(renderer.reset_and_shrink(&mut resources));
}

#[wasm_bindgen_test]
fn webgl_context_lost_before_probe() {
    let (mut renderer, _) = renderer();
    lose_context(renderer.gl_context());

    let result = renderer.probe(vello_common::probe::ALL_PROBE_ELEMENTS);
    assert!(
        matches!(result, Err(WebGlProbeError::WebGl(WebGlError::ContextLost))),
        "expected `WebGlError::ContextLost`, got {result:?}"
    );
}

#[wasm_bindgen_test]
fn webgl_context_lost_before_probe_finishes() {
    let (mut renderer, _) = renderer();
    let pending = renderer
        .probe(vello_common::probe::ALL_PROBE_ELEMENTS)
        .unwrap();
    lose_context(renderer.gl_context());

    let result = pending.try_finish();
    assert!(
        matches!(result, Err(WebGlProbeError::WebGl(WebGlError::ContextLost))),
        "expected `WebGlError::ContextLost`, got {result:?}"
    );
}
