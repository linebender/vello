// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for rendering into a texture with the WebGL renderer that aren't covered by the
//! `webgl_texture` variants of the other tests.

use vello_common::color::palette::css::{BLUE, RED};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::Rect;
use vello_gpu::{
    RenderSettings, Resources, Scene, WebGlRenderOptions, WebGlRenderTarget, WebGlRenderer,
    WebGlTextureBindings,
};
use vello_tests::renderer::{TestTextureTarget, create_webgl_depth_buffer, read_webgl_pixels};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;

const SIZE: u16 = 8;

fn renderer() -> (WebGlRenderer, Resources) {
    let canvas: web_sys::HtmlCanvasElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap();
    canvas.set_width(SIZE.into());
    canvas.set_height(SIZE.into());
    WebGlRenderer::new_with(&canvas, RenderSettings::default(), true).unwrap()
}

fn nested_rects(colors: &[AlphaColor<Srgb>]) -> Scene {
    let mut scene = Scene::new(SIZE, SIZE);
    for (inset, color) in colors.iter().enumerate() {
        let inset = inset as f64;
        let end = f64::from(SIZE) - inset;
        scene.set_paint(*color);
        scene.fill_rect(&Rect::new(inset, inset, end, end));
    }
    scene
}

fn assert_all(pixels: &[u8], color: AlphaColor<Srgb>) {
    let expected = color.to_rgba8().to_u8_array();
    assert!(
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| *pixel == expected),
        "expected every pixel to be {expected:?}, got {pixels:?}"
    );
}

#[wasm_bindgen_test]
fn texture_target_restores_drawing_buffer() {
    let (mut renderer, mut resources) = renderer();
    let gl = renderer.gl_context().clone();
    let target = TestTextureTarget::new(&renderer, SIZE, SIZE, None);
    let bindings = WebGlTextureBindings::new();

    renderer
        .render(
            &nested_rects(&[RED]),
            &mut resources,
            target.render_target(),
            &bindings,
            WebGlRenderOptions::default(),
        )
        .unwrap();
    renderer
        .render(
            &nested_rects(&[BLUE]),
            &mut resources,
            WebGlRenderTarget::DrawingBuffer,
            &bindings,
            WebGlRenderOptions::default(),
        )
        .unwrap();

    assert_all(&read_webgl_pixels(&gl, None, SIZE, SIZE), BLUE);
}

#[wasm_bindgen_test]
fn texture_target_clears_shared_depth_buffer() {
    let (mut renderer, mut resources) = renderer();
    let gl = renderer.gl_context().clone();
    let depth = create_webgl_depth_buffer(&gl, SIZE, SIZE);
    let first = TestTextureTarget::new(&renderer, SIZE, SIZE, Some(&depth));
    let second = TestTextureTarget::new(&renderer, SIZE, SIZE, Some(&depth));
    let bindings = WebGlTextureBindings::new();

    for (target, scene) in [
        (&first, nested_rects(&[RED, RED, RED])),
        (&second, nested_rects(&[BLUE])),
    ] {
        renderer
            .render(
                &scene,
                &mut resources,
                target.render_target(),
                &bindings,
                WebGlRenderOptions::default(),
            )
            .unwrap();
    }

    assert_all(&second.read_pixels(&gl), BLUE);
}
