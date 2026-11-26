// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for rendering into a texture or reading back pixels with the WebGL renderer that aren't
//! covered by the `webgl_texture` variants of the other tests.

use vello_common::color::palette::css::{BLUE, RED};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::Rect;
use vello_common::paint::{Image, ImageSource, TextureRegion};
use vello_common::peniko::ImageSampler;
use vello_gpu::{
    RenderError, RenderSettings, Resources, Scene, TextureId, WebGlError, WebGlRenderOptions,
    WebGlRenderTarget, WebGlRenderer, WebGlTextureBindings,
};
use vello_tests::renderer::{TestTextureTarget, create_webgl_depth_buffer, read_webgl_pixels};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
use web_sys::WebGl2RenderingContext;

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

#[wasm_bindgen_test]
fn texture_target_rejects_sampling_from_itself() {
    let (mut renderer, mut resources) = renderer();
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
    let target = renderer
        .create_texture_target(&texture, None, SIZE, SIZE)
        .unwrap();
    let texture_id = TextureId(0);
    let mut bindings = WebGlTextureBindings::new();
    bindings.insert(texture_id, texture, SIZE, SIZE);

    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_paint(Image {
        image: ImageSource::external_texture(texture_id, TextureRegion::Full, false),
        sampler: ImageSampler::default(),
    });
    scene.fill_rect(&Rect::new(0.0, 0.0, f64::from(SIZE), f64::from(SIZE)));

    let result = renderer.render(
        &scene,
        &mut resources,
        WebGlRenderTarget::Texture(&target),
        &bindings,
        WebGlRenderOptions::default(),
    );
    assert!(
        matches!(
            result,
            Err(WebGlError::Render(RenderError::TextureFeedbackLoop(id))) if id == texture_id
        ),
        "sampling from the render target should fail, got {result:?}"
    );
}

#[wasm_bindgen_test]
fn flip_y_controls_row_order() {
    let (mut renderer, mut resources) = renderer();
    let gl = renderer.gl_context().clone();
    let texture_target = TestTextureTarget::new(&renderer, SIZE, SIZE, None);
    let bindings = WebGlTextureBindings::new();

    let (width, half) = (f64::from(SIZE), f64::from(SIZE / 2));
    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_paint(RED);
    scene.fill_rect(&Rect::new(0.0, 0.0, width, half));
    scene.set_paint(BLUE);
    scene.fill_rect(&Rect::new(0.0, half, width, f64::from(SIZE)));

    for flip_y in [true, false] {
        let mut options = WebGlRenderOptions::default();
        options.flip_y = flip_y;
        for texture_target in [None, Some(&texture_target)] {
            let target =
                texture_target.map_or(WebGlRenderTarget::DrawingBuffer, |t| t.render_target());
            renderer
                .render(&scene, &mut resources, target, &bindings, options)
                .unwrap();
            let pixels = texture_target.map_or_else(
                || read_webgl_pixels(&gl, None, SIZE, SIZE),
                |t| t.read_pixels(&gl),
            );
            let (first, last) = pixels.split_at(pixels.len() / 2);
            let (first_color, last_color) = if flip_y { (BLUE, RED) } else { (RED, BLUE) };
            assert_all(first, first_color);
            assert_all(last, last_color);
        }
    }
}
