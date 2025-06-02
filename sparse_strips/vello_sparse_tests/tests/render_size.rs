// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests where the surface dimensions and scene dimensions do not match.
//!
//! Note: These tests do not use vello_test proc macro as they test an edge case where the render
//! surface is a different size than the scene dimensions.

use crate::renderer::Renderer;
use crate::util::{check_ref_encoded, pixmap_to_png};
use vello_common::color::palette::css::{LIME, REBECCA_PURPLE};
use vello_common::kurbo::Rect;
use vello_common::pixmap::Pixmap;
use vello_hybrid::Scene;

fn draw_simple_scene(scene: &mut Scene, width: f64, height: f64) {
    scene.set_paint(LIME.into());
    // Cover background of scene
    scene.fill_rect(&Rect::new(0.0, 0.0, width, height));

    scene.set_paint(REBECCA_PURPLE.into());
    scene.fill_rect(&Rect::new(10., 20., width - 10., height - 20.));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn scene_smaller_than_surface_hybrid() {
    let scene_width = 50;
    let scene_height = 50;
    let surface_width = 80;
    let surface_height = 80;

    let mut scene = Scene::new(scene_width, scene_height);
    draw_simple_scene(&mut scene, scene_width as f64, scene_height as f64);

    let mut pixmap = Pixmap::new(surface_width, surface_height);

    scene.render_to_pixmap(&mut pixmap, vello_cpu::RenderMode::OptimizeSpeed);
    let encoded_image = pixmap_to_png(pixmap, surface_width as u32, surface_height as u32);

    check_ref_encoded(
        &encoded_image,
        "scene_smaller_than_surface",
        "scene_smaller_than_surface_hybrid",
        1,
        false,
        &[],
    );
}

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn scene_smaller_than_surface_hybrid_webgl() {
    use wasm_bindgen::JsCast;
    use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

    let scene_width = 50;
    let scene_height = 50;
    let surface_width = 80;
    let surface_height = 80;

    let mut scene = Scene::new(scene_width, scene_height);
    draw_simple_scene(&mut scene, scene_width as f64, scene_height as f64);

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();

    canvas.set_width(surface_width as u32);
    canvas.set_height(surface_height as u32);

    // Render the smaller scene to the larger canvas
    let mut renderer = vello_hybrid::WebGlRenderer::new(&canvas);
    let render_size = vello_hybrid::RenderSize {
        width: scene_width as u32,
        height: scene_height as u32,
    };

    renderer.render(&scene, &render_size).unwrap();

    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<WebGl2RenderingContext>()
        .unwrap();

    let mut pixels = vec![0_u8; (surface_width as usize) * (surface_height as usize) * 4];
    gl.read_pixels_with_opt_u8_array(
        0,
        0,
        surface_width.into(),
        surface_height.into(),
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(&mut pixels),
    )
    .unwrap();

    let mut pixmap = Pixmap::new(surface_width, surface_height);
    pixmap.data_as_u8_slice_mut().copy_from_slice(&pixels);

    let encoded_image = pixmap_to_png(pixmap, surface_width as u32, surface_height as u32);

    check_ref_encoded(
        &encoded_image,
        "",
        "scene_smaller_than_surface_hybrid_webgl",
        1,
        false,
        include_bytes!("../snapshots/scene_smaller_than_surface.png"),
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn scene_larger_than_surface_hybrid() {
    let scene_width = 80;
    let scene_height = 80;
    let surface_width = 50;
    let surface_height = 50;

    let mut scene = Scene::new(scene_width, scene_height);
    draw_simple_scene(&mut scene, scene_width as f64, scene_height as f64);

    let mut pixmap = Pixmap::new(surface_width, surface_height);

    scene.render_to_pixmap(&mut pixmap, vello_cpu::RenderMode::OptimizeSpeed);
    let encoded_image = pixmap_to_png(pixmap, surface_width as u32, surface_height as u32);

    check_ref_encoded(
        &encoded_image,
        "scene_larger_than_surface",
        "scene_larger_than_surface_hybrid",
        1,
        false,
        &[],
    );
}

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
#[wasm_bindgen_test::wasm_bindgen_test]
async fn scene_larger_than_surface_hybrid_webgl() {
    use wasm_bindgen::JsCast;
    use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

    let scene_width = 80;
    let scene_height = 80;
    let surface_width = 50;
    let surface_height = 50;

    let mut scene = Scene::new(scene_width, scene_height);
    draw_simple_scene(&mut scene, scene_width as f64, scene_height as f64);

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();

    canvas.set_width(surface_width as u32);
    canvas.set_height(surface_height as u32);

    // Render the larger scene to the smaller canvas
    let mut renderer = vello_hybrid::WebGlRenderer::new(&canvas);
    let render_size = vello_hybrid::RenderSize {
        width: scene_width as u32,
        height: scene_height as u32,
    };

    renderer.render(&scene, &render_size).unwrap();

    let gl = canvas
        .get_context("webgl2")
        .unwrap()
        .unwrap()
        .dyn_into::<WebGl2RenderingContext>()
        .unwrap();

    let mut pixels = vec![0_u8; (surface_width as usize) * (surface_height as usize) * 4];
    gl.read_pixels_with_opt_u8_array(
        0,
        0,
        surface_width.into(),
        surface_height.into(),
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(&mut pixels),
    )
    .unwrap();

    let mut pixmap = Pixmap::new(surface_width, surface_height);
    pixmap.data_as_u8_slice_mut().copy_from_slice(&pixels);

    let encoded_image = pixmap_to_png(pixmap, surface_width as u32, surface_height as u32);

    check_ref_encoded(
        &encoded_image,
        "",
        "scene_larger_than_surface_hybrid_webgl",
        1,
        false,
        include_bytes!("../snapshots/scene_larger_than_surface.png"),
    );
}
