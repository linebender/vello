// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{Pixmap, ProbeFeature, probe};
use vello_common::{
    color::palette::css, paint::ImageSource, peniko::ImageAlphaType, pixmap::PixelMetadata,
};
use vello_gpu::{RenderSettings, RenderSize, Scene, WebGlRenderer, WebGlTextureBindings};
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

pub(crate) fn render(elements: &[ProbeFeature]) -> Pixmap {
    let (width, height) = probe::canvas_size(elements);
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();

    canvas.set_width(width.into());
    canvas.set_height(height.into());

    let (mut renderer, mut resources) =
        WebGlRenderer::new_with(&canvas, RenderSettings::default(), true).unwrap();

    let image = probe::probe_image_pixmap();
    let image_id = renderer.upload_image(&mut resources, &image).unwrap();
    let mut scene = Scene::new(width, height);

    probe::draw_scene(
        &mut scene,
        ImageSource::opaque_id_with_transparency_hint(image_id, image.may_have_transparency()),
        elements,
    );

    renderer
        .render(
            &scene,
            &mut resources,
            &RenderSize { width, height },
            &WebGlTextureBindings::new(),
            css::WHITE,
        )
        .unwrap();

    let gl = renderer.gl_context();
    let mut pixels = vec![0; usize::from(width) * usize::from(height) * 4];

    gl.read_pixels_with_opt_u8_array(
        0,
        0,
        width.into(),
        height.into(),
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(&mut pixels),
    )
    .unwrap();

    let row_bytes = usize::from(width) * 4;

    for y in 0..usize::from(height) / 2 {
        let (top, bottom) = pixels.split_at_mut((usize::from(height) - 1 - y) * row_bytes);
        top[y * row_bytes..(y + 1) * row_bytes].swap_with_slice(&mut bottom[..row_bytes]);
    }

    Pixmap::from_parts(
        pixels,
        width,
        height,
        PixelMetadata::new(ImageAlphaType::AlphaPremultiplied, true),
    )
}
