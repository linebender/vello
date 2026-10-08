// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{Pixmap, ProbeFeature, probe};
use vello_common::{
    TextureId,
    color::palette::css,
    paint::{ImageSource, TextureRegion},
    peniko::ImageAlphaType,
    pixmap::PixelMetadata,
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
    let gl = renderer.gl_context();
    let image_texture = gl.create_texture().unwrap();
    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&image_texture));
    gl.tex_storage_2d(
        WebGl2RenderingContext::TEXTURE_2D,
        1,
        WebGl2RenderingContext::RGBA8,
        image.width().into(),
        image.height().into(),
    );
    gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        0,
        0,
        image.width().into(),
        image.height().into(),
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(image.data_as_u8_slice()),
    )
    .unwrap();
    let texture_id = TextureId(0);
    let mut texture_bindings = WebGlTextureBindings::new();
    texture_bindings.insert(texture_id, image_texture.clone());
    let mut scene = Scene::new(width, height);

    probe::draw_scene(
        &mut scene,
        ImageSource::external_texture(
            texture_id,
            TextureRegion::Full {
                width: image.width(),
                height: image.height(),
            },
            image.may_have_transparency(),
        ),
        elements,
    );

    renderer
        .render(
            &scene,
            &mut resources,
            &RenderSize { width, height },
            &texture_bindings,
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

    gl.delete_texture(Some(&image_texture));

    Pixmap::from_parts(
        pixels,
        width,
        height,
        PixelMetadata::new(ImageAlphaType::AlphaPremultiplied, true),
    )
}
