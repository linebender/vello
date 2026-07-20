// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for the image/glyph atlas configuration of the WebGL renderer.

use vello_common::pixmap::Pixmap;
use vello_hybrid::{AtlasConfig, AtlasTextureInfo, RenderSettings, Resources, WebGlRenderer};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

#[wasm_bindgen_test]
fn image_atlas_placeholder_is_promoted_on_first_upload() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(100);
    canvas.set_height(100);

    let atlas_config = AtlasConfig {
        initial_atlas_count: 0,
        atlas_size: (10, 10),
        ..AtlasConfig::default()
    };
    let settings = RenderSettings {
        image_atlas_config: atlas_config,
        ..RenderSettings::default()
    };
    let mut renderer = WebGlRenderer::new_with(&canvas, settings);
    let mut resources = Resources::new_with_config(atlas_config);

    assert_eq!(
        renderer.atlas_info(),
        AtlasTextureInfo {
            width: 1,
            height: 1,
            layer_count: 0,
        },
        "renderer should start with only the 1x1 placeholder atlas texture"
    );
    assert_eq!(
        renderer.gl_context().get_error(),
        WebGl2RenderingContext::NO_ERROR,
        "renderer initialization should not produce a WebGL error"
    );

    renderer.upload_image(&mut resources, &Pixmap::new(2, 2));

    assert_eq!(
        renderer.atlas_info(),
        AtlasTextureInfo {
            width: 10,
            height: 10,
            layer_count: 1,
        },
        "first upload should promote the placeholder to the configured atlas"
    );
    assert_eq!(
        renderer.gl_context().get_error(),
        WebGl2RenderingContext::NO_ERROR,
        "first atlas upload should not produce a WebGL error"
    );
}

/// Uploading an image that is larger than the configured atlas must fail with
/// `AtlasError::TextureTooLarge`.
///
/// The renderer and the resources are both configured with a 10x10 atlas: the renderer sizes the
/// GPU atlas texture, while the resources own the allocator that runs the `TextureTooLarge` check.
#[wasm_bindgen_test]
#[should_panic(expected = "TextureTooLarge")]
fn image_atlas_upload_larger_than_atlas_fails() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(100);
    canvas.set_height(100);

    // Both the renderer and the resources must use the same 10x10 atlas config.
    let atlas_config = AtlasConfig {
        atlas_size: (10, 10),
        ..AtlasConfig::default()
    };
    let settings = RenderSettings {
        image_atlas_config: atlas_config,
        ..RenderSettings::default()
    };

    let mut renderer = WebGlRenderer::new_with(&canvas, settings);
    let mut resources = Resources::new_with_config(atlas_config);

    // The image is much larger than the 10x10 atlas, so the upload must fail.
    let image = Pixmap::new(64, 64);
    renderer.upload_image(&mut resources, &image);
}
