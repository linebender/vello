// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for the image/glyph atlas configuration of the WebGL renderer.

use vello_common::{
    geometry::RectU16,
    kurbo::Rect,
    paint::{Image, ImageSource},
    peniko::ImageSampler,
    pixmap::Pixmap,
};
use vello_gpu::{
    AtlasConfig, AtlasId, AtlasTextureInfo, MemorySettings, RenderError, RenderSettings, Scene,
    TextureId, WebGlError, WebGlRenderer, WebGlTextureBindings,
};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

fn create_canvas() -> HtmlCanvasElement {
    web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap()
}

#[wasm_bindgen_test]
fn image_atlas_texture_is_created_on_first_upload() {
    let canvas = create_canvas();
    canvas.set_width(100);
    canvas.set_height(100);

    let atlas_config = AtlasConfig {
        initial_atlas_count: 0,
        atlas_size: (10, 10),
        ..AtlasConfig::default()
    };
    let settings = RenderSettings {
        memory_settings: MemorySettings {
            image_atlas_config: atlas_config,
            ..MemorySettings::default()
        },
        ..RenderSettings::default()
    };
    let (mut renderer, mut resources) = WebGlRenderer::new_with(&canvas, settings, true).unwrap();

    assert_eq!(
        renderer.atlas_info(),
        AtlasTextureInfo {
            width: 10,
            height: 10,
            texture_count: 0,
        },
        "renderer should start without allocated atlas textures"
    );
    assert_eq!(
        renderer.gl_context().get_error(),
        WebGl2RenderingContext::NO_ERROR,
        "renderer initialization should not produce a WebGL error"
    );

    renderer
        .upload_image(&mut resources, &Pixmap::new(2, 2))
        .unwrap();

    assert_eq!(
        renderer.atlas_info(),
        AtlasTextureInfo {
            width: 10,
            height: 10,
            texture_count: 1,
        },
        "first upload should allocate the first configured atlas texture"
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
/// The renderer constructor configures the allocator in the returned resources, which runs the
/// `TextureTooLarge` check.
#[wasm_bindgen_test]
fn image_atlas_upload_larger_than_atlas_fails() {
    let canvas = create_canvas();
    canvas.set_width(100);
    canvas.set_height(100);

    let atlas_config = AtlasConfig {
        atlas_size: (10, 10),
        ..AtlasConfig::default()
    };
    let settings = RenderSettings {
        memory_settings: MemorySettings {
            image_atlas_config: atlas_config,
            ..MemorySettings::default()
        },
        ..RenderSettings::default()
    };

    let (mut renderer, mut resources) = WebGlRenderer::new_with(&canvas, settings, true).unwrap();

    // The image is much larger than the 10x10 atlas, so the upload must fail.
    let image = Pixmap::new(64, 64);
    assert!(
        matches!(
            renderer.upload_image(&mut resources, &image),
            Err(WebGlError::Render(RenderError::AtlasError(_)))
        ),
        "oversized image upload should return an atlas error"
    );
}

#[wasm_bindgen_test]
fn image_atlas_rejects_sampling_from_its_render_target() {
    let canvas = create_canvas();
    let atlas_config = AtlasConfig {
        initial_atlas_count: 1,
        atlas_size: (1, 1),
        ..AtlasConfig::default()
    };
    let settings = RenderSettings {
        memory_settings: MemorySettings {
            image_atlas_config: atlas_config,
            ..MemorySettings::default()
        },
        ..RenderSettings::default()
    };
    let (mut renderer, _) = WebGlRenderer::new_with(&canvas, settings, false).unwrap();
    let texture_id = TextureId(0);
    let mut bindings = WebGlTextureBindings::new();
    bindings.insert(texture_id, renderer.atlas_texture(AtlasId::new(0)).clone());

    let mut scene = Scene::new(1, 1);
    scene.set_paint(Image {
        image: ImageSource::external_texture(texture_id, RectU16::new(0, 0, 1, 1), false),
        sampler: ImageSampler::default(),
    });
    scene.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));

    assert!(
        matches!(
            renderer.render_to_atlas(&scene, 1, atlas_config, AtlasId::new(0), &bindings),
            Err(WebGlError::Render(RenderError::TextureFeedbackLoop(id))) if id == texture_id
        ),
        "sampling from the render target should fail"
    );
}
