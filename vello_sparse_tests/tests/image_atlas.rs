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
use vello_hybrid::{
    AtlasConfig, AtlasId, AtlasTextureInfo, MemorySettings, RenderError, RenderSettings, Scene,
    TextureId, WebGlRenderer, WebGlTextureBindings, WebGlTextureWithDimensions,
};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext, WebGlTexture};

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

fn solid_texture(gl: &WebGl2RenderingContext, pixel: [u8; 4]) -> WebGlTexture {
    let texture = gl.create_texture().unwrap();
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
    gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        WebGl2RenderingContext::RGBA8 as i32,
        1,
        1,
        0,
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(&pixel),
    )
    .unwrap();
    texture
}

fn texture_binding_2d(gl: &WebGl2RenderingContext) -> WebGlTexture {
    gl.get_parameter(WebGl2RenderingContext::TEXTURE_BINDING_2D)
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
    let (mut renderer, mut resources) = WebGlRenderer::new_with(&canvas, settings, true);

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

    renderer.upload_image(&mut resources, &Pixmap::new(2, 2));

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
#[should_panic(expected = "TextureTooLarge")]
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

    let (mut renderer, mut resources) = WebGlRenderer::new_with(&canvas, settings, true);

    // The image is much larger than the 10x10 atlas, so the upload must fail.
    let image = Pixmap::new(64, 64);
    renderer.upload_image(&mut resources, &image);
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
    let (mut renderer, _) = WebGlRenderer::new_with(&canvas, settings, false);
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
            Err(RenderError::TextureFeedbackLoop(id)) if id == texture_id
        ),
        "sampling from the render target should fail"
    );
}

#[wasm_bindgen_test]
fn texture_copy_restores_texture_zero_binding_when_another_unit_is_active() {
    let canvas = create_canvas();
    let settings = RenderSettings {
        memory_settings: MemorySettings {
            image_atlas_config: AtlasConfig {
                initial_atlas_count: 1,
                max_atlases: 1,
                atlas_size: (1, 1),
                ..AtlasConfig::default()
            },
            ..MemorySettings::default()
        },
        ..RenderSettings::default()
    };
    let (mut renderer, mut resources) = WebGlRenderer::new_with(&canvas, settings, true);
    let gl = renderer.gl_context().clone();

    let source = solid_texture(&gl, [255, 0, 0, 255]);
    let texture_zero = solid_texture(&gl, [0, 255, 0, 255]);
    let texture_seven = solid_texture(&gl, [0, 0, 255, 255]);

    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture_zero));
    gl.active_texture(WebGl2RenderingContext::TEXTURE7);
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture_seven));

    renderer.upload_image(
        &mut resources,
        &WebGlTextureWithDimensions {
            texture: source,
            width: 1,
            height: 1,
        },
    );

    assert_eq!(
        gl.get_parameter(WebGl2RenderingContext::ACTIVE_TEXTURE)
            .unwrap()
            .as_f64()
            .unwrap() as u32,
        WebGl2RenderingContext::TEXTURE7,
        "the texture copy should restore the active texture unit",
    );
    assert_eq!(
        texture_binding_2d(&gl),
        texture_seven,
        "the texture copy should preserve the active unit's binding",
    );

    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    assert_eq!(
        texture_binding_2d(&gl),
        texture_zero,
        "the texture copy should preserve texture unit 0's binding",
    );
}
