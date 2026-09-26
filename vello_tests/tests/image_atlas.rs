// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for the image/glyph atlas of the WebGL renderer.

use glifo::GlyphCacheConfig;
use vello_common::{
    color::palette::css::{GREEN, RED},
    color::{AlphaColor, Srgb},
    geometry::RectU16,
    kurbo::{Affine, Rect},
    paint::{Image, ImageId, ImageSource},
    peniko::ImageSampler,
    pixmap::Pixmap,
};
use vello_gpu::{
    AtlasConfig, AtlasId, AtlasTextureInfo, MemorySettings, RenderError, RenderSettings,
    RenderSize, Resources, Scene, TextureId, WebGlError, WebGlRenderer, WebGlTextureBindings,
};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

use crate::util::layout_glyphs_roboto;

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

const SIZE: u16 = 16;

fn renderer_with_atlas_size(atlas_size: u16) -> (WebGlRenderer, Resources) {
    let canvas = create_canvas();
    canvas.set_width(SIZE.into());
    canvas.set_height(SIZE.into());
    let settings = RenderSettings {
        memory_settings: MemorySettings {
            image_atlas_config: AtlasConfig {
                atlas_size: (atlas_size, atlas_size),
                ..AtlasConfig::default()
            },
            ..MemorySettings::default()
        },
        ..RenderSettings::default()
    };
    WebGlRenderer::new_with(&canvas, settings, false).unwrap()
}

fn upload(
    renderer: &mut WebGlRenderer,
    resources: &mut Resources,
    size: u16,
    color: AlphaColor<Srgb>,
) -> ImageId {
    let mut pixmap = Pixmap::new(size, size);
    pixmap.data_mut().fill(color.premultiply().to_rgba8());
    renderer.upload_image(resources, &pixmap).unwrap()
}

fn render(renderer: &mut WebGlRenderer, resources: &mut Resources, scene: &Scene) {
    let render_size = RenderSize {
        width: SIZE,
        height: SIZE,
    };
    renderer
        .render(
            scene,
            resources,
            &render_size,
            &WebGlTextureBindings::new(),
            AlphaColor::TRANSPARENT,
        )
        .unwrap();
}

/// Reads back `region` of an atlas texture as RGBA8 pixels.
fn read_atlas(renderer: &WebGlRenderer, atlas_id: AtlasId, region: RectU16) -> Vec<u8> {
    let gl = renderer.gl_context();
    let framebuffer = gl.create_framebuffer().unwrap();
    gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&framebuffer));
    gl.framebuffer_texture_2d(
        WebGl2RenderingContext::FRAMEBUFFER,
        WebGl2RenderingContext::COLOR_ATTACHMENT0,
        WebGl2RenderingContext::TEXTURE_2D,
        Some(renderer.atlas_texture(atlas_id)),
        0,
    );
    let mut pixels = vec![0; usize::from(region.width()) * usize::from(region.height()) * 4];
    gl.read_pixels_with_opt_u8_array(
        region.x0.into(),
        region.y0.into(),
        region.width().into(),
        region.height().into(),
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(&mut pixels),
    )
    .unwrap();
    gl.delete_framebuffer(Some(&framebuffer));
    pixels
}

/// `destroy_image` keeps the region allocated until the next render, which clears and frees it.
#[wasm_bindgen_test]
fn image_atlas_destroyed_image_is_cleared_by_next_render() {
    let (mut renderer, mut resources) = renderer_with_atlas_size(64);
    let red = upload(&mut renderer, &mut resources, 8, RED);
    let image = resources.image_cache().get(red).unwrap();
    let (atlas_id, [x, y]) = (image.atlas_id, image.offset);

    resources.destroy_image(red);
    assert!(
        resources.image_cache().get(red).is_some(),
        "the region must stay allocated until the next render"
    );
    render(&mut renderer, &mut resources, &Scene::new(SIZE, SIZE));

    assert!(
        resources.image_cache().get(red).is_none(),
        "the next render must free the region"
    );
    let region = RectU16::new(x, y, x + 8, y + 8);
    assert!(
        read_atlas(&renderer, atlas_id, region)
            .iter()
            .all(|&byte| byte == 0),
        "the freed region must be transparent"
    );
}

/// Clearing evicted glyphs leaves the canvas bound, so the render can be read back.
#[wasm_bindgen_test]
fn image_atlas_glyph_eviction_leaves_canvas_bound() {
    const ATLAS_SIZE: u16 = 256;

    let (mut renderer, mut resources) = renderer_with_atlas_size(ATLAS_SIZE);
    let atlas = RectU16::new(0, 0, ATLAS_SIZE, ATLAS_SIZE);
    let (font, glyphs) = layout_glyphs_roboto("Hello", 12.0);
    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_transform(Affine::translate((0.0, 12.0)));
    scene
        .glyph_run(&mut resources, &font)
        .font_size(12.0)
        .atlas_cache(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
    render(&mut renderer, &mut resources, &scene);
    assert!(
        read_atlas(&renderer, AtlasId::new(0), atlas)
            .iter()
            .any(|&byte| byte != 0),
        "the glyphs must be cached in the atlas"
    );

    let mut green = Scene::new(SIZE, SIZE);
    green.set_paint(GREEN);
    green.fill_rect(&Rect::new(0.0, 0.0, f64::from(SIZE), f64::from(SIZE)));
    // Unused entries are evicted by the first eviction pass after they exceed the maximum age.
    let config = GlyphCacheConfig::default();
    for frame in 0..config.max_entry_age + config.eviction_frequency {
        render(&mut renderer, &mut resources, &green);
        let mut pixel = [0; 4];
        renderer
            .gl_context()
            .read_pixels_with_opt_u8_array(
                0,
                0,
                1,
                1,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                Some(&mut pixel),
            )
            .unwrap();
        assert_eq!(
            pixel,
            GREEN.premultiply().to_rgba8().to_u8_array(),
            "frame {frame} must leave the canvas bound"
        );
    }

    assert!(
        read_atlas(&renderer, AtlasId::new(0), atlas)
            .iter()
            .all(|&byte| byte == 0),
        "evicted glyphs must be cleared"
    );
}
