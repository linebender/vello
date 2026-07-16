// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Browser WebGL test asserting that rendering a scene with an image into a 10x10 atlas fails.
//!
//! This is a separate integration-test file (its own wasm module) on purpose: it uses
//! `#[should_panic]`, which does not mix with tests that install `console_error_panic_hook` in
//! the same module. Because wasm is `panic = "abort"` (no unwinding), once a `should_panic` test
//! panics the module is left in a panicking state, and a sibling test's `set_once()` then fails
//! with "cannot modify the panic hook from a panicking thread". Keeping it isolated avoids that.
#![cfg(target_arch = "wasm32")]
#![allow(
    clippy::cast_possible_truncation,
    reason = "canvas dimensions are small and fit in u16"
)]

mod wasm {
    use vello_common::kurbo::{Affine, Rect};
    use vello_common::paint::{Image, ImageSource};
    use vello_common::peniko::{Extend, ImageQuality, ImageSampler};
    use vello_example_scenes::image::ImageScene;
    use vello_hybrid::{AtlasConfig, RenderSettings, RenderSize, Resources, Scene, WebGlRenderer};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_test::*;
    use web_sys::HtmlCanvasElement;

    wasm_bindgen_test_configure!(run_in_browser);

    /// Renders a scene containing an image bundled in the vello repo, using a WebGL renderer
    /// configured with a 10x10 image/glyph atlas.
    ///
    /// The image is much larger than 10x10, so uploading it must fail with
    /// `AtlasError::TextureTooLarge`.
    #[wasm_bindgen_test]
    #[should_panic(expected = "TextureTooLarge")]
    async fn test_renders_image_with_10x10_atlas() {
        let width: u32 = 100;
        let height: u32 = 100;

        // Create a canvas and attach it to the document so a WebGL2 context can be acquired.
        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document
            .create_element("canvas")
            .unwrap()
            .dyn_into::<HtmlCanvasElement>()
            .unwrap();
        canvas.set_width(width);
        canvas.set_height(height);
        document.body().unwrap().append_child(&canvas).unwrap();

        // Configure the renderer and the resources with a 10x10 image/glyph atlas. Both must use
        // the same config: the renderer sizes the GPU atlas texture, while the resources own the
        // allocator that runs the `TextureTooLarge` check.
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

        // Load an image bundled in the vello repo and upload it into the atlas.
        let image = ImageScene::read_flower_image();
        let img_width = image.width() as f64;
        let img_height = image.height() as f64;
        let image_id = renderer.upload_image(&mut resources, &image);

        // Draw the uploaded image, scaled to fit the canvas.
        let mut scene = Scene::new(width as u16, height as u16);
        let scale = (width as f64 / img_width).min(height as f64 / img_height);
        scene.set_transform(Affine::scale(scale));
        scene.set_paint(Image {
            image: ImageSource::opaque_id(image_id),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        });
        scene.fill_rect(&Rect::new(0.0, 0.0, img_width, img_height));

        let render_size = RenderSize { width, height };
        renderer
            .render(&scene, &mut resources, &render_size)
            .unwrap();
    }
}
