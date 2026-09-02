// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests whether the WebGL example compiles and runs without panicking.
#![cfg(target_arch = "wasm32")]

mod wasm {
    use native_webgl::render_scene;
    use vello_common::peniko::{color::palette, kurbo::BezPath};
    use vello_hybrid::Scene;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_renders_triangle() {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Debug).unwrap();

        let mut scene = Scene::new(100, 100);

        // Draw a blue triangle
        let mut path = BezPath::new();
        path.move_to((30.0, 40.0));
        path.line_to((50.0, 20.0));
        path.line_to((70.0, 40.0));
        path.close_path();
        scene.set_paint(palette::css::BLUE);
        scene.fill_path(&path);

        render_scene(scene, 100, 100).await;
    }
}
