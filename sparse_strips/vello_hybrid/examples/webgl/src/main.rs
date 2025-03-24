// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Demonstrates using Vello Hybrid using a WebGL2 backend in the browser.

use webgl::draw_triangle;
use webgl::render_scene;

fn main() {
    let mut scene = vello_hybrid::Scene::new(100, 100);
    draw_triangle(&mut scene);

    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(async move {
        render_scene(scene, 100, 100).await;
    });
}
