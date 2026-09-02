// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Demonstrates using Vello Hybrid using a WebGL2 backend in the browser.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this demo"
)]

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        use native_webgl::run_interactive;

        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Debug).unwrap();

        let window = web_sys::window().unwrap();
        let dpr = window.device_pixel_ratio();

        let width = window.inner_width().unwrap().as_f64().unwrap() as u16 * dpr as u16;
        let height = window.inner_height().unwrap().as_f64().unwrap() as u16 * dpr as u16;

        wasm_bindgen_futures::spawn_local(async move {
            run_interactive(width, height).await;
        });
    }
}
