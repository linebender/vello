// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! WebGL benchmark tool for Vello Hybrid.
//!
//! Provides an interactive browser-based benchmark with tweakable parameters,
//! FPS measurement, and multiple benchmark scenes.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this benchmark"
)]
#![cfg(target_arch = "wasm32")]

mod fps;
pub(crate) mod rng;
pub mod scenes;
pub mod ui;

use std::cell::RefCell;
use std::rc::Rc;

use fps::FpsTracker;
use scenes::BenchScene;
use ui::{Ui, SIDEBAR_WIDTH};
use vello_hybrid::Scene;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

type RafClosure = Rc<RefCell<Option<Closure<dyn FnMut()>>>>;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = requestAnimationFrame)]
    fn request_animation_frame(f: &Closure<dyn FnMut()>);
}

/// Main application state.
struct AppState {
    scenes: Vec<Box<dyn BenchScene>>,
    current_scene: usize,
    scene: Scene,
    renderer: vello_hybrid::WebGlRenderer,
    canvas: HtmlCanvasElement,
    width: u32,
    height: u32,
    fps_tracker: FpsTracker,
    ui: Ui,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("current_scene", &self.current_scene)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl AppState {
    fn tick(&mut self, now: f64) {
        // Check for scene change
        let selected = self.ui.selected_scene();
        if selected != self.current_scene && selected < self.scenes.len() {
            self.current_scene = selected;
            let document = web_sys::window().unwrap().document().unwrap();
            let params = self.scenes[self.current_scene].params();
            self.ui.rebuild_params(&document, &params);
        }

        // Read params from sliders and apply
        let params = self.ui.read_params();
        let idx = self.current_scene;
        for (name, value) in params {
            self.scenes[idx].set_param(name, value);
        }

        // Render
        self.scene.reset();
        let (w, h) = (self.width, self.height);
        self.scenes[idx].render(&mut self.scene, w, h, now);

        let render_size = vello_hybrid::RenderSize {
            width: w,
            height: h,
        };
        self.renderer.render(&self.scene, &render_size).unwrap();

        // Update FPS
        let (fps, frame_time) = self.fps_tracker.frame(now);
        self.ui.update_fps(fps, frame_time);
    }
}

/// Entry point: sets up canvas, UI, renderer, and starts the animation loop.
pub async fn run() {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let performance = window.performance().unwrap();
    let dpr = window.device_pixel_ratio();

    let canvas_css_width = window.inner_width().unwrap().as_f64().unwrap() as u32 - SIDEBAR_WIDTH;
    let canvas_css_height = window.inner_height().unwrap().as_f64().unwrap() as u32;
    let canvas_px_width = (canvas_css_width as f64 * dpr) as u32;
    let canvas_px_height = (canvas_css_height as f64 * dpr) as u32;

    // Create canvas
    let canvas: HtmlCanvasElement = document
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap();
    canvas.set_width(canvas_px_width);
    canvas.set_height(canvas_px_height);

    let cs = canvas.style();
    cs.set_property("position", "fixed").unwrap();
    cs.set_property("top", "0").unwrap();
    cs.set_property("left", &format!("{SIDEBAR_WIDTH}px"))
        .unwrap();
    cs.set_property("width", &format!("{canvas_css_width}px"))
        .unwrap();
    cs.set_property("height", &format!("{canvas_css_height}px"))
        .unwrap();

    document.body().unwrap().append_child(&canvas).unwrap();

    // Build scenes and UI
    let bench_scenes = scenes::all_scenes();
    let ui = Ui::build(&document, &bench_scenes, 0, canvas_px_width, canvas_px_height);

    let renderer = vello_hybrid::WebGlRenderer::new(&canvas);
    let scene = Scene::new(canvas_px_width as u16, canvas_px_height as u16);

    let now = performance.now();

    let state = Rc::new(RefCell::new(AppState {
        scenes: bench_scenes,
        current_scene: 0,
        scene,
        renderer,
        canvas,
        width: canvas_px_width,
        height: canvas_px_height,
        fps_tracker: FpsTracker::new(now),
        ui,
    }));

    // Animation loop
    {
        let f: RafClosure = Rc::new(RefCell::new(None));
        let g = f.clone();
        let state = state.clone();

        *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            let now = web_sys::window().unwrap().performance().unwrap().now();
            state.borrow_mut().tick(now);
            request_animation_frame(f.borrow().as_ref().unwrap());
        }) as Box<dyn FnMut()>));

        request_animation_frame(g.borrow().as_ref().unwrap());
    }

    // Window resize handler
    {
        let state = state.clone();
        let closure = Closure::wrap(Box::new(move |_: web_sys::Event| {
            let mut s = state.borrow_mut();
            let window = web_sys::window().unwrap();
            let dpr = window.device_pixel_ratio();

            let css_w = window.inner_width().unwrap().as_f64().unwrap() as u32 - SIDEBAR_WIDTH;
            let css_h = window.inner_height().unwrap().as_f64().unwrap() as u32;
            let px_w = (css_w as f64 * dpr) as u32;
            let px_h = (css_h as f64 * dpr) as u32;

            s.canvas.set_width(px_w);
            s.canvas.set_height(px_h);
            s.canvas
                .style()
                .set_property("width", &format!("{css_w}px"))
                .unwrap();
            s.canvas
                .style()
                .set_property("height", &format!("{css_h}px"))
                .unwrap();

            s.width = px_w;
            s.height = px_h;
            s.scene = Scene::new(px_w as u16, px_h as u16);
            s.ui.update_viewport(px_w, px_h);
        }) as Box<dyn FnMut(_)>);

        web_sys::window()
            .unwrap()
            .add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }
}
