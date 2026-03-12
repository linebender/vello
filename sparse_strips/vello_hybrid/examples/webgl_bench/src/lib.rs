// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! WebGL benchmark tool for Vello Hybrid.
//!
//! Two modes:
//! - **Interactive** — tweak parameters in real-time, observe FPS.
//! - **Benchmark** — automated suite with warmup calibration, vsync-independent timing.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this benchmark"
)]
#![cfg(target_arch = "wasm32")]

mod fps;
pub(crate) mod harness;
pub(crate) mod rng;
pub mod scenes;
pub mod ui;

use std::cell::RefCell;
use std::rc::Rc;

use fps::FpsTracker;
use harness::{BenchDef, BenchHarness, bench_defs};
use scenes::BenchScene;
use ui::{AppMode, Ui};
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
    // Benchmark mode
    harness: BenchHarness,
    bench_defs: Vec<BenchDef>,
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
    /// Interactive mode tick.
    fn tick_interactive(&mut self, now: f64) {
        // Check for scene change
        let selected = self.ui.selected_scene();
        if selected != self.current_scene && selected < self.scenes.len() {
            self.current_scene = selected;
            let params = self.scenes[self.current_scene].params();
            self.ui.rebuild_params(&params);
        }

        // Read params from sliders and apply
        let params = self.ui.read_params();
        let idx = self.current_scene;
        for (name, value) in params {
            self.scenes[idx].set_param(name, value);
        }

        // Measure CPU+GPU render time
        let perf = web_sys::window().unwrap().performance().unwrap();
        let render_start = perf.now();

        self.scene.reset();
        let (w, h) = (self.width, self.height);
        self.scenes[idx].render(&mut self.scene, w, h, now);

        let render_size = vello_hybrid::RenderSize {
            width: w,
            height: h,
        };
        self.renderer.render(&self.scene, &render_size).unwrap();
        gpu_sync(&self.renderer);

        let render_ms = perf.now() - render_start;

        let (fps, frame_time) = self.fps_tracker.frame(now);
        self.ui.update_timing(fps, frame_time, render_ms);
    }

    /// Benchmark mode tick.
    fn tick_benchmark(&mut self, now: f64) {
        if !self.harness.is_running() {
            // Still render the last frame so the screen isn't blank
            self.render_current_frame(now);
            return;
        }

        // Update status
        if let Some(idx) = self.harness.current_bench_idx() {
            let total = self.bench_defs.len();
            let def = &self.bench_defs[idx];
            self.ui.set_bench_status(&format!(
                "Running {}/{total}: {}",
                idx + 1,
                def.name
            ));

            // Show read-only params for current bench
            let scene = &self.scenes[def.scene_idx];
            // We need to apply params first so the scene reports correct values
            let mut params = scene.params();
            for &(pname, pval) in def.params {
                if let Some(p) = params.iter_mut().find(|p| p.name == pname) {
                    p.value = pval;
                }
            }
            self.ui.show_bench_params(&params);
        }

        let (w, h) = (self.width, self.height);
        let did_work = self.harness.tick(
            &self.bench_defs,
            &mut self.scenes,
            &mut self.scene,
            &mut self.renderer,
            w,
            h,
        );

        if did_work && self.harness.is_complete() {
            self.ui.set_bench_status("Complete!");
            self.ui.show_results(&self.harness.results);
            self.ui.set_bench_running(false);
        }

        // Update FPS display
        let (fps, frame_time) = self.fps_tracker.frame(now);
        self.ui.update_timing(fps, frame_time, 0.0);
    }

    /// Render one frame of the current interactive scene (used as backdrop in bench mode).
    fn render_current_frame(&mut self, now: f64) {
        self.scene.reset();
        let (w, h) = (self.width, self.height);
        let idx = self.current_scene;
        self.scenes[idx].render(&mut self.scene, w, h, now);
        let render_size = vello_hybrid::RenderSize {
            width: w,
            height: h,
        };
        self.renderer.render(&self.scene, &render_size).unwrap();
    }

    fn tick(&mut self, now: f64) {
        match self.ui.mode {
            AppMode::Interactive => self.tick_interactive(now),
            AppMode::Benchmark => self.tick_benchmark(now),
        }
    }
}

/// Force GPU sync by reading back a single pixel.
fn gpu_sync(renderer: &vello_hybrid::WebGlRenderer) {
    let gl = renderer.gl_context();
    let mut pixel = [0_u8; 4];
    gl.read_pixels_with_opt_u8_array(
        0,
        0,
        1,
        1,
        web_sys::WebGl2RenderingContext::RGBA,
        web_sys::WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(&mut pixel),
    )
    .unwrap();
}

/// Entry point.
pub async fn run() {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let performance = window.performance().unwrap();
    let dpr = window.device_pixel_ratio();

    let css_w = window.inner_width().unwrap().as_f64().unwrap() as u32;
    let css_h = window.inner_height().unwrap().as_f64().unwrap() as u32;
    let px_w = (css_w as f64 * dpr) as u32;
    let px_h = (css_h as f64 * dpr) as u32;

    let canvas: HtmlCanvasElement = document
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap();
    canvas.set_width(px_w);
    canvas.set_height(px_h);
    let cs = canvas.style();
    cs.set_property("position", "fixed").unwrap();
    cs.set_property("top", "0").unwrap();
    cs.set_property("left", "0").unwrap();
    cs.set_property("width", &format!("{css_w}px")).unwrap();
    cs.set_property("height", &format!("{css_h}px")).unwrap();
    document.body().unwrap().append_child(&canvas).unwrap();

    let bench_scenes = scenes::all_scenes();
    let ui = Ui::build(&document, &bench_scenes, 0, px_w, px_h);
    let renderer = vello_hybrid::WebGlRenderer::new(&canvas);
    let scene = Scene::new(px_w as u16, px_h as u16);
    let now = performance.now();

    let state = Rc::new(RefCell::new(AppState {
        scenes: bench_scenes,
        current_scene: 0,
        scene,
        renderer,
        canvas,
        width: px_w,
        height: px_h,
        fps_tracker: FpsTracker::new(now),
        ui,
        harness: BenchHarness::new(),
        bench_defs: bench_defs(),
    }));

    // --- Event handlers ---

    // Toggle button
    {
        let s = state.clone();
        let btn = state.borrow().ui.toggle_btn().clone();
        let closure = Closure::wrap(Box::new(move || {
            s.borrow_mut().ui.toggle();
        }) as Box<dyn FnMut()>);
        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Mode tabs
    {
        let borrow = state.borrow();
        let (itab, btab) = borrow.ui.tab_elements();
        let itab = itab.clone();
        let btab = btab.clone();
        drop(borrow);

        let s = state.clone();
        let closure = Closure::wrap(Box::new(move || {
            s.borrow_mut().ui.set_mode(AppMode::Interactive);
        }) as Box<dyn FnMut()>);
        itab.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();

        let s = state.clone();
        let closure = Closure::wrap(Box::new(move || {
            s.borrow_mut().ui.set_mode(AppMode::Benchmark);
        }) as Box<dyn FnMut()>);
        btab.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Start benchmarks button
    {
        let s = state.clone();
        let btn = state.borrow().ui.start_btn().clone();
        let closure = Closure::wrap(Box::new(move || {
            let mut st = s.borrow_mut();
            st.harness.warmup_ms = st.ui.warmup_ms();
            st.harness.run_ms = st.ui.run_ms();
            st.harness.start();
            st.ui.set_bench_running(true);
            st.ui.set_bench_status("Starting...");
            st.ui.show_results(&[]);
        }) as Box<dyn FnMut()>);
        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Animation loop
    {
        let f: RafClosure = Rc::new(RefCell::new(None));
        let g = f.clone();
        let s = state.clone();
        *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            let now = web_sys::window().unwrap().performance().unwrap().now();
            s.borrow_mut().tick(now);
            request_animation_frame(f.borrow().as_ref().unwrap());
        }) as Box<dyn FnMut()>));
        request_animation_frame(g.borrow().as_ref().unwrap());
    }

    // Window resize handler
    {
        let s = state.clone();
        let closure = Closure::wrap(Box::new(move |_: web_sys::Event| {
            let mut st = s.borrow_mut();
            let window = web_sys::window().unwrap();
            let dpr = window.device_pixel_ratio();
            let css_w = window.inner_width().unwrap().as_f64().unwrap() as u32;
            let css_h = window.inner_height().unwrap().as_f64().unwrap() as u32;
            let px_w = (css_w as f64 * dpr) as u32;
            let px_h = (css_h as f64 * dpr) as u32;

            st.canvas.set_width(px_w);
            st.canvas.set_height(px_h);
            st.canvas
                .style()
                .set_property("width", &format!("{css_w}px"))
                .unwrap();
            st.canvas
                .style()
                .set_property("height", &format!("{css_h}px"))
                .unwrap();
            st.width = px_w;
            st.height = px_h;
            st.scene = Scene::new(px_w as u16, px_h as u16);
            st.ui.update_viewport(px_w, px_h);
        }) as Box<dyn FnMut(_)>);
        web_sys::window()
            .unwrap()
            .add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }
}
