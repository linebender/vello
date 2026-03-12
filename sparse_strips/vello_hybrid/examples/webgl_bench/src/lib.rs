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
pub(crate) mod storage;
pub mod ui;

use std::cell::RefCell;
use std::rc::Rc;

use fps::FpsTracker;
use harness::{BenchDef, BenchHarness, HarnessEvent, bench_defs};
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
    harness: BenchHarness,
    bench_defs: Vec<BenchDef>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("mode", &self.ui.mode)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl AppState {
    fn tick(&mut self, now: f64) {
        match self.ui.mode {
            AppMode::Interactive => self.tick_interactive(now),
            AppMode::Benchmark => self.tick_benchmark(now),
        }
    }

    fn tick_interactive(&mut self, now: f64) {
        let selected = self.ui.selected_scene();
        if selected != self.current_scene && selected < self.scenes.len() {
            self.current_scene = selected;
            let params = self.scenes[self.current_scene].params();
            self.ui.rebuild_params(&params);
        }

        let params = self.ui.read_params();
        let idx = self.current_scene;
        for (name, value) in params {
            self.scenes[idx].set_param(name, value);
        }

        let perf = web_sys::window().unwrap().performance().unwrap();
        let t0 = perf.now();

        self.scene.reset();
        let (w, h) = (self.width, self.height);
        self.scenes[idx].render(&mut self.scene, w, h, now);
        let rs = vello_hybrid::RenderSize { width: w, height: h };
        self.renderer.render(&self.scene, &rs).unwrap();
        gpu_sync(&self.renderer);

        let render_ms = perf.now() - t0;
        let (fps, frame_time) = self.fps_tracker.frame(now);
        self.ui.update_timing(fps, frame_time, render_ms);
    }

    fn tick_benchmark(&mut self, _now: f64) {
        if !self.harness.is_running() {
            return;
        }

        // Highlight the currently running bench row
        if let Some(idx) = self.harness.current_bench_idx() {
            self.ui.bench_set_running(idx);
        }

        let (w, h) = (self.width, self.height);
        let events = self.harness.tick(
            &self.bench_defs,
            &mut self.scenes,
            &mut self.scene,
            &mut self.renderer,
            w,
            h,
        );

        for event in events {
            match event {
                HarnessEvent::ScreenshotReady => {
                    if let (Some(idx), Ok(url)) =
                        (self.harness.current_bench_idx(), self.canvas.to_data_url())
                    {
                        self.ui.set_screenshot(idx, &url);
                    }
                }
                HarnessEvent::BenchDone(ref result) => {
                    // Find which def index this result belongs to
                    if let Some(idx) = self.bench_defs.iter().position(|d| d.name == result.name) {
                        self.ui.bench_set_done(idx, result);
                    }
                }
                HarnessEvent::AllDone => {
                    self.ui.bench_all_done();
                }
            }
        }
    }
}

fn gpu_sync(renderer: &vello_hybrid::WebGlRenderer) {
    let gl = renderer.gl_context();
    let mut pixel = [0_u8; 4];
    gl.read_pixels_with_opt_u8_array(
        0, 0, 1, 1,
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

    let canvas: HtmlCanvasElement = document.create_element("canvas").unwrap().dyn_into().unwrap();
    canvas.set_width(px_w);
    canvas.set_height(px_h);
    let cs = canvas.style();
    cs.set_property("position", "fixed").unwrap();
    cs.set_property("top", "40px").unwrap(); // below top bar
    cs.set_property("left", "0").unwrap();
    cs.set_property("width", &format!("{css_w}px")).unwrap();
    cs.set_property("height", &format!("{}px", css_h.saturating_sub(40))).unwrap();
    document.body().unwrap().append_child(&canvas).unwrap();

    let bench_scenes = scenes::all_scenes();
    let defs = bench_defs();
    let ui = Ui::build(&document, &bench_scenes, &defs, 0, px_w, px_h);
    let renderer = vello_hybrid::WebGlRenderer::new(&canvas);
    let scene = Scene::new(px_w as u16, px_h as u16);
    let now = performance.now();

    // Canvas starts hidden (Benchmark tab is default).
    canvas.style().set_property("visibility", "hidden").unwrap();

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
        bench_defs: defs,
    }));

    // ── Event handlers ───────────────────────────────────────────────────

    // Sidebar toggle
    {
        let s = state.clone();
        let btn = state.borrow().ui.toggle_btn().clone();
        let cb = Closure::wrap(Box::new(move || s.borrow_mut().ui.toggle_sidebar()) as Box<dyn FnMut()>);
        btn.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref()).unwrap();
        cb.forget();
    }

    // Mode tabs
    {
        let borrow = state.borrow();
        let (itab, btab) = borrow.ui.tab_elements();
        let itab = itab.clone();
        let btab = btab.clone();
        drop(borrow);

        let s = state.clone();
        let canvas_ref = state.borrow().canvas.clone();
        let cb = Closure::wrap(Box::new(move || {
            s.borrow_mut().ui.set_mode(AppMode::Interactive);
            canvas_ref.style().set_property("visibility", "visible").unwrap();
        }) as Box<dyn FnMut()>);
        itab.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref()).unwrap();
        cb.forget();

        let s = state.clone();
        let canvas_ref = state.borrow().canvas.clone();
        let cb = Closure::wrap(Box::new(move || {
            s.borrow_mut().ui.set_mode(AppMode::Benchmark);
            // Canvas stays in DOM for WebGL but is invisible.
            canvas_ref.style().set_property("visibility", "hidden").unwrap();
        }) as Box<dyn FnMut()>);
        btab.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref()).unwrap();
        cb.forget();
    }

    // Start benchmarks
    {
        let s = state.clone();
        let btn = state.borrow().ui.start_btn().clone();
        let cb = Closure::wrap(Box::new(move || {
            let mut st = s.borrow_mut();
            let selected = st.ui.selected_bench_indices();
            if selected.is_empty() {
                return;
            }
            st.harness.warmup_ms = st.ui.warmup_ms();
            st.harness.run_ms = st.ui.run_ms();
            st.ui.bench_started(&selected);
            st.harness.start(selected);
        }) as Box<dyn FnMut()>);
        btn.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref()).unwrap();
        cb.forget();
    }

    // Save button
    {
        let s = state.clone();
        let btn = state.borrow().ui.save_btn().clone();
        let cb = Closure::wrap(Box::new(move || {
            let st = s.borrow();
            st.ui.save_results(&st.bench_defs);
        }) as Box<dyn FnMut()>);
        btn.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref()).unwrap();
        cb.forget();
    }

    // Compare dropdown
    {
        let s = state.clone();
        let sel = state.borrow().ui.compare_select().clone();
        let cb = Closure::wrap(Box::new(move || {
            s.borrow_mut().ui.load_comparison();
        }) as Box<dyn FnMut()>);
        sel.add_event_listener_with_callback("change", cb.as_ref().unchecked_ref()).unwrap();
        cb.forget();
    }

    // Delete report button
    {
        let s = state.clone();
        let btn = state.borrow().ui.delete_btn.clone();
        let cb = Closure::wrap(Box::new(move || {
            let mut st = s.borrow_mut();
            st.ui.delete_selected_report();
        }) as Box<dyn FnMut()>);
        btn.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref()).unwrap();
        cb.forget();
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

    // Window resize
    {
        let s = state.clone();
        let cb = Closure::wrap(Box::new(move |_: web_sys::Event| {
            let mut st = s.borrow_mut();
            let w = web_sys::window().unwrap();
            let dpr = w.device_pixel_ratio();
            let css_w = w.inner_width().unwrap().as_f64().unwrap() as u32;
            let css_h = w.inner_height().unwrap().as_f64().unwrap() as u32;
            let px_w = (css_w as f64 * dpr) as u32;
            let px_h = (css_h as f64 * dpr) as u32;

            st.canvas.set_width(px_w);
            st.canvas.set_height(px_h);
            st.canvas.style().set_property("width", &format!("{css_w}px")).unwrap();
            st.canvas.style().set_property("height", &format!("{}px", css_h.saturating_sub(40))).unwrap();
            st.width = px_w;
            st.height = px_h;
            st.scene = Scene::new(px_w as u16, px_h as u16);
            st.ui.update_viewport(px_w, px_h);
        }) as Box<dyn FnMut(_)>);
        web_sys::window().unwrap().add_event_listener_with_callback("resize", cb.as_ref().unchecked_ref()).unwrap();
        cb.forget();
    }
}
