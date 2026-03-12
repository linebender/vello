// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! DOM-based UI for Interactive and Benchmark modes.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this benchmark"
)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::harness::{BenchDef, BenchResult};
use crate::scenes::{BenchScene, Param, ParamKind};
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element, HtmlElement, HtmlImageElement, HtmlInputElement, HtmlSelectElement};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn doc() -> Document {
    web_sys::window().unwrap().document().unwrap()
}

fn div(d: &Document) -> HtmlElement {
    d.create_element("div").unwrap().dyn_into().unwrap()
}

fn set(el: &HtmlElement, props: &[(&str, &str)]) {
    let s = el.style();
    for &(k, v) in props {
        s.set_property(k, v).unwrap();
    }
}

fn select_style(sel: &HtmlSelectElement) {
    let s = sel.style();
    for &(k, v) in &[
        ("width", "100%"),
        ("padding", "5px 8px"),
        ("background", "#1e1e2e"),
        ("color", "#cdd6f4"),
        ("border", "1px solid #45475a"),
        ("border-radius", "6px"),
        ("font-family", "'JetBrains Mono', monospace"),
        ("font-size", "12px"),
    ] {
        s.set_property(k, v).unwrap();
    }
}

fn format_val(v: f64, step: f64) -> String {
    if step >= 1.0 { format!("{}", v as i64) } else { format!("{v:.1}") }
}

// ── Mode ─────────────────────────────────────────────────────────────────────

/// App mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    /// Interactive exploration.
    Interactive,
    /// Automated benchmarks.
    Benchmark,
}

// ── Param control ────────────────────────────────────────────────────────────

enum ParamCtrl {
    Slider(HtmlInputElement),
    Select(HtmlSelectElement),
}

// ── UI ───────────────────────────────────────────────────────────────────────

/// Full UI state.
pub struct Ui {
    // Layout
    #[allow(dead_code, reason = "kept alive to prevent GC")]
    top_bar: HtmlElement,
    interactive_view: HtmlElement,
    benchmark_view: HtmlElement,

    // Top bar
    tab_interactive: HtmlElement,
    tab_benchmark: HtmlElement,

    // Interactive: sidebar
    sidebar: HtmlElement,
    toggle_btn: HtmlElement,
    sidebar_collapsed: bool,
    fps_label: HtmlElement,
    render_label: HtmlElement,
    viewport_label: HtmlElement,
    /// Scene selector.
    pub scene_select: HtmlSelectElement,
    controls: Vec<(ParamCtrl, HtmlElement, &'static str)>,

    // Benchmark
    warmup_input: HtmlInputElement,
    run_input: HtmlInputElement,
    /// Start button.
    pub start_btn: HtmlElement,
    /// Checkboxes for each bench def (in order of `bench_defs`).
    bench_checkboxes: Vec<HtmlInputElement>,
    /// Row elements for each bench def — used for live status styling.
    bench_rows: Vec<HtmlElement>,
    /// Status indicator elements per row (the left dot/icon).
    bench_status_dots: Vec<HtmlElement>,
    /// Result text elements per row.
    bench_result_texts: Vec<HtmlElement>,
    /// Stored screenshot data URLs per bench (empty string if not yet captured).
    bench_screenshots: Vec<Rc<RefCell<String>>>,
    screenshot_img: HtmlImageElement,

    /// Current mode.
    pub mode: AppMode,
}

impl std::fmt::Debug for Ui {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ui")
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl Ui {
    /// Build the entire UI.
    pub(crate) fn build(
        document: &Document,
        scenes: &[Box<dyn BenchScene>],
        bench_defs: &[BenchDef],
        current_scene: usize,
        vp_w: u32,
        vp_h: u32,
    ) -> Self {
        let body = document.body().unwrap();
        set(
            &body,
            &[
                ("margin", "0"),
                ("padding", "0"),
                ("overflow", "hidden"),
                ("background", "#11111b"),
                ("color", "#cdd6f4"),
                ("font-family", "'JetBrains Mono', ui-monospace, monospace"),
                ("font-size", "13px"),
            ],
        );

        // ── Top bar ──────────────────────────────────────────────────────
        let top_bar = div(document);
        set(&top_bar, &[
            ("position", "fixed"), ("top", "0"), ("left", "0"), ("right", "0"),
            ("height", "40px"), ("background", "rgba(30, 30, 46, 0.95)"),
            ("backdrop-filter", "blur(8px)"),
            ("display", "flex"), ("align-items", "center"),
            ("padding", "0 16px"), ("z-index", "100"),
            ("border-bottom", "1px solid #313244"),
        ]);

        let logo = div(document);
        logo.set_text_content(Some("vello bench"));
        set(&logo, &[("color", "#89b4fa"), ("font-weight", "700"), ("margin-right", "24px"), ("font-size", "14px")]);
        top_bar.append_child(&logo).unwrap();

        let tab_interactive = div(document);
        tab_interactive.set_text_content(Some("Interactive"));
        style_tab(&tab_interactive, true);

        let tab_benchmark = div(document);
        tab_benchmark.set_text_content(Some("Benchmark"));
        style_tab(&tab_benchmark, false);

        top_bar.append_child(&tab_interactive).unwrap();
        top_bar.append_child(&tab_benchmark).unwrap();
        body.append_child(&top_bar).unwrap();

        // ── Interactive view ─────────────────────────────────────────────
        let interactive_view = div(document);
        set(&interactive_view, &[("position", "fixed"), ("top", "40px"), ("left", "0"), ("right", "0"), ("bottom", "0")]);

        // Sidebar
        let sidebar: HtmlElement = div(document);
        set(&sidebar, &[
            ("position", "absolute"), ("top", "0"), ("left", "0"),
            ("width", "280px"), ("height", "100%"),
            ("background", "rgba(30, 30, 46, 0.88)"),
            ("backdrop-filter", "blur(8px)"),
            ("padding", "16px"), ("box-sizing", "border-box"),
            ("overflow-y", "auto"), ("z-index", "10"),
            ("transition", "transform 0.2s ease"),
            ("border-right", "1px solid #313244"),
        ]);

        let toggle_btn = div(document);
        set(&toggle_btn, &[
            ("position", "absolute"), ("top", "8px"), ("left", "284px"),
            ("width", "24px"), ("height", "24px"),
            ("background", "rgba(30, 30, 46, 0.88)"),
            ("color", "#89b4fa"), ("border-radius", "0 6px 6px 0"),
            ("cursor", "pointer"), ("z-index", "11"),
            ("display", "flex"), ("align-items", "center"), ("justify-content", "center"),
            ("font-size", "14px"), ("user-select", "none"),
            ("transition", "left 0.2s ease"),
            ("border", "1px solid #313244"), ("border-left", "none"),
        ]);
        toggle_btn.set_inner_html("&#x25C0;");
        interactive_view.append_child(&toggle_btn).unwrap();

        // FPS / Render / Viewport
        let fps_label = div(document);
        fps_label.set_text_content(Some("FPS: --"));
        set(&fps_label, &[("font-size", "15px"), ("font-weight", "700"), ("color", "#a6e3a1"), ("margin-bottom", "2px")]);
        sidebar.append_child(&fps_label).unwrap();

        let render_label = div(document);
        render_label.set_text_content(Some("Render: --"));
        set(&render_label, &[("color", "#9399b2"), ("margin-bottom", "2px")]);
        sidebar.append_child(&render_label).unwrap();

        let viewport_label = div(document);
        viewport_label.set_text_content(Some(&format!("Viewport: {vp_w} x {vp_h}")));
        set(&viewport_label, &[("color", "#6c7086"), ("margin-bottom", "14px")]);
        sidebar.append_child(&viewport_label).unwrap();

        // Scene selector
        let lbl = div(document);
        lbl.set_text_content(Some("Scene"));
        set(&lbl, &[("color", "#9399b2"), ("margin-bottom", "4px"), ("font-size", "11px"), ("text-transform", "uppercase"), ("letter-spacing", "1px")]);
        sidebar.append_child(&lbl).unwrap();

        let scene_select: HtmlSelectElement = document.create_element("select").unwrap().dyn_into().unwrap();
        select_style(&scene_select);
        set_prop(&scene_select, "margin-bottom", "16px");
        for (i, s) in scenes.iter().enumerate() {
            let opt = document.create_element("option").unwrap();
            opt.set_text_content(Some(s.name()));
            opt.set_attribute("value", &i.to_string()).unwrap();
            scene_select.append_child(&opt).unwrap();
        }
        scene_select.set_selected_index(current_scene as i32);
        sidebar.append_child(&scene_select).unwrap();

        // Separator
        let sep = div(document);
        set(&sep, &[("border-top", "1px solid #313244"), ("margin", "4px 0 12px 0")]);
        sidebar.append_child(&sep).unwrap();

        // Param controls
        let controls = build_controls(document, &sidebar, &scenes[current_scene].params(), false);

        interactive_view.append_child(&sidebar).unwrap();
        body.append_child(&interactive_view).unwrap();

        // ── Benchmark view (hidden initially) ────────────────────────────
        let benchmark_view = div(document);
        set(&benchmark_view, &[
            ("position", "fixed"), ("top", "40px"), ("left", "0"), ("right", "0"), ("bottom", "0"),
            ("display", "none"), ("overflow-y", "auto"),
            ("padding", "32px"), ("box-sizing", "border-box"),
        ]);

        let inner = div(document);
        set(&inner, &[("max-width", "720px"), ("margin", "0 auto")]);

        let title = div(document);
        title.set_text_content(Some("Benchmark Suite"));
        set(&title, &[("font-size", "20px"), ("font-weight", "700"), ("color", "#cdd6f4"), ("margin-bottom", "20px")]);
        inner.append_child(&title).unwrap();

        // Config row
        let config_row = div(document);
        set(&config_row, &[("display", "flex"), ("gap", "16px"), ("margin-bottom", "20px"), ("align-items", "center"), ("flex-wrap", "wrap")]);

        let warmup_input = num_input(document, "Warmup", "250");
        config_row.append_child(&warmup_input.0).unwrap();
        let run_input = num_input(document, "Run", "1000");
        config_row.append_child(&run_input.0).unwrap();

        let start_btn = div(document);
        start_btn.set_text_content(Some("Run Selected"));
        set(&start_btn, &[
            ("padding", "8px 24px"), ("background", "#89b4fa"), ("color", "#1e1e2e"),
            ("border-radius", "8px"), ("font-weight", "700"), ("cursor", "pointer"),
            ("user-select", "none"), ("font-size", "13px"),
            ("transition", "opacity 0.15s"),
        ]);
        config_row.append_child(&start_btn).unwrap();
        inner.append_child(&config_row).unwrap();

        // Screenshot (shown during/after runs)
        let screenshot_img: HtmlImageElement = document.create_element("img").unwrap().dyn_into().unwrap();
        set_prop(&screenshot_img, "max-width", "360px");
        set_prop(&screenshot_img, "border-radius", "10px");
        set_prop(&screenshot_img, "border", "1px solid #313244");
        set_prop(&screenshot_img, "margin-bottom", "16px");
        set_prop(&screenshot_img, "display", "none");
        inner.append_child(&screenshot_img).unwrap();

        // Bench rows — always visible, styled for status
        let mut bench_checkboxes = Vec::new();
        let mut bench_rows = Vec::new();
        let mut bench_status_dots = Vec::new();
        let mut bench_result_texts = Vec::new();
        let mut bench_screenshots: Vec<Rc<RefCell<String>>> = Vec::new();
        let screenshot_img_rc = Rc::new(screenshot_img);

        for def in bench_defs {
            let row = div(document);
            set(&row, &[
                ("background", "#1e1e2e"), ("border", "1px solid #313244"),
                ("border-radius", "10px"), ("padding", "12px 16px"),
                ("margin-bottom", "8px"), ("display", "flex"),
                ("align-items", "center"), ("gap", "12px"),
                ("transition", "border-color 0.3s, background 0.3s"),
            ]);

            // Checkbox
            let cb: HtmlInputElement = document.create_element("input").unwrap().dyn_into().unwrap();
            cb.set_type("checkbox");
            cb.set_checked(true);
            set_prop(&cb, "accent-color", "#89b4fa");
            set_prop(&cb, "width", "16px");
            set_prop(&cb, "height", "16px");
            set_prop(&cb, "cursor", "pointer");
            set_prop(&cb, "flex-shrink", "0");
            row.append_child(&cb).unwrap();

            // Status dot
            let dot = div(document);
            set(&dot, &[
                ("width", "8px"), ("height", "8px"), ("border-radius", "50%"),
                ("background", "#45475a"), ("flex-shrink", "0"),
                ("transition", "background 0.3s"),
            ]);
            row.append_child(&dot).unwrap();

            // Name + params
            let info = div(document);
            set(&info, &[("flex", "1"), ("min-width", "0")]);

            let name_el = div(document);
            name_el.set_text_content(Some(def.name));
            set(&name_el, &[("font-weight", "600"), ("color", "#cdd6f4")]);
            info.append_child(&name_el).unwrap();

            let params_text = def
                .params
                .iter()
                .map(|(k, v)| {
                    if *v == (*v as i64) as f64 {
                        format!("{k}: {}", *v as i64)
                    } else {
                        format!("{k}: {v}")
                    }
                })
                .collect::<Vec<_>>()
                .join("  ·  ");
            let params_el = div(document);
            params_el.set_text_content(Some(&params_text));
            set(&params_el, &[("color", "#6c7086"), ("font-size", "11px"), ("margin-top", "2px")]);
            info.append_child(&params_el).unwrap();

            row.append_child(&info).unwrap();

            // Result text (hidden until done)
            let result_text = div(document);
            set(&result_text, &[("color", "#a6e3a1"), ("font-size", "12px"), ("white-space", "nowrap"), ("display", "none")]);
            row.append_child(&result_text).unwrap();

            // Screenshot storage + click handler
            let screenshot_data: Rc<RefCell<String>> = Rc::new(RefCell::new(String::new()));
            {
                let sd = screenshot_data.clone();
                let img = screenshot_img_rc.clone();
                let row_ref = row.clone();
                set(&row_ref, &[("cursor", "pointer")]);
                let cb = Closure::wrap(Box::new(move || {
                    let url = sd.borrow();
                    if !url.is_empty() {
                        img.set_src(&url);
                        img.style().set_property("display", "block").unwrap();
                    }
                }) as Box<dyn FnMut()>);
                row.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref()).unwrap();
                cb.forget();
            }

            inner.append_child(&row).unwrap();

            bench_checkboxes.push(cb);
            bench_rows.push(row);
            bench_status_dots.push(dot);
            bench_result_texts.push(result_text);
            bench_screenshots.push(screenshot_data);
        }

        benchmark_view.append_child(&inner).unwrap();
        body.append_child(&benchmark_view).unwrap();

        Self {
            top_bar,
            interactive_view,
            benchmark_view,
            tab_interactive,
            tab_benchmark,
            sidebar,
            toggle_btn,
            sidebar_collapsed: false,
            fps_label,
            render_label,
            viewport_label,
            scene_select,
            controls,
            warmup_input: warmup_input.1,
            run_input: run_input.1,
            start_btn,
            bench_checkboxes,
            bench_rows,
            bench_status_dots,
            bench_result_texts,
            bench_screenshots,
            screenshot_img: (*screenshot_img_rc).clone(),
            mode: AppMode::Interactive,
        }
    }

    // ── Mode switching ───────────────────────────────────────────────────

    /// Switch mode.
    pub fn set_mode(&mut self, mode: AppMode) {
        self.mode = mode;
        match mode {
            AppMode::Interactive => {
                self.interactive_view.style().set_property("display", "block").unwrap();
                self.benchmark_view.style().set_property("display", "none").unwrap();
                style_tab(&self.tab_interactive, true);
                style_tab(&self.tab_benchmark, false);
            }
            AppMode::Benchmark => {
                self.interactive_view.style().set_property("display", "none").unwrap();
                self.benchmark_view.style().set_property("display", "block").unwrap();
                style_tab(&self.tab_interactive, false);
                style_tab(&self.tab_benchmark, true);
            }
        }
    }

    /// Tab elements for event binding.
    pub fn tab_elements(&self) -> (&HtmlElement, &HtmlElement) {
        (&self.tab_interactive, &self.tab_benchmark)
    }

    // ── Sidebar toggle ───────────────────────────────────────────────────

    /// Toggle sidebar.
    pub fn toggle_sidebar(&mut self) {
        self.sidebar_collapsed = !self.sidebar_collapsed;
        if self.sidebar_collapsed {
            self.sidebar.style().set_property("transform", "translateX(-100%)").unwrap();
            self.toggle_btn.style().set_property("left", "0").unwrap();
            self.toggle_btn.set_inner_html("&#x25B6;");
        } else {
            self.sidebar.style().set_property("transform", "translateX(0)").unwrap();
            self.toggle_btn.style().set_property("left", "284px").unwrap();
            self.toggle_btn.set_inner_html("&#x25C0;");
        }
    }

    /// Toggle button for event binding.
    pub fn toggle_btn(&self) -> &HtmlElement {
        &self.toggle_btn
    }

    // ── Interactive displays ─────────────────────────────────────────────

    /// Update FPS/render displays.
    pub fn update_timing(&self, fps: f64, frame_time: f64, render_time: f64) {
        self.fps_label.set_text_content(Some(&format!("FPS: {fps:.1}  ({frame_time:.1}ms)")));
        self.render_label.set_text_content(Some(&format!("Render: {render_time:.2}ms")));
    }

    /// Update viewport display.
    pub fn update_viewport(&self, w: u32, h: u32) {
        self.viewport_label.set_text_content(Some(&format!("Viewport: {w} x {h}")));
    }

    /// Read interactive param values.
    pub fn read_params(&self) -> Vec<(&'static str, f64)> {
        self.controls
            .iter()
            .map(|(ctrl, _, name)| {
                let v: f64 = match ctrl {
                    ParamCtrl::Slider(i) => i.value().parse().unwrap_or(0.0),
                    ParamCtrl::Select(s) => s.value().parse().unwrap_or(0.0),
                };
                (*name, v)
            })
            .collect()
    }

    /// Rebuild interactive params.
    pub fn rebuild_params(&mut self, params: &[Param]) {
        for (ctrl, _, _) in self.controls.drain(..) {
            let el: &Element = match &ctrl {
                ParamCtrl::Slider(i) => i,
                ParamCtrl::Select(s) => s,
            };
            if let Some(row) = el.parent_element() {
                row.remove();
            }
        }
        let document = doc();
        self.controls = build_controls(&document, &self.sidebar, params, false);
    }

    /// Selected interactive scene index.
    pub fn selected_scene(&self) -> usize {
        self.scene_select.selected_index() as usize
    }

    // ── Benchmark displays ───────────────────────────────────────────────

    /// Read warmup ms from input.
    pub fn warmup_ms(&self) -> f64 {
        self.warmup_input.value().parse().unwrap_or(250.0)
    }

    /// Read run ms from input.
    pub fn run_ms(&self) -> f64 {
        self.run_input.value().parse().unwrap_or(1000.0)
    }

    /// Start button ref.
    pub fn start_btn(&self) -> &HtmlElement {
        &self.start_btn
    }

    /// Return indices of checked benchmarks.
    pub fn selected_bench_indices(&self) -> Vec<usize> {
        self.bench_checkboxes
            .iter()
            .enumerate()
            .filter(|(_, cb)| cb.checked())
            .map(|(i, _)| i)
            .collect()
    }

    /// Reset all rows to idle state before a run.
    pub fn bench_started(&self, selected: &[usize]) {
        self.screenshot_img.style().set_property("display", "none").unwrap();
        for (i, (row, (dot, result_text))) in self.bench_rows.iter()
            .zip(self.bench_status_dots.iter().zip(self.bench_result_texts.iter()))
            .enumerate()
        {
            result_text.style().set_property("display", "none").unwrap();
            result_text.set_text_content(Some(""));
            if selected.contains(&i) {
                // Pending
                dot.style().set_property("background", "#f9e2af").unwrap(); // yellow
                row.style().set_property("border-color", "#313244").unwrap();
                row.style().set_property("background", "#1e1e2e").unwrap();
            } else {
                // Skipped
                dot.style().set_property("background", "#313244").unwrap();
                row.style().set_property("opacity", "0.4").unwrap();
            }
            // Disable checkboxes during run
            self.bench_checkboxes[i].set_disabled(true);
        }
        self.start_btn.style().set_property("opacity", "0.4").unwrap();
        self.start_btn.style().set_property("pointer-events", "none").unwrap();
    }

    /// Mark a bench as currently running (pulsing blue).
    pub fn bench_set_running(&self, idx: usize) {
        let row = &self.bench_rows[idx];
        let dot = &self.bench_status_dots[idx];
        row.style().set_property("border-color", "#89b4fa").unwrap();
        row.style().set_property("background", "rgba(137, 180, 250, 0.08)").unwrap();
        dot.style().set_property("background", "#89b4fa").unwrap();
    }

    /// Mark a bench as complete with result.
    pub(crate) fn bench_set_done(&self, idx: usize, r: &BenchResult) {
        let row = &self.bench_rows[idx];
        let dot = &self.bench_status_dots[idx];
        let result_text = &self.bench_result_texts[idx];

        row.style().set_property("border-color", "#313244").unwrap();
        row.style().set_property("background", "#1e1e2e").unwrap();
        dot.style().set_property("background", "#a6e3a1").unwrap(); // green

        result_text.set_text_content(Some(&format!(
            "{:.2} ms/f  ({} iters)",
            r.ms_per_frame, r.iterations
        )));
        result_text.style().set_property("display", "block").unwrap();
    }

    /// Show screenshot from data URL and store it for the given bench index.
    pub fn set_screenshot(&self, bench_idx: usize, data_url: &str) {
        self.screenshot_img.set_src(data_url);
        self.screenshot_img.style().set_property("display", "block").unwrap();
        if let Some(slot) = self.bench_screenshots.get(bench_idx) {
            *slot.borrow_mut() = data_url.to_string();
        }
    }

    /// All benchmarks done — re-enable UI.
    pub fn bench_all_done(&self) {
        for (i, cb) in self.bench_checkboxes.iter().enumerate() {
            cb.set_disabled(false);
            self.bench_rows[i].style().set_property("opacity", "1").unwrap();
        }
        self.start_btn.style().set_property("opacity", "1").unwrap();
        self.start_btn.style().set_property("pointer-events", "auto").unwrap();
    }
}

// ── Tab styling ──────────────────────────────────────────────────────────────

fn style_tab(el: &HtmlElement, active: bool) {
    set(el, &[
        ("padding", "8px 16px"), ("cursor", "pointer"), ("user-select", "none"),
        ("font-size", "13px"), ("border-radius", "6px 6px 0 0"),
        ("transition", "color 0.15s, border-color 0.15s"),
        ("border-bottom", "2px solid"),
    ]);
    if active {
        el.style().set_property("color", "#89b4fa").unwrap();
        el.style().set_property("border-color", "#89b4fa").unwrap();
    } else {
        el.style().set_property("color", "#6c7086").unwrap();
        el.style().set_property("border-color", "transparent").unwrap();
    }
}

// ── Number input helper ──────────────────────────────────────────────────────

fn num_input(document: &Document, label: &str, default: &str) -> (HtmlElement, HtmlInputElement) {
    let wrapper = div(document);
    set(&wrapper, &[("display", "flex"), ("align-items", "center"), ("gap", "6px")]);

    let lbl = div(document);
    lbl.set_text_content(Some(label));
    set(&lbl, &[("color", "#9399b2"), ("font-size", "12px")]);
    wrapper.append_child(&lbl).unwrap();

    let input: HtmlInputElement = document.create_element("input").unwrap().dyn_into().unwrap();
    input.set_type("number");
    input.set_value(default);
    set_prop(&input, "width", "70px");
    set_prop(&input, "background", "#1e1e2e");
    set_prop(&input, "color", "#cdd6f4");
    set_prop(&input, "border", "1px solid #45475a");
    set_prop(&input, "border-radius", "6px");
    set_prop(&input, "padding", "4px 8px");
    set_prop(&input, "font-family", "inherit");
    set_prop(&input, "font-size", "12px");
    wrapper.append_child(&input).unwrap();

    let ms = div(document);
    ms.set_text_content(Some("ms"));
    set(&ms, &[("color", "#6c7086"), ("font-size", "11px")]);
    wrapper.append_child(&ms).unwrap();

    (wrapper, input)
}

fn set_prop(el: &impl AsRef<HtmlElement>, k: &str, v: &str) {
    el.as_ref().style().set_property(k, v).unwrap();
}

// ── Param controls ───────────────────────────────────────────────────────────

fn build_controls(
    document: &Document,
    container: &Element,
    params: &[Param],
    read_only: bool,
) -> Vec<(ParamCtrl, HtmlElement, &'static str)> {
    let mut out = Vec::new();

    for p in params {
        let row = div(document);
        set(&row, &[("margin-bottom", "12px")]);

        let label = div(document);
        label.set_text_content(Some(p.label));
        set(&label, &[("color", "#9399b2"), ("margin-bottom", "4px"), ("font-size", "11px"), ("text-transform", "uppercase"), ("letter-spacing", "1px")]);
        row.append_child(&label).unwrap();

        let val_span = div(document);
        set(&val_span, &[("display", "inline"), ("margin-left", "8px"), ("color", "#cdd6f4")]);

        let ctrl = match &p.kind {
            ParamKind::Slider { min, max, step } => {
                let input: HtmlInputElement = document.create_element("input").unwrap().dyn_into().unwrap();
                input.set_type("range");
                input.set_min(&min.to_string());
                input.set_max(&max.to_string());
                input.set_step(&step.to_string());
                input.set_value(&p.value.to_string());
                input.set_disabled(read_only);
                set_prop(&input, "width", "160px");
                set_prop(&input, "vertical-align", "middle");
                set_prop(&input, "accent-color", "#89b4fa");
                row.append_child(&input).unwrap();

                val_span.set_text_content(Some(&format_val(p.value, *step)));
                row.append_child(&val_span).unwrap();

                if !read_only {
                    let vc = val_span.clone();
                    let ic = input.clone();
                    let st = *step;
                    let cb = Closure::wrap(Box::new(move || {
                        let v: f64 = ic.value().parse().unwrap_or(0.0);
                        vc.set_text_content(Some(&format_val(v, st)));
                    }) as Box<dyn FnMut()>);
                    input.add_event_listener_with_callback("input", cb.as_ref().unchecked_ref()).unwrap();
                    cb.forget();
                }

                ParamCtrl::Slider(input)
            }
            ParamKind::Select(options) => {
                let sel: HtmlSelectElement = document.create_element("select").unwrap().dyn_into().unwrap();
                select_style(&sel);
                sel.set_disabled(read_only);
                for &(text, val) in options {
                    let opt = document.create_element("option").unwrap();
                    opt.set_text_content(Some(text));
                    opt.set_attribute("value", &val.to_string()).unwrap();
                    sel.append_child(&opt).unwrap();
                }
                let idx = options.iter().position(|&(_, v)| (v - p.value).abs() < f64::EPSILON).unwrap_or(0);
                sel.set_selected_index(idx as i32);
                row.append_child(&sel).unwrap();

                ParamCtrl::Select(sel)
            }
        };

        container.append_child(&row).unwrap();
        out.push((ctrl, val_span, p.name));
    }

    out
}
