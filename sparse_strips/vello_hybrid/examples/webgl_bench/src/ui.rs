// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! DOM-based UI controls panel for the benchmark tool.
//!
//! Supports two modes: **Interactive** (tweakable params) and **Benchmark** (automated runs).

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this benchmark"
)]

use crate::harness::BenchResult;
use crate::scenes::{BenchScene, Param, ParamKind};
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element, HtmlElement, HtmlInputElement, HtmlSelectElement};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn doc() -> Document {
    web_sys::window().unwrap().document().unwrap()
}

fn create_div(document: &Document) -> HtmlElement {
    document
        .create_element("div")
        .unwrap()
        .dyn_into::<HtmlElement>()
        .unwrap()
}

fn el_style(el: &Element) -> web_sys::CssStyleDeclaration {
    el.dyn_ref::<HtmlElement>().unwrap().style()
}

fn apply_select_style(select: &HtmlSelectElement) {
    let s = select.style();
    s.set_property("width", "100%").unwrap();
    s.set_property("padding", "4px").unwrap();
    s.set_property("background", "#16213e").unwrap();
    s.set_property("color", "#e0e0e0").unwrap();
    s.set_property("border", "1px solid #333").unwrap();
    s.set_property("font-family", "monospace").unwrap();
}

fn apply_button_style(el: &HtmlElement) {
    let s = el.style();
    s.set_property("padding", "6px 16px").unwrap();
    s.set_property("background", "#16213e").unwrap();
    s.set_property("color", "#7ecfff").unwrap();
    s.set_property("border", "1px solid #444").unwrap();
    s.set_property("border-radius", "4px").unwrap();
    s.set_property("font-family", "monospace").unwrap();
    s.set_property("font-size", "13px").unwrap();
    s.set_property("cursor", "pointer").unwrap();
}

fn format_value(value: f64, step: f64) -> String {
    if step >= 1.0 {
        format!("{}", value as i64)
    } else {
        format!("{value:.1}")
    }
}

// ---------------------------------------------------------------------------
// App mode
// ---------------------------------------------------------------------------

/// Which mode the app is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    /// Free-form interactive exploration.
    Interactive,
    /// Automated benchmark suite.
    Benchmark,
}

// ---------------------------------------------------------------------------
// Param control (used in both modes)
// ---------------------------------------------------------------------------

enum ParamControl {
    Slider(HtmlInputElement),
    Select(HtmlSelectElement),
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

/// The full UI state.
pub struct Ui {
    // Sidebar shell
    sidebar: HtmlElement,
    toggle_btn: HtmlElement,
    collapsed: bool,

    // Shared displays
    fps_label: HtmlElement,
    render_label: HtmlElement,
    viewport_label: HtmlElement,

    // Mode
    /// Current app mode.
    pub mode: AppMode,
    interactive_tab: HtmlElement,
    benchmark_tab: HtmlElement,

    // Interactive pane
    interactive_pane: HtmlElement,
    /// Scene selector dropdown.
    pub scene_select: HtmlSelectElement,
    controls: Vec<(ParamControl, HtmlElement, &'static str)>,

    // Benchmark pane
    benchmark_pane: HtmlElement,
    /// Start benchmarks button.
    pub start_btn: HtmlElement,
    warmup_input: HtmlInputElement,
    run_input: HtmlInputElement,
    bench_status: HtmlElement,
    bench_params_container: HtmlElement,
    results_container: HtmlElement,
}

impl std::fmt::Debug for Ui {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ui")
            .field("mode", &self.mode)
            .field("collapsed", &self.collapsed)
            .finish_non_exhaustive()
    }
}

impl Ui {
    /// Build the full sidebar UI.
    pub fn build(
        document: &Document,
        scenes: &[Box<dyn BenchScene>],
        current_scene: usize,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Self {
        let body = document.body().unwrap();
        let bs = body.style();
        bs.set_property("margin", "0").unwrap();
        bs.set_property("padding", "0").unwrap();
        bs.set_property("overflow", "hidden").unwrap();
        bs.set_property("background-color", "#111").unwrap();

        // --- Sidebar ---
        let sidebar: HtmlElement = document
            .create_element("div")
            .unwrap()
            .dyn_into()
            .unwrap();
        {
            let s = sidebar.style();
            s.set_property("position", "fixed").unwrap();
            s.set_property("top", "0").unwrap();
            s.set_property("left", "0").unwrap();
            s.set_property("width", "300px").unwrap();
            s.set_property("height", "100vh").unwrap();
            s.set_property("background", "rgba(26, 26, 46, 0.88)")
                .unwrap();
            s.set_property("backdrop-filter", "blur(4px)").unwrap();
            s.set_property("color", "#e0e0e0").unwrap();
            s.set_property("font-family", "monospace").unwrap();
            s.set_property("font-size", "13px").unwrap();
            s.set_property("padding", "14px").unwrap();
            s.set_property("box-sizing", "border-box").unwrap();
            s.set_property("overflow-y", "auto").unwrap();
            s.set_property("z-index", "10").unwrap();
            s.set_property("transition", "transform 0.2s ease")
                .unwrap();
        }

        // --- Toggle button ---
        let toggle_btn = create_div(document);
        {
            let s = toggle_btn.style();
            s.set_property("position", "fixed").unwrap();
            s.set_property("top", "8px").unwrap();
            s.set_property("left", "304px").unwrap();
            s.set_property("width", "28px").unwrap();
            s.set_property("height", "28px").unwrap();
            s.set_property("background", "rgba(26, 26, 46, 0.85)")
                .unwrap();
            s.set_property("color", "#7ecfff").unwrap();
            s.set_property("border-radius", "0 4px 4px 0").unwrap();
            s.set_property("cursor", "pointer").unwrap();
            s.set_property("z-index", "11").unwrap();
            s.set_property("display", "flex").unwrap();
            s.set_property("align-items", "center").unwrap();
            s.set_property("justify-content", "center").unwrap();
            s.set_property("font-family", "monospace").unwrap();
            s.set_property("font-size", "16px").unwrap();
            s.set_property("user-select", "none").unwrap();
            s.set_property("transition", "left 0.2s ease").unwrap();
        }
        toggle_btn.set_inner_html("&#x25C0;");
        body.append_child(&toggle_btn).unwrap();

        // --- Title ---
        let title = document.create_element("h3").unwrap();
        title.set_text_content(Some("vello_hybrid bench"));
        let ts = el_style(&title);
        ts.set_property("margin", "0 0 8px 0").unwrap();
        ts.set_property("color", "#7ecfff").unwrap();
        sidebar.append_child(&title).unwrap();

        // --- FPS / render / viewport ---
        let fps_label = create_div(document);
        fps_label.set_text_content(Some("FPS: --"));
        fps_label
            .style()
            .set_property("font-size", "16px")
            .unwrap();
        fps_label
            .style()
            .set_property("font-weight", "bold")
            .unwrap();
        sidebar.append_child(&fps_label).unwrap();

        let render_label = create_div(document);
        render_label.set_text_content(Some("Render: --"));
        render_label
            .style()
            .set_property("color", "#aaa")
            .unwrap();
        sidebar.append_child(&render_label).unwrap();

        let viewport_label = create_div(document);
        viewport_label.set_text_content(Some(&format!("Viewport: {viewport_width} x {viewport_height}")));
        viewport_label
            .style()
            .set_property("color", "#888")
            .unwrap();
        viewport_label
            .style()
            .set_property("margin-bottom", "10px")
            .unwrap();
        sidebar.append_child(&viewport_label).unwrap();

        // --- Mode tabs ---
        let tab_row = create_div(document);
        tab_row
            .style()
            .set_property("display", "flex")
            .unwrap();
        tab_row
            .style()
            .set_property("gap", "0")
            .unwrap();
        tab_row
            .style()
            .set_property("margin-bottom", "12px")
            .unwrap();

        let interactive_tab = create_div(document);
        interactive_tab.set_text_content(Some("Interactive"));
        style_tab(&interactive_tab, true);

        let benchmark_tab = create_div(document);
        benchmark_tab.set_text_content(Some("Benchmark"));
        style_tab(&benchmark_tab, false);

        tab_row.append_child(&interactive_tab).unwrap();
        tab_row.append_child(&benchmark_tab).unwrap();
        sidebar.append_child(&tab_row).unwrap();

        // === Interactive pane ===
        let interactive_pane = create_div(document);

        // Scene selector
        let scene_label = document.create_element("div").unwrap();
        scene_label.set_text_content(Some("Scene:"));
        el_style(&scene_label)
            .set_property("margin-bottom", "4px")
            .unwrap();
        interactive_pane.append_child(&scene_label).unwrap();

        let scene_select: HtmlSelectElement = document
            .create_element("select")
            .unwrap()
            .dyn_into()
            .unwrap();
        apply_select_style(&scene_select);
        scene_select
            .style()
            .set_property("margin-bottom", "12px")
            .unwrap();
        for (i, sc) in scenes.iter().enumerate() {
            let opt = document.create_element("option").unwrap();
            opt.set_text_content(Some(sc.name()));
            opt.set_attribute("value", &i.to_string()).unwrap();
            scene_select.append_child(&opt).unwrap();
        }
        scene_select.set_selected_index(current_scene as i32);
        interactive_pane.append_child(&scene_select).unwrap();

        // Params header
        let ph = document.create_element("div").unwrap();
        ph.set_text_content(Some("--- Parameters ---"));
        el_style(&ph)
            .set_property("margin-bottom", "12px")
            .unwrap();
        el_style(&ph).set_property("color", "#888").unwrap();
        interactive_pane.append_child(&ph).unwrap();

        let controls = build_controls(document, &interactive_pane, &scenes[current_scene].params(), false);

        sidebar.append_child(&interactive_pane).unwrap();

        // === Benchmark pane (hidden initially) ===
        let benchmark_pane = create_div(document);
        benchmark_pane
            .style()
            .set_property("display", "none")
            .unwrap();

        // Warmup / run duration inputs
        let timing_row = |label_text: &str, default: &str| -> HtmlInputElement {
            let row = create_div(document);
            row.style()
                .set_property("margin-bottom", "8px")
                .unwrap();
            let label = document.create_element("span").unwrap();
            label.set_text_content(Some(label_text));
            row.append_child(&label).unwrap();

            let input: HtmlInputElement = document
                .create_element("input")
                .unwrap()
                .dyn_into()
                .unwrap();
            input.set_type("number");
            input.set_value(default);
            let is = input.style();
            is.set_property("width", "80px").unwrap();
            is.set_property("margin-left", "8px").unwrap();
            is.set_property("background", "#16213e").unwrap();
            is.set_property("color", "#e0e0e0").unwrap();
            is.set_property("border", "1px solid #333").unwrap();
            is.set_property("font-family", "monospace").unwrap();
            is.set_property("padding", "2px 4px").unwrap();
            row.append_child(&input).unwrap();

            let ms = document.create_element("span").unwrap();
            ms.set_text_content(Some(" ms"));
            row.append_child(&ms).unwrap();

            benchmark_pane.append_child(&row).unwrap();
            input
        };

        let warmup_input = timing_row("Warmup:", "250");
        let run_input = timing_row("Run:", "1000");

        // Start button
        let start_btn = create_div(document);
        start_btn.set_text_content(Some("Start Benchmarks"));
        apply_button_style(&start_btn);
        start_btn
            .style()
            .set_property("margin", "12px 0")
            .unwrap();
        start_btn
            .style()
            .set_property("text-align", "center")
            .unwrap();
        benchmark_pane.append_child(&start_btn).unwrap();

        // Status line
        let bench_status = create_div(document);
        bench_status
            .style()
            .set_property("margin-bottom", "8px")
            .unwrap();
        bench_status
            .style()
            .set_property("color", "#aaa")
            .unwrap();
        benchmark_pane.append_child(&bench_status).unwrap();

        // Read-only params container (shown during/after a bench)
        let bench_params_container = create_div(document);
        benchmark_pane
            .append_child(&bench_params_container)
            .unwrap();

        // Results container
        let results_container = create_div(document);
        benchmark_pane.append_child(&results_container).unwrap();

        sidebar.append_child(&benchmark_pane).unwrap();

        body.append_child(&sidebar).unwrap();

        Self {
            sidebar,
            toggle_btn,
            collapsed: false,
            fps_label,
            render_label,
            viewport_label,
            mode: AppMode::Interactive,
            interactive_tab,
            benchmark_tab,
            interactive_pane,
            scene_select,
            controls,
            benchmark_pane,
            start_btn,
            warmup_input,
            run_input,
            bench_status,
            bench_params_container,
            results_container,
        }
    }

    // --- Mode switching ---

    /// Switch to the given mode.
    pub fn set_mode(&mut self, mode: AppMode) {
        self.mode = mode;
        match mode {
            AppMode::Interactive => {
                self.interactive_pane
                    .style()
                    .set_property("display", "block")
                    .unwrap();
                self.benchmark_pane
                    .style()
                    .set_property("display", "none")
                    .unwrap();
                style_tab(&self.interactive_tab, true);
                style_tab(&self.benchmark_tab, false);
            }
            AppMode::Benchmark => {
                self.interactive_pane
                    .style()
                    .set_property("display", "none")
                    .unwrap();
                self.benchmark_pane
                    .style()
                    .set_property("display", "block")
                    .unwrap();
                style_tab(&self.interactive_tab, false);
                style_tab(&self.benchmark_tab, true);
            }
        }
    }

    /// Return references to the tab elements for click binding.
    pub fn tab_elements(&self) -> (&HtmlElement, &HtmlElement) {
        (&self.interactive_tab, &self.benchmark_tab)
    }

    // --- Collapse ---

    /// Toggle sidebar visibility.
    pub fn toggle(&mut self) {
        self.collapsed = !self.collapsed;
        if self.collapsed {
            self.sidebar
                .style()
                .set_property("transform", "translateX(-100%)")
                .unwrap();
            self.toggle_btn
                .style()
                .set_property("left", "0")
                .unwrap();
            self.toggle_btn.set_inner_html("&#x25B6;");
        } else {
            self.sidebar
                .style()
                .set_property("transform", "translateX(0)")
                .unwrap();
            self.toggle_btn
                .style()
                .set_property("left", "304px")
                .unwrap();
            self.toggle_btn.set_inner_html("&#x25C0;");
        }
    }

    /// Return a reference to the toggle button element.
    pub fn toggle_btn(&self) -> &HtmlElement {
        &self.toggle_btn
    }

    // --- Timing displays ---

    /// Update the FPS and render time displays.
    pub fn update_timing(&self, fps: f64, frame_time: f64, render_time: f64) {
        self.fps_label.set_text_content(Some(&format!(
            "FPS: {fps:.1}  ({frame_time:.1}ms)"
        )));
        self.render_label.set_text_content(Some(&format!(
            "Render: {render_time:.2}ms"
        )));
    }

    /// Update the viewport size display.
    pub fn update_viewport(&self, width: u32, height: u32) {
        self.viewport_label
            .set_text_content(Some(&format!("Viewport: {width} x {height}")));
    }

    // --- Interactive mode ---

    /// Read the current value of each interactive control.
    pub fn read_params(&self) -> Vec<(&'static str, f64)> {
        self.controls
            .iter()
            .map(|(control, _span, name)| {
                let v: f64 = match control {
                    ParamControl::Slider(input) => input.value().parse().unwrap_or(0.0),
                    ParamControl::Select(select) => select.value().parse().unwrap_or(0.0),
                };
                (*name, v)
            })
            .collect()
    }

    /// Rebuild interactive parameter controls for a new scene.
    pub fn rebuild_params(&mut self, params: &[Param]) {
        remove_controls(&mut self.controls);
        let document = doc();
        self.controls = build_controls(&document, &self.interactive_pane, params, false);
    }

    /// Read the selected scene index.
    pub fn selected_scene(&self) -> usize {
        self.scene_select.selected_index() as usize
    }

    // --- Benchmark mode ---

    /// Read warmup duration from the input.
    pub fn warmup_ms(&self) -> f64 {
        self.warmup_input.value().parse().unwrap_or(250.0)
    }

    /// Read run duration from the input.
    pub fn run_ms(&self) -> f64 {
        self.run_input.value().parse().unwrap_or(1000.0)
    }

    /// Return a reference to the start button.
    pub fn start_btn(&self) -> &HtmlElement {
        &self.start_btn
    }

    /// Show read-only params for a benchmark definition.
    pub fn show_bench_params(&self, params: &[Param]) {
        self.bench_params_container.set_inner_html("");
        let document = doc();

        let header = document.create_element("div").unwrap();
        header.set_text_content(Some("--- Parameters ---"));
        el_style(&header)
            .set_property("color", "#888")
            .unwrap();
        el_style(&header)
            .set_property("margin-bottom", "8px")
            .unwrap();
        self.bench_params_container
            .append_child(&header)
            .unwrap();

        // Use build_controls in read-only mode (we discard the control handles)
        build_controls(&document, &self.bench_params_container, params, true);
    }

    /// Update the benchmark status line.
    pub fn set_bench_status(&self, text: &str) {
        self.bench_status.set_text_content(Some(text));
    }

    /// Display benchmark results.
    pub(crate) fn show_results(&self, results: &[BenchResult]) {
        self.results_container.set_inner_html("");
        let document = doc();

        let header = document.create_element("div").unwrap();
        header.set_text_content(Some("--- Results ---"));
        el_style(&header)
            .set_property("color", "#7ecfff")
            .unwrap();
        el_style(&header)
            .set_property("margin", "12px 0 8px 0")
            .unwrap();
        self.results_container.append_child(&header).unwrap();

        for r in results {
            let row = create_div(&document);
            row.style()
                .set_property("margin-bottom", "6px")
                .unwrap();

            let name_span = document.create_element("div").unwrap();
            name_span.set_text_content(Some(r.name));
            el_style(&name_span)
                .set_property("color", "#e0e0e0")
                .unwrap();
            row.append_child(&name_span).unwrap();

            let detail = document.create_element("div").unwrap();
            detail.set_text_content(Some(&format!(
                "  {:.2}ms/frame  ({} iters, {:.0}ms total)",
                r.time_per_frame_ms, r.iterations, r.total_time_ms
            )));
            el_style(&detail)
                .set_property("color", "#aaa")
                .unwrap();
            el_style(&detail)
                .set_property("font-size", "12px")
                .unwrap();
            row.append_child(&detail).unwrap();

            self.results_container.append_child(&row).unwrap();
        }
    }

    /// Enable or disable the start button and timing inputs.
    pub fn set_bench_running(&self, running: bool) {
        let opacity = if running { "0.5" } else { "1" };
        let events = if running { "none" } else { "auto" };
        self.start_btn
            .style()
            .set_property("opacity", opacity)
            .unwrap();
        self.start_btn
            .style()
            .set_property("pointer-events", events)
            .unwrap();
        self.warmup_input.set_disabled(running);
        self.run_input.set_disabled(running);
    }
}

// ---------------------------------------------------------------------------
// Tab styling
// ---------------------------------------------------------------------------

fn style_tab(el: &HtmlElement, active: bool) {
    let s = el.style();
    s.set_property("flex", "1").unwrap();
    s.set_property("text-align", "center").unwrap();
    s.set_property("padding", "6px 0").unwrap();
    s.set_property("cursor", "pointer").unwrap();
    s.set_property("user-select", "none").unwrap();
    s.set_property("border-bottom", "2px solid").unwrap();
    if active {
        s.set_property("border-color", "#7ecfff").unwrap();
        s.set_property("color", "#7ecfff").unwrap();
    } else {
        s.set_property("border-color", "transparent").unwrap();
        s.set_property("color", "#888").unwrap();
    }
}

// ---------------------------------------------------------------------------
// Build / remove controls
// ---------------------------------------------------------------------------

fn remove_controls(controls: &mut Vec<(ParamControl, HtmlElement, &'static str)>) {
    for (control, _span, _) in controls.drain(..) {
        let el: &Element = match &control {
            ParamControl::Slider(input) => input,
            ParamControl::Select(select) => select,
        };
        if let Some(row) = el.parent_element() {
            row.remove();
        }
    }
}

fn build_controls(
    document: &Document,
    container: &Element,
    params: &[Param],
    read_only: bool,
) -> Vec<(ParamControl, HtmlElement, &'static str)> {
    let mut controls = Vec::new();

    for p in params {
        let row = document.create_element("div").unwrap();
        el_style(&row)
            .set_property("margin-bottom", "10px")
            .unwrap();

        let label = document.create_element("div").unwrap();
        label.set_text_content(Some(p.label));
        el_style(&label)
            .set_property("margin-bottom", "2px")
            .unwrap();
        row.append_child(&label).unwrap();

        let val_span = create_div(document);
        val_span
            .style()
            .set_property("display", "inline")
            .unwrap();
        val_span
            .style()
            .set_property("margin-left", "8px")
            .unwrap();

        let control = match &p.kind {
            ParamKind::Slider { min, max, step } => {
                let input: HtmlInputElement = document
                    .create_element("input")
                    .unwrap()
                    .dyn_into()
                    .unwrap();
                input.set_type("range");
                input.set_min(&min.to_string());
                input.set_max(&max.to_string());
                input.set_step(&step.to_string());
                input.set_value(&p.value.to_string());
                input.set_disabled(read_only);
                input.style().set_property("width", "160px").unwrap();
                input
                    .style()
                    .set_property("vertical-align", "middle")
                    .unwrap();
                row.append_child(&input).unwrap();

                val_span.set_text_content(Some(&format_value(p.value, *step)));
                row.append_child(&val_span).unwrap();

                if !read_only {
                    let val_clone = val_span.clone();
                    let input_clone = input.clone();
                    let step = *step;
                    let closure = Closure::wrap(Box::new(move || {
                        let v: f64 = input_clone.value().parse().unwrap_or(0.0);
                        val_clone.set_text_content(Some(&format_value(v, step)));
                    }) as Box<dyn FnMut()>);
                    input
                        .add_event_listener_with_callback(
                            "input",
                            closure.as_ref().unchecked_ref(),
                        )
                        .unwrap();
                    closure.forget();
                }

                ParamControl::Slider(input)
            }
            ParamKind::Select(options) => {
                let select: HtmlSelectElement = document
                    .create_element("select")
                    .unwrap()
                    .dyn_into()
                    .unwrap();
                apply_select_style(&select);
                select.set_disabled(read_only);

                for &(text, val) in options {
                    let opt = document.create_element("option").unwrap();
                    opt.set_text_content(Some(text));
                    opt.set_attribute("value", &val.to_string()).unwrap();
                    select.append_child(&opt).unwrap();
                }

                let current_idx = options
                    .iter()
                    .position(|&(_, v)| (v - p.value).abs() < f64::EPSILON)
                    .unwrap_or(0);
                select.set_selected_index(current_idx as i32);
                row.append_child(&select).unwrap();

                ParamControl::Select(select)
            }
        };

        container.append_child(&row).unwrap();
        controls.push((control, val_span, p.name));
    }

    controls
}
