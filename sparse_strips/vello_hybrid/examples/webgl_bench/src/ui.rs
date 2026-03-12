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
use crate::storage::BenchReport;
use wasm_bindgen::prelude::*;
use web_sys::{
    Document, Element, HtmlElement, HtmlImageElement, HtmlInputElement, HtmlSelectElement,
};

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
    if step >= 1.0 {
        format!("{}", v as i64)
    } else {
        format!("{v:.1}")
    }
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
    /// Reset view button.
    pub reset_view_btn: HtmlElement,

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
    /// Delta text elements per row (for comparison).
    bench_delta_texts: Vec<HtmlElement>,
    /// Bench names (in order of `bench_defs`).
    bench_names: Vec<&'static str>,
    screenshot_img: HtmlImageElement,

    // Viewport config
    vp_width_input: HtmlInputElement,
    vp_height_input: HtmlInputElement,

    // Save/load
    save_name_input: HtmlInputElement,
    /// Save button.
    pub save_btn: HtmlElement,
    /// Compare dropdown.
    pub compare_select: HtmlSelectElement,
    /// Delete button for saved reports.
    pub delete_btn: HtmlElement,

    /// Currently loaded comparison report (if any).
    compare_report: Option<BenchReport>,

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
        set(
            &top_bar,
            &[
                ("position", "fixed"),
                ("top", "0"),
                ("left", "0"),
                ("right", "0"),
                ("height", "40px"),
                ("background", "rgba(30, 30, 46, 0.95)"),
                ("backdrop-filter", "blur(8px)"),
                ("display", "flex"),
                ("align-items", "center"),
                ("padding", "0 16px"),
                ("z-index", "100"),
                ("border-bottom", "1px solid #313244"),
            ],
        );

        let logo = div(document);
        logo.set_text_content(Some("vello bench"));
        set(
            &logo,
            &[
                ("color", "#89b4fa"),
                ("font-weight", "700"),
                ("margin-right", "24px"),
                ("font-size", "14px"),
            ],
        );
        top_bar.append_child(&logo).unwrap();

        let tab_interactive = div(document);
        tab_interactive.set_text_content(Some("Interactive"));
        style_tab(&tab_interactive, true);

        let tab_benchmark = div(document);
        tab_benchmark.set_text_content(Some("Benchmark"));
        style_tab(&tab_benchmark, false);

        top_bar.append_child(&tab_benchmark).unwrap();
        top_bar.append_child(&tab_interactive).unwrap();
        body.append_child(&top_bar).unwrap();

        // ── Interactive view ─────────────────────────────────────────────
        let interactive_view = div(document);
        set(
            &interactive_view,
            &[
                ("position", "fixed"),
                ("top", "40px"),
                ("left", "0"),
                ("right", "0"),
                ("bottom", "0"),
            ],
        );

        // Sidebar
        let sidebar: HtmlElement = div(document);
        set(
            &sidebar,
            &[
                ("position", "absolute"),
                ("top", "0"),
                ("left", "0"),
                ("width", "280px"),
                ("height", "100%"),
                ("background", "rgba(30, 30, 46, 0.88)"),
                ("backdrop-filter", "blur(8px)"),
                ("padding", "16px"),
                ("box-sizing", "border-box"),
                ("overflow-y", "auto"),
                ("z-index", "10"),
                ("transition", "transform 0.2s ease"),
                ("border-right", "1px solid #313244"),
            ],
        );

        let toggle_btn = div(document);
        set(
            &toggle_btn,
            &[
                ("position", "absolute"),
                ("top", "8px"),
                ("left", "284px"),
                ("width", "24px"),
                ("height", "24px"),
                ("background", "rgba(30, 30, 46, 0.88)"),
                ("color", "#89b4fa"),
                ("border-radius", "0 6px 6px 0"),
                ("cursor", "pointer"),
                ("z-index", "11"),
                ("display", "flex"),
                ("align-items", "center"),
                ("justify-content", "center"),
                ("font-size", "14px"),
                ("user-select", "none"),
                ("transition", "left 0.2s ease"),
                ("border", "1px solid #313244"),
                ("border-left", "none"),
            ],
        );
        toggle_btn.set_inner_html("&#x25C0;");
        interactive_view.append_child(&toggle_btn).unwrap();

        // FPS / Render / Viewport
        let fps_label = div(document);
        fps_label.set_text_content(Some("FPS: --"));
        set(
            &fps_label,
            &[
                ("font-size", "15px"),
                ("font-weight", "700"),
                ("color", "#a6e3a1"),
                ("margin-bottom", "2px"),
            ],
        );
        sidebar.append_child(&fps_label).unwrap();

        let render_label = div(document);
        render_label.set_text_content(Some("Render: --"));
        set(
            &render_label,
            &[("color", "#9399b2"), ("margin-bottom", "2px")],
        );
        sidebar.append_child(&render_label).unwrap();

        let viewport_label = div(document);
        viewport_label.set_text_content(Some(&format!("Viewport: {vp_w} x {vp_h}")));
        set(
            &viewport_label,
            &[("color", "#6c7086"), ("margin-bottom", "14px")],
        );
        sidebar.append_child(&viewport_label).unwrap();

        // Scene selector
        let lbl = div(document);
        lbl.set_text_content(Some("Scene"));
        set(
            &lbl,
            &[
                ("color", "#9399b2"),
                ("margin-bottom", "4px"),
                ("font-size", "11px"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "1px"),
            ],
        );
        sidebar.append_child(&lbl).unwrap();

        let scene_select: HtmlSelectElement = document
            .create_element("select")
            .unwrap()
            .dyn_into()
            .unwrap();
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
        set(
            &sep,
            &[
                ("border-top", "1px solid #313244"),
                ("margin", "4px 0 12px 0"),
            ],
        );
        sidebar.append_child(&sep).unwrap();

        // Param controls
        let controls = build_controls(document, &sidebar, &scenes[current_scene].params(), false);

        // Reset View button
        let reset_view_btn = div(document);
        reset_view_btn.set_text_content(Some("Reset View"));
        set(
            &reset_view_btn,
            &[
                ("margin-top", "12px"),
                ("padding", "6px 12px"),
                ("background", "#313244"),
                ("color", "#cdd6f4"),
                ("border-radius", "6px"),
                ("cursor", "pointer"),
                ("text-align", "center"),
                ("font-size", "12px"),
                ("user-select", "none"),
                ("display", "none"),
            ],
        );
        sidebar.append_child(&reset_view_btn).unwrap();

        interactive_view.append_child(&sidebar).unwrap();
        body.append_child(&interactive_view).unwrap();

        // ── Benchmark view (hidden initially) ────────────────────────────
        let benchmark_view = div(document);
        set(
            &benchmark_view,
            &[
                ("position", "fixed"),
                ("top", "40px"),
                ("left", "0"),
                ("right", "0"),
                ("bottom", "0"),
                ("display", "none"),
                ("overflow-y", "auto"),
                ("padding", "16px 16px 16px 12px"),
                ("box-sizing", "border-box"),
            ],
        );

        // Two-column layout: left sidebar (config + screenshot), right main (bench rows)
        let bench_layout = div(document);
        set(
            &bench_layout,
            &[
                ("display", "flex"),
                ("gap", "16px"),
                ("align-items", "flex-start"),
            ],
        );

        // ── Left column: config + screenshot below ──────────────────────
        let left_wrapper = div(document);
        set(&left_wrapper, &[("width", "240px"), ("flex-shrink", "0")]);

        let left_col = div(document);
        set(
            &left_col,
            &[
                ("background", "#1e1e2e"),
                ("border", "1px solid #313244"),
                ("border-radius", "12px"),
                ("padding", "16px"),
                ("box-sizing", "border-box"),
            ],
        );

        // Section: Run config
        let section_label = |doc: &Document, text: &str| -> HtmlElement {
            let el = div(doc);
            el.set_text_content(Some(text));
            set(
                &el,
                &[
                    ("color", "#9399b2"),
                    ("font-size", "10px"),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "1.5px"),
                    ("margin-bottom", "8px"),
                    ("font-weight", "600"),
                ],
            );
            el
        };

        left_col
            .append_child(&section_label(document, "Run Config"))
            .unwrap();

        let warmup_input = num_input(document, "Warmup", "250");
        warmup_input
            .0
            .style()
            .set_property("margin-bottom", "6px")
            .unwrap();
        left_col.append_child(&warmup_input.0).unwrap();
        let run_input = num_input(document, "Run", "1000");
        run_input
            .0
            .style()
            .set_property("margin-bottom", "12px")
            .unwrap();
        left_col.append_child(&run_input.0).unwrap();

        // Viewport
        left_col
            .append_child(&section_label(document, "Viewport"))
            .unwrap();

        let vp_row = div(document);
        set(
            &vp_row,
            &[
                ("display", "flex"),
                ("gap", "6px"),
                ("margin-bottom", "16px"),
                ("align-items", "center"),
            ],
        );
        let vp_width_input = sized_num_input(document, &vp_w.to_string(), "70px");
        vp_row.append_child(&vp_width_input).unwrap();
        let x_label = div(document);
        x_label.set_text_content(Some("x"));
        set(&x_label, &[("color", "#6c7086")]);
        vp_row.append_child(&x_label).unwrap();
        let vp_height_input = sized_num_input(document, &vp_h.to_string(), "70px");
        vp_row.append_child(&vp_height_input).unwrap();
        let px_label = div(document);
        px_label.set_text_content(Some("px"));
        set(&px_label, &[("color", "#6c7086"), ("font-size", "11px")]);
        vp_row.append_child(&px_label).unwrap();
        left_col.append_child(&vp_row).unwrap();

        // Start button (full width)
        let start_btn = div(document);
        start_btn.set_text_content(Some("Run Selected"));
        set(
            &start_btn,
            &[
                ("padding", "10px 0"),
                ("background", "#89b4fa"),
                ("color", "#1e1e2e"),
                ("border-radius", "8px"),
                ("font-weight", "700"),
                ("cursor", "pointer"),
                ("user-select", "none"),
                ("font-size", "13px"),
                ("text-align", "center"),
                ("transition", "opacity 0.15s"),
                ("margin-bottom", "16px"),
            ],
        );
        left_col.append_child(&start_btn).unwrap();

        // Separator
        let sep = div(document);
        set(
            &sep,
            &[
                ("border-top", "1px solid #313244"),
                ("margin-bottom", "16px"),
            ],
        );
        left_col.append_child(&sep).unwrap();

        // Save/load section
        left_col
            .append_child(&section_label(document, "Reports"))
            .unwrap();

        let save_name_input = sized_num_input(document, "baseline", "100%");
        save_name_input.set_type("text");
        save_name_input.set_placeholder("Report name");
        save_name_input
            .style()
            .set_property("margin-bottom", "8px")
            .unwrap();
        save_name_input
            .style()
            .set_property("box-sizing", "border-box")
            .unwrap();
        left_col.append_child(&save_name_input).unwrap();

        let save_btn = div(document);
        save_btn.set_text_content(Some("Save"));
        set(
            &save_btn,
            &[
                ("padding", "7px 0"),
                ("background", "#a6e3a1"),
                ("color", "#1e1e2e"),
                ("border-radius", "6px"),
                ("font-weight", "700"),
                ("cursor", "pointer"),
                ("user-select", "none"),
                ("font-size", "12px"),
                ("text-align", "center"),
                ("margin-bottom", "12px"),
            ],
        );
        left_col.append_child(&save_btn).unwrap();

        let compare_label = div(document);
        compare_label.set_text_content(Some("Compare with"));
        set(
            &compare_label,
            &[
                ("color", "#9399b2"),
                ("font-size", "11px"),
                ("margin-bottom", "4px"),
            ],
        );
        left_col.append_child(&compare_label).unwrap();

        let compare_select: HtmlSelectElement = document
            .create_element("select")
            .unwrap()
            .dyn_into()
            .unwrap();
        select_style(&compare_select);
        compare_select
            .style()
            .set_property("margin-bottom", "8px")
            .unwrap();
        {
            let opt = document.create_element("option").unwrap();
            opt.set_text_content(Some("(none)"));
            opt.set_attribute("value", "").unwrap();
            compare_select.append_child(&opt).unwrap();
        }
        let saved = crate::storage::load_reports();
        for (i, r) in saved.reports.iter().enumerate() {
            let opt = document.create_element("option").unwrap();
            let label = format!("{} ({}x{})", r.label, r.viewport_width, r.viewport_height);
            opt.set_text_content(Some(&label));
            opt.set_attribute("value", &i.to_string()).unwrap();
            compare_select.append_child(&opt).unwrap();
        }
        left_col.append_child(&compare_select).unwrap();

        let delete_btn = div(document);
        delete_btn.set_text_content(Some("Delete Selected Report"));
        set(
            &delete_btn,
            &[
                ("padding", "6px 0"),
                ("background", "#45475a"),
                ("color", "#f38ba8"),
                ("border-radius", "6px"),
                ("font-weight", "600"),
                ("cursor", "pointer"),
                ("user-select", "none"),
                ("font-size", "11px"),
                ("text-align", "center"),
            ],
        );
        left_col.append_child(&delete_btn).unwrap();

        left_wrapper.append_child(&left_col).unwrap();

        // Screenshot (below config card, full left-column width)
        let screenshot_img: HtmlImageElement =
            document.create_element("img").unwrap().dyn_into().unwrap();
        set_prop(&screenshot_img, "width", "100%");
        set_prop(&screenshot_img, "border-radius", "8px");
        set_prop(&screenshot_img, "border", "1px solid #313244");
        set_prop(&screenshot_img, "margin-top", "12px");
        set_prop(&screenshot_img, "display", "none");
        left_wrapper.append_child(&screenshot_img).unwrap();

        bench_layout.append_child(&left_wrapper).unwrap();

        // ── Right column: bench rows ────────────────────────────────────
        let inner = div(document);
        set(&inner, &[("flex", "1"), ("min-width", "0")]);

        // Bench rows — always visible, styled for status
        let mut bench_checkboxes = Vec::new();
        let mut bench_rows = Vec::new();
        let mut bench_status_dots = Vec::new();
        let mut bench_result_texts = Vec::new();
        let mut bench_screenshots: Vec<Rc<RefCell<String>>> = Vec::new();
        let mut bench_delta_texts = Vec::new();
        let mut bench_names: Vec<&'static str> = Vec::new();
        let screenshot_img_rc = Rc::new(screenshot_img);

        for def in bench_defs {
            let row = div(document);
            set(
                &row,
                &[
                    ("background", "#1e1e2e"),
                    ("border", "1px solid #313244"),
                    ("border-radius", "10px"),
                    ("padding", "12px 16px"),
                    ("margin-bottom", "8px"),
                    ("display", "flex"),
                    ("align-items", "center"),
                    ("gap", "12px"),
                    ("transition", "border-color 0.3s, background 0.3s"),
                ],
            );

            // Checkbox
            let cb: HtmlInputElement = document
                .create_element("input")
                .unwrap()
                .dyn_into()
                .unwrap();
            cb.set_type("checkbox");
            cb.set_checked(true);
            set_prop(&cb, "accent-color", "#89b4fa");
            set_prop(&cb, "width", "16px");
            set_prop(&cb, "height", "16px");
            set_prop(&cb, "cursor", "pointer");
            set_prop(&cb, "flex-shrink", "0");
            // Prevent checkbox clicks from triggering the row's screenshot handler.
            {
                let stop = Closure::wrap(Box::new(move |e: web_sys::Event| {
                    e.stop_propagation();
                }) as Box<dyn FnMut(_)>);
                cb.add_event_listener_with_callback("click", stop.as_ref().unchecked_ref())
                    .unwrap();
                stop.forget();
            }
            row.append_child(&cb).unwrap();

            // Status dot
            let dot = div(document);
            set(
                &dot,
                &[
                    ("width", "8px"),
                    ("height", "8px"),
                    ("border-radius", "50%"),
                    ("background", "#45475a"),
                    ("flex-shrink", "0"),
                    ("transition", "background 0.3s"),
                ],
            );
            row.append_child(&dot).unwrap();

            // Name + params
            let info = div(document);
            set(&info, &[("flex", "1"), ("min-width", "0")]);

            let name_el = div(document);
            name_el.set_text_content(Some(def.name));
            set(&name_el, &[("font-weight", "600"), ("color", "#cdd6f4")]);
            info.append_child(&name_el).unwrap();

            let params_el = div(document);
            params_el.set_text_content(Some(def.description));
            set(
                &params_el,
                &[
                    ("color", "#6c7086"),
                    ("font-size", "11px"),
                    ("margin-top", "2px"),
                ],
            );
            info.append_child(&params_el).unwrap();

            row.append_child(&info).unwrap();

            // Result text (hidden until done)
            let result_text = div(document);
            set(
                &result_text,
                &[
                    ("color", "#a6e3a1"),
                    ("font-size", "12px"),
                    ("white-space", "nowrap"),
                    ("display", "none"),
                ],
            );
            row.append_child(&result_text).unwrap();

            // Delta text (for comparison, hidden until populated)
            let delta_text = div(document);
            set(
                &delta_text,
                &[
                    ("font-size", "12px"),
                    ("white-space", "nowrap"),
                    ("display", "none"),
                    ("font-weight", "600"),
                ],
            );
            row.append_child(&delta_text).unwrap();

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
                row.add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())
                    .unwrap();
                cb.forget();
            }

            inner.append_child(&row).unwrap();

            bench_checkboxes.push(cb);
            bench_rows.push(row);
            bench_status_dots.push(dot);
            bench_result_texts.push(result_text);
            bench_screenshots.push(screenshot_data);
            bench_delta_texts.push(delta_text);
            bench_names.push(def.name);
        }

        bench_layout.append_child(&inner).unwrap();
        benchmark_view.append_child(&bench_layout).unwrap();
        body.append_child(&benchmark_view).unwrap();

        let mut ui = Self {
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
            reset_view_btn,
            warmup_input: warmup_input.1,
            run_input: run_input.1,
            start_btn,
            bench_checkboxes,
            bench_rows,
            bench_status_dots,
            bench_result_texts,
            bench_screenshots,
            bench_delta_texts,
            bench_names,
            screenshot_img: (*screenshot_img_rc).clone(),
            vp_width_input,
            vp_height_input,
            save_name_input,
            save_btn,
            compare_select,
            delete_btn,
            compare_report: None,
            mode: AppMode::Benchmark,
        };
        // Start in Benchmark mode.
        ui.set_mode(AppMode::Benchmark);
        ui
    }

    // ── Mode switching ───────────────────────────────────────────────────

    /// Switch mode.
    pub fn set_mode(&mut self, mode: AppMode) {
        self.mode = mode;
        match mode {
            AppMode::Interactive => {
                self.interactive_view
                    .style()
                    .set_property("display", "block")
                    .unwrap();
                self.benchmark_view
                    .style()
                    .set_property("display", "none")
                    .unwrap();
                style_tab(&self.tab_interactive, true);
                style_tab(&self.tab_benchmark, false);
            }
            AppMode::Benchmark => {
                self.interactive_view
                    .style()
                    .set_property("display", "none")
                    .unwrap();
                self.benchmark_view
                    .style()
                    .set_property("display", "block")
                    .unwrap();
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
            self.sidebar
                .style()
                .set_property("transform", "translateX(-100%)")
                .unwrap();
            self.toggle_btn.style().set_property("left", "0").unwrap();
            self.toggle_btn.set_inner_html("&#x25B6;");
        } else {
            self.sidebar
                .style()
                .set_property("transform", "translateX(0)")
                .unwrap();
            self.toggle_btn
                .style()
                .set_property("left", "284px")
                .unwrap();
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
        self.fps_label
            .set_text_content(Some(&format!("FPS: {fps:.1}  ({frame_time:.1}ms)")));
        self.render_label
            .set_text_content(Some(&format!("Render: {render_time:.2}ms")));
    }

    /// Update viewport display.
    pub fn update_viewport(&self, w: u32, h: u32) {
        self.viewport_label
            .set_text_content(Some(&format!("Viewport: {w} x {h}")));
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
        self.screenshot_img
            .style()
            .set_property("display", "none")
            .unwrap();
        for (i, (row, (dot, result_text))) in self
            .bench_rows
            .iter()
            .zip(
                self.bench_status_dots
                    .iter()
                    .zip(self.bench_result_texts.iter()),
            )
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
        self.start_btn
            .style()
            .set_property("opacity", "0.4")
            .unwrap();
        self.start_btn
            .style()
            .set_property("pointer-events", "none")
            .unwrap();
    }

    /// Mark a bench as currently running — prominent red-tinted card.
    pub fn bench_set_running(&self, idx: usize) {
        let row = &self.bench_rows[idx];
        let dot = &self.bench_status_dots[idx];
        row.style().set_property("border-color", "#f38ba8").unwrap();
        row.style()
            .set_property("background", "rgba(243, 139, 168, 0.15)")
            .unwrap();
        dot.style().set_property("background", "#f38ba8").unwrap();
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
        result_text
            .style()
            .set_property("display", "block")
            .unwrap();
        self.show_delta_for(idx, r.ms_per_frame);
    }

    /// Show screenshot from data URL and store it for the given bench index.
    pub fn set_screenshot(&self, bench_idx: usize, data_url: &str) {
        self.screenshot_img.set_src(data_url);
        self.screenshot_img
            .style()
            .set_property("display", "block")
            .unwrap();
        if let Some(slot) = self.bench_screenshots.get(bench_idx) {
            *slot.borrow_mut() = data_url.to_string();
        }
    }

    /// All benchmarks done — re-enable UI and show deltas if comparison loaded.
    pub fn bench_all_done(&self) {
        for (i, cb) in self.bench_checkboxes.iter().enumerate() {
            cb.set_disabled(false);
            self.bench_rows[i]
                .style()
                .set_property("opacity", "1")
                .unwrap();
        }
        self.start_btn.style().set_property("opacity", "1").unwrap();
        self.start_btn
            .style()
            .set_property("pointer-events", "auto")
            .unwrap();
        self.show_deltas();
    }

    // ── Save / Load / Compare ─────────────────────────────────────────

    /// Save current benchmark results to localStorage.
    pub(crate) fn save_results(&self, bench_defs: &[BenchDef]) {
        let label = self.save_name_input.value();
        let label = label.trim().to_string();
        if label.is_empty() {
            return;
        }
        // Prevent duplicate names — overwrite existing report with same name.
        let store = crate::storage::load_reports();
        if let Some(idx) = store.reports.iter().position(|r| r.label == label) {
            crate::storage::delete_report(idx);
        }
        let vp_w: u32 = self.vp_width_input.value().parse().unwrap_or(0);
        let vp_h: u32 = self.vp_height_input.value().parse().unwrap_or(0);

        let mut results = Vec::new();
        for (i, rt) in self.bench_result_texts.iter().enumerate() {
            let text = rt.text_content().unwrap_or_default();
            if text.is_empty() {
                continue;
            }
            // Parse "X.XX ms/f  (N iters)"
            if let Some(ms_str) = text.split(" ms/f").next()
                && let Ok(ms) = ms_str.trim().parse::<f64>()
            {
                let iters = text
                    .split('(')
                    .nth(1)
                    .and_then(|s| s.split(' ').next())
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                results.push(crate::storage::SavedResult {
                    name: bench_defs[i].name.to_string(),
                    ms_per_frame: ms,
                    iterations: iters,
                });
            }
        }

        if results.is_empty() {
            return;
        }

        crate::storage::save_report(BenchReport {
            label,
            viewport_width: vp_w,
            viewport_height: vp_h,
            results,
        });

        self.refresh_compare_dropdown();
    }

    /// Refresh the compare dropdown with current saved reports.
    pub fn refresh_compare_dropdown(&self) {
        // Clear existing options.
        self.compare_select.set_inner_html("");
        let d = doc();
        let none_opt = d.create_element("option").unwrap();
        none_opt.set_text_content(Some("(none)"));
        none_opt.set_attribute("value", "").unwrap();
        self.compare_select.append_child(&none_opt).unwrap();

        let saved = crate::storage::load_reports();
        for (i, r) in saved.reports.iter().enumerate() {
            let opt = d.create_element("option").unwrap();
            let lbl = format!("{} ({}x{})", r.label, r.viewport_width, r.viewport_height);
            opt.set_text_content(Some(&lbl));
            opt.set_attribute("value", &i.to_string()).unwrap();
            self.compare_select.append_child(&opt).unwrap();
        }
    }

    /// Load a comparison report by index, or clear if empty.
    pub fn load_comparison(&mut self) {
        let val = self.compare_select.value();
        if val.is_empty() {
            self.compare_report = None;
            self.hide_deltas();
            return;
        }
        let idx: usize = match val.parse() {
            Ok(i) => i,
            Err(_) => {
                self.compare_report = None;
                self.hide_deltas();
                return;
            }
        };
        let store = crate::storage::load_reports();
        if let Some(report) = store.reports.get(idx).cloned() {
            self.compare_report = Some(report);
            self.show_deltas();
        }
    }

    /// Show delta for a single bench row given its current ms/frame.
    fn show_delta_for(&self, idx: usize, cur_ms: f64) {
        let Some(ref report) = self.compare_report else {
            return;
        };
        let delta_el = &self.bench_delta_texts[idx];
        let name = self.bench_names[idx];
        let Some(base) = report
            .results
            .iter()
            .find(|r| r.name == name)
            .map(|r| r.ms_per_frame)
        else {
            delta_el.style().set_property("display", "none").unwrap();
            return;
        };
        let pct = ((cur_ms - base) / base) * 100.0;
        let abs_pct = pct.abs();
        if abs_pct < 5.0 {
            delta_el.set_text_content(Some(&format!("{pct:+.1}%")));
            delta_el.style().set_property("color", "#6c7086").unwrap();
        } else if pct < 0.0 {
            delta_el.set_text_content(Some(&format!("{pct:+.1}%")));
            delta_el.style().set_property("color", "#a6e3a1").unwrap();
        } else {
            delta_el.set_text_content(Some(&format!("+{pct:.1}%")));
            delta_el.style().set_property("color", "#f38ba8").unwrap();
        }
        delta_el.style().set_property("display", "block").unwrap();
    }

    /// Show delta indicators comparing current results to loaded report.
    fn show_deltas(&self) {
        let Some(ref report) = self.compare_report else {
            return;
        };
        for (i, delta_el) in self.bench_delta_texts.iter().enumerate() {
            let cur_text = self.bench_result_texts[i]
                .text_content()
                .unwrap_or_default();
            let cur_ms = cur_text
                .split(" ms/f")
                .next()
                .and_then(|s| s.trim().parse::<f64>().ok());
            let Some(cur) = cur_ms else {
                delta_el.style().set_property("display", "none").unwrap();
                continue;
            };

            let name = self.bench_names[i];
            let baseline = report
                .results
                .iter()
                .find(|r| r.name == name)
                .map(|r| r.ms_per_frame);

            let Some(base) = baseline else {
                delta_el.style().set_property("display", "none").unwrap();
                continue;
            };

            let pct = ((cur - base) / base) * 100.0;
            let abs_pct = pct.abs();

            if abs_pct < 5.0 {
                // Within noise — grey.
                delta_el.set_text_content(Some(&format!("{pct:+.1}%")));
                delta_el.style().set_property("color", "#6c7086").unwrap();
            } else if pct < 0.0 {
                // Faster — green.
                delta_el.set_text_content(Some(&format!("{pct:+.1}%")));
                delta_el.style().set_property("color", "#a6e3a1").unwrap();
            } else {
                // Slower — red.
                delta_el.set_text_content(Some(&format!("+{pct:.1}%")));
                delta_el.style().set_property("color", "#f38ba8").unwrap();
            }
            delta_el.style().set_property("display", "block").unwrap();
        }
    }

    /// Hide all delta indicators.
    pub fn hide_deltas(&self) {
        for el in &self.bench_delta_texts {
            el.style().set_property("display", "none").unwrap();
        }
    }

    /// Read configured viewport width.
    pub fn configured_viewport(&self) -> (u32, u32) {
        let w: u32 = self.vp_width_input.value().parse().unwrap_or(0);
        let h: u32 = self.vp_height_input.value().parse().unwrap_or(0);
        (w, h)
    }

    /// Save button ref.
    pub fn save_btn(&self) -> &HtmlElement {
        &self.save_btn
    }

    /// Compare select ref.
    pub fn compare_select(&self) -> &HtmlSelectElement {
        &self.compare_select
    }

    /// Delete the currently selected comparison report.
    pub fn delete_selected_report(&mut self) {
        let val = self.compare_select.value();
        if val.is_empty() {
            return;
        }
        let idx: usize = match val.parse() {
            Ok(i) => i,
            Err(_) => return,
        };
        crate::storage::delete_report(idx);
        self.compare_report = None;
        self.hide_deltas();
        self.refresh_compare_dropdown();
    }
}

// ── Tab styling ──────────────────────────────────────────────────────────────

fn style_tab(el: &HtmlElement, active: bool) {
    set(
        el,
        &[
            ("padding", "8px 16px"),
            ("cursor", "pointer"),
            ("user-select", "none"),
            ("font-size", "13px"),
            ("border-radius", "6px 6px 0 0"),
            ("transition", "color 0.15s, border-color 0.15s"),
            ("border-bottom", "2px solid"),
        ],
    );
    if active {
        el.style().set_property("color", "#89b4fa").unwrap();
        el.style().set_property("border-color", "#89b4fa").unwrap();
    } else {
        el.style().set_property("color", "#6c7086").unwrap();
        el.style()
            .set_property("border-color", "transparent")
            .unwrap();
    }
}

// ── Number input helper ──────────────────────────────────────────────────────

fn num_input(document: &Document, label: &str, default: &str) -> (HtmlElement, HtmlInputElement) {
    let wrapper = div(document);
    set(
        &wrapper,
        &[
            ("display", "flex"),
            ("align-items", "center"),
            ("gap", "6px"),
        ],
    );

    let lbl = div(document);
    lbl.set_text_content(Some(label));
    set(&lbl, &[("color", "#9399b2"), ("font-size", "12px")]);
    wrapper.append_child(&lbl).unwrap();

    let input: HtmlInputElement = document
        .create_element("input")
        .unwrap()
        .dyn_into()
        .unwrap();
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

fn sized_num_input(document: &Document, default: &str, width: &str) -> HtmlInputElement {
    let input: HtmlInputElement = document
        .create_element("input")
        .unwrap()
        .dyn_into()
        .unwrap();
    input.set_type("number");
    input.set_value(default);
    set_prop(&input, "width", width);
    set_prop(&input, "background", "#1e1e2e");
    set_prop(&input, "color", "#cdd6f4");
    set_prop(&input, "border", "1px solid #45475a");
    set_prop(&input, "border-radius", "6px");
    set_prop(&input, "padding", "4px 8px");
    set_prop(&input, "font-family", "inherit");
    set_prop(&input, "font-size", "12px");
    input
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
        set(
            &label,
            &[
                ("color", "#9399b2"),
                ("margin-bottom", "4px"),
                ("font-size", "11px"),
                ("text-transform", "uppercase"),
                ("letter-spacing", "1px"),
            ],
        );
        row.append_child(&label).unwrap();

        let val_span = div(document);
        set(
            &val_span,
            &[
                ("display", "inline"),
                ("margin-left", "8px"),
                ("color", "#cdd6f4"),
            ],
        );

        let ctrl = match &p.kind {
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
                    input
                        .add_event_listener_with_callback("input", cb.as_ref().unchecked_ref())
                        .unwrap();
                    cb.forget();
                }

                ParamCtrl::Slider(input)
            }
            ParamKind::Select(options) => {
                let sel: HtmlSelectElement = document
                    .create_element("select")
                    .unwrap()
                    .dyn_into()
                    .unwrap();
                select_style(&sel);
                sel.set_disabled(read_only);
                for &(text, val) in options {
                    let opt = document.create_element("option").unwrap();
                    opt.set_text_content(Some(text));
                    opt.set_attribute("value", &val.to_string()).unwrap();
                    sel.append_child(&opt).unwrap();
                }
                let idx = options
                    .iter()
                    .position(|&(_, v)| (v - p.value).abs() < f64::EPSILON)
                    .unwrap_or(0);
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
