// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! DOM-based UI controls panel for the benchmark tool.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this benchmark"
)]

use crate::scenes::{BenchScene, Param, ParamKind};
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element, HtmlElement, HtmlInputElement, HtmlSelectElement};

/// A single parameter control — either a slider or a select.
enum ParamControl {
    Slider(HtmlInputElement),
    Select(HtmlSelectElement),
}

/// References to DOM elements used by the UI.
pub struct Ui {
    sidebar: HtmlElement,
    toggle_btn: HtmlElement,
    fps_label: HtmlElement,
    render_label: HtmlElement,
    viewport_label: HtmlElement,
    /// Scene selector dropdown.
    pub scene_select: HtmlSelectElement,
    controls: Vec<(ParamControl, HtmlElement, &'static str)>,
    collapsed: bool,
}

impl std::fmt::Debug for Ui {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ui")
            .field("control_count", &self.controls.len())
            .field("collapsed", &self.collapsed)
            .finish_non_exhaustive()
    }
}

impl Ui {
    /// Build the sidebar UI for the given scenes. Returns the UI handle.
    pub fn build(
        document: &Document,
        scenes: &[Box<dyn BenchScene>],
        current_scene: usize,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Self {
        let body = document.body().unwrap();

        let body_style = body.style();
        body_style.set_property("margin", "0").unwrap();
        body_style.set_property("padding", "0").unwrap();
        body_style.set_property("overflow", "hidden").unwrap();
        body_style
            .set_property("background-color", "#111")
            .unwrap();

        // Sidebar container — semi-transparent overlay
        let sidebar: HtmlElement = document
            .create_element("div")
            .unwrap()
            .dyn_into()
            .unwrap();
        let ss = sidebar.style();
        ss.set_property("position", "fixed").unwrap();
        ss.set_property("top", "0").unwrap();
        ss.set_property("left", "0").unwrap();
        ss.set_property("width", "280px").unwrap();
        ss.set_property("height", "100vh").unwrap();
        ss.set_property("background", "rgba(26, 26, 46, 0.85)")
            .unwrap();
        ss.set_property("backdrop-filter", "blur(4px)").unwrap();
        ss.set_property("color", "#e0e0e0").unwrap();
        ss.set_property("font-family", "monospace").unwrap();
        ss.set_property("font-size", "13px").unwrap();
        ss.set_property("padding", "16px").unwrap();
        ss.set_property("box-sizing", "border-box").unwrap();
        ss.set_property("overflow-y", "auto").unwrap();
        ss.set_property("z-index", "10").unwrap();
        ss.set_property("transition", "transform 0.2s ease").unwrap();

        // Collapse/expand toggle button
        let toggle_btn: HtmlElement = document
            .create_element("div")
            .unwrap()
            .dyn_into()
            .unwrap();
        let tb = toggle_btn.style();
        tb.set_property("position", "fixed").unwrap();
        tb.set_property("top", "8px").unwrap();
        tb.set_property("left", "284px").unwrap();
        tb.set_property("width", "28px").unwrap();
        tb.set_property("height", "28px").unwrap();
        tb.set_property("background", "rgba(26, 26, 46, 0.85)")
            .unwrap();
        tb.set_property("color", "#7ecfff").unwrap();
        tb.set_property("border-radius", "0 4px 4px 0").unwrap();
        tb.set_property("cursor", "pointer").unwrap();
        tb.set_property("z-index", "11").unwrap();
        tb.set_property("display", "flex").unwrap();
        tb.set_property("align-items", "center").unwrap();
        tb.set_property("justify-content", "center").unwrap();
        tb.set_property("font-family", "monospace").unwrap();
        tb.set_property("font-size", "16px").unwrap();
        tb.set_property("user-select", "none").unwrap();
        tb.set_property("transition", "left 0.2s ease").unwrap();
        toggle_btn.set_inner_html("&#x25C0;"); // left arrow
        body.append_child(&toggle_btn).unwrap();

        // Title
        let title = document.create_element("h3").unwrap();
        title.set_text_content(Some("vello_hybrid bench"));
        let ts = el_style(&title);
        ts.set_property("margin", "0 0 12px 0").unwrap();
        ts.set_property("color", "#7ecfff").unwrap();
        sidebar.append_child(&title).unwrap();

        // FPS display
        let fps_label = create_html_element(document);
        fps_label.set_text_content(Some("FPS: --"));
        fps_label
            .style()
            .set_property("margin-bottom", "2px")
            .unwrap();
        fps_label
            .style()
            .set_property("font-size", "16px")
            .unwrap();
        fps_label
            .style()
            .set_property("font-weight", "bold")
            .unwrap();
        sidebar.append_child(&fps_label).unwrap();

        // Render time display
        let render_label = create_html_element(document);
        render_label.set_text_content(Some("Render: --"));
        render_label
            .style()
            .set_property("margin-bottom", "4px")
            .unwrap();
        render_label
            .style()
            .set_property("color", "#aaa")
            .unwrap();
        sidebar.append_child(&render_label).unwrap();

        // Viewport size display
        let viewport_label = create_html_element(document);
        viewport_label.set_text_content(Some(&format_viewport(viewport_width, viewport_height)));
        viewport_label
            .style()
            .set_property("margin-bottom", "12px")
            .unwrap();
        viewport_label
            .style()
            .set_property("color", "#888")
            .unwrap();
        sidebar.append_child(&viewport_label).unwrap();

        // Scene selector
        let scene_label = document.create_element("div").unwrap();
        scene_label.set_text_content(Some("Scene:"));
        el_style(&scene_label)
            .set_property("margin-bottom", "4px")
            .unwrap();
        sidebar.append_child(&scene_label).unwrap();

        let scene_select: HtmlSelectElement = document
            .create_element("select")
            .unwrap()
            .dyn_into()
            .unwrap();
        apply_select_style(&scene_select);
        scene_select
            .style()
            .set_property("margin-bottom", "16px")
            .unwrap();

        for (i, s) in scenes.iter().enumerate() {
            let opt = document.create_element("option").unwrap();
            opt.set_text_content(Some(s.name()));
            opt.set_attribute("value", &i.to_string()).unwrap();
            scene_select.append_child(&opt).unwrap();
        }
        scene_select.set_selected_index(current_scene as i32);
        sidebar.append_child(&scene_select).unwrap();

        // Parameters header
        let params_header = document.create_element("div").unwrap();
        params_header.set_text_content(Some("--- Parameters ---"));
        let ph = el_style(&params_header);
        ph.set_property("margin-bottom", "12px").unwrap();
        ph.set_property("color", "#888").unwrap();
        sidebar.append_child(&params_header).unwrap();

        // Build controls for current scene
        let controls = build_controls(document, &sidebar, &scenes[current_scene].params());

        body.append_child(&sidebar).unwrap();

        Self {
            sidebar,
            toggle_btn,
            fps_label,
            render_label,
            viewport_label,
            scene_select,
            controls,
            collapsed: false,
        }
    }

    /// Toggle sidebar visibility. Called from the main loop when the toggle button is clicked.
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
            self.toggle_btn
                .style()
                .set_property("border-radius", "0 4px 4px 0")
                .unwrap();
            self.toggle_btn.set_inner_html("&#x25B6;"); // right arrow
        } else {
            self.sidebar
                .style()
                .set_property("transform", "translateX(0)")
                .unwrap();
            self.toggle_btn
                .style()
                .set_property("left", "284px")
                .unwrap();
            self.toggle_btn
                .style()
                .set_property("border-radius", "0 4px 4px 0")
                .unwrap();
            self.toggle_btn.set_inner_html("&#x25C0;"); // left arrow
        }
    }

    /// Return a reference to the toggle button element (for event binding).
    pub fn toggle_btn(&self) -> &HtmlElement {
        &self.toggle_btn
    }

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
            .set_text_content(Some(&format_viewport(width, height)));
    }

    /// Read the current value of each control and return `(name, value)` pairs.
    pub fn read_params(&self) -> Vec<(&'static str, f64)> {
        self.controls
            .iter()
            .map(|(control, _label, name)| {
                let v: f64 = match control {
                    ParamControl::Slider(input) => input.value().parse().unwrap_or(0.0),
                    ParamControl::Select(select) => select.value().parse().unwrap_or(0.0),
                };
                (*name, v)
            })
            .collect()
    }

    /// Rebuild the parameter controls for a new scene.
    pub fn rebuild_params(&mut self, document: &Document, params: &[Param]) {
        // Remove old control rows
        for (control, _label, _) in self.controls.drain(..) {
            let el: &Element = match &control {
                ParamControl::Slider(input) => input,
                ParamControl::Select(select) => select,
            };
            if let Some(row) = el.parent_element() {
                row.remove();
            }
        }

        self.controls = build_controls(document, &self.sidebar, params);
    }

    /// Read the selected scene index.
    pub fn selected_scene(&self) -> usize {
        self.scene_select.selected_index() as usize
    }
}

fn create_html_element(document: &Document) -> HtmlElement {
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

fn format_viewport(w: u32, h: u32) -> String {
    format!("Viewport: {w} x {h}")
}

fn build_controls(
    document: &Document,
    sidebar: &Element,
    params: &[Param],
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

        let val_span = create_html_element(document);
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
                input.style().set_property("width", "160px").unwrap();
                input
                    .style()
                    .set_property("vertical-align", "middle")
                    .unwrap();
                row.append_child(&input).unwrap();

                val_span.set_text_content(Some(&format_value(p.value, *step)));
                row.append_child(&val_span).unwrap();

                // Update value label on input change
                {
                    let val_span_clone = val_span.clone();
                    let input_clone = input.clone();
                    let step = *step;
                    let closure = Closure::wrap(Box::new(move || {
                        let v: f64 = input_clone.value().parse().unwrap_or(0.0);
                        val_span_clone.set_text_content(Some(&format_value(v, step)));
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

                for &(text, val) in options {
                    let opt = document.create_element("option").unwrap();
                    opt.set_text_content(Some(text));
                    opt.set_attribute("value", &val.to_string()).unwrap();
                    select.append_child(&opt).unwrap();
                }

                // Set selected to match current value
                let current_idx = options
                    .iter()
                    .position(|&(_, v)| (v - p.value).abs() < f64::EPSILON)
                    .unwrap_or(0);
                select.set_selected_index(current_idx as i32);

                row.append_child(&select).unwrap();

                ParamControl::Select(select)
            }
        };

        sidebar.append_child(&row).unwrap();
        controls.push((control, val_span, p.name));
    }

    controls
}

fn format_value(value: f64, step: f64) -> String {
    if step >= 1.0 {
        format!("{}", value as i64)
    } else {
        format!("{value:.1}")
    }
}
