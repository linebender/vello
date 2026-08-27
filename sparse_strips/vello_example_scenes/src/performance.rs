// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "Internal support for browser examples")]
#![allow(
    clippy::cast_possible_truncation,
    reason = "Sample windows are bounded to 120 entries"
)]

use std::fmt::Write as _;
use wasm_bindgen::JsCast;
use web_sys::HtmlElement;

const MAX_SAMPLES: usize = 120;
const PANEL_UPDATE_INTERVAL_MS: f64 = 250.0;
const SUSPENDED_FRAME_THRESHOLD_MS: f64 = 250.0;

#[derive(Clone, Copy, Debug)]
pub struct PerformanceStage {
    pub label: &'static str,
    pub description: &'static str,
    pub color: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameTiming<const N: usize> {
    pub stages_ms: [f64; N],
    pub total_ms: f64,
}

#[derive(Debug)]
pub struct PerformancePanel<const N: usize> {
    backend: &'static str,
    stages: [PerformanceStage; N],
    timing_note: &'static str,
    element: HtmlElement,
    last_frame_timestamp: Option<f64>,
    last_panel_update: f64,
    frame_intervals: Vec<f64>,
    stage_times: [Vec<f64>; N],
    total_times: Vec<f64>,
}

impl<const N: usize> PerformancePanel<N> {
    pub fn new(
        backend: &'static str,
        stages: [PerformanceStage; N],
        timing_note: &'static str,
    ) -> Self {
        let document = web_sys::window().unwrap().document().unwrap();
        let element: HtmlElement = document
            .create_element("div")
            .unwrap()
            .dyn_into::<HtmlElement>()
            .unwrap();
        let style = element.style();
        style.set_property("position", "fixed").unwrap();
        style.set_property("right", "10px").unwrap();
        style.set_property("bottom", "10px").unwrap();
        style.set_property("z-index", "1").unwrap();
        style
            .set_property("background", "rgba(0, 0, 0, 0.75)")
            .unwrap();
        style.set_property("color", "white").unwrap();
        style.set_property("padding", "8px 10px").unwrap();
        style.set_property("border-radius", "5px").unwrap();
        style
            .set_property(
                "font-family",
                "ui-monospace, SFMono-Regular, Menlo, monospace",
            )
            .unwrap();
        style.set_property("font-size", "12px").unwrap();
        style.set_property("line-height", "1.45").unwrap();
        style
            .set_property("width", "min(420px, calc(100vw - 20px))")
            .unwrap();
        style
            .set_property("max-width", "calc(100vw - 20px)")
            .unwrap();
        style.set_property("box-sizing", "border-box").unwrap();
        style
            .set_property("font-variant-numeric", "tabular-nums")
            .unwrap();
        style.set_property("pointer-events", "auto").unwrap();
        element.set_inner_text("Collecting performance samples…");
        document.body().unwrap().append_child(&element).unwrap();

        Self {
            backend,
            stages,
            timing_note,
            element,
            last_frame_timestamp: None,
            last_panel_update: 0.0,
            frame_intervals: Vec::with_capacity(MAX_SAMPLES),
            stage_times: std::array::from_fn(|_| Vec::with_capacity(MAX_SAMPLES)),
            total_times: Vec::with_capacity(MAX_SAMPLES),
        }
    }

    pub fn record(
        &mut self,
        timestamp: f64,
        timing: Option<FrameTiming<N>>,
        scene: usize,
        scene_count: usize,
        width: u32,
        height: u32,
    ) {
        if let Some(last_timestamp) = self.last_frame_timestamp {
            let interval = timestamp - last_timestamp;
            if interval > SUSPENDED_FRAME_THRESHOLD_MS {
                self.clear_samples();
            } else if interval > 0.0 {
                push_sample(&mut self.frame_intervals, interval);
            }
        }
        self.last_frame_timestamp = Some(timestamp);

        if let Some(timing) = timing {
            for (samples, duration) in self.stage_times.iter_mut().zip(timing.stages_ms) {
                push_sample(samples, duration);
            }
            push_sample(&mut self.total_times, timing.total_ms);
        }

        if timestamp - self.last_panel_update < PANEL_UPDATE_INTERVAL_MS
            && self.last_panel_update != 0.0
        {
            return;
        }
        if self.element.matches(":hover").unwrap_or(false) {
            return;
        }
        self.last_panel_update = timestamp;
        self.update_text(scene, scene_count, width, height);
    }

    pub fn reset(&mut self) {
        self.clear_samples();
        self.last_frame_timestamp = None;
        self.last_panel_update = 0.0;
    }

    fn clear_samples(&mut self) {
        self.frame_intervals.clear();
        for samples in &mut self.stage_times {
            samples.clear();
        }
        self.total_times.clear();
    }

    fn update_text(&self, scene: usize, scene_count: usize, width: u32, height: u32) {
        let Some(frame) = SampleStats::from_samples(&self.frame_intervals) else {
            return;
        };
        let Some(total) = SampleStats::from_samples(&self.total_times) else {
            return;
        };
        let stage_stats = std::array::from_fn(|index| {
            SampleStats::from_samples(&self.stage_times[index]).unwrap()
        });

        let html = format_panel_html(&PanelSnapshot {
            backend: self.backend,
            stages: &self.stages,
            timing_note: self.timing_note,
            scene,
            scene_count,
            width,
            height,
            frame: &frame,
            total: &total,
            stage_stats: &stage_stats,
            sample_count: self.total_times.len(),
        });
        self.element.set_inner_html(&html);
    }
}

struct PanelSnapshot<'a, const N: usize> {
    backend: &'a str,
    stages: &'a [PerformanceStage; N],
    timing_note: &'a str,
    scene: usize,
    scene_count: usize,
    width: u32,
    height: u32,
    frame: &'a SampleStats,
    total: &'a SampleStats,
    stage_stats: &'a [SampleStats; N],
    sample_count: usize,
}

fn format_panel_html<const N: usize>(panel: &PanelSnapshot<'_, N>) -> String {
    let fps = 1_000.0 / panel.frame.average;
    let timeline_duration = panel.frame.average.max(panel.total.average);
    let other_percent = (100.0 - 100.0 * panel.total.average / timeline_duration).max(0.0);
    let other_average = (panel.frame.average - panel.total.average).max(0.0);

    let mut html = String::with_capacity(4096);
    html.push_str(
        r#"<style>
.vello-perf-info { position:relative;cursor:help;opacity:.7 }
.vello-perf-tooltip { display:none;position:absolute;right:0;bottom:calc(100% + 6px);width:min(360px, calc(100vw - 40px));padding:9px 11px;background:#222;color:white;border:1px solid rgba(255,255,255,.35);border-radius:4px;box-shadow:0 4px 14px rgba(0,0,0,.45);font-size:12px;font-weight:400;line-height:1.35;text-align:left;z-index:2;pointer-events:none }
.vello-perf-info:hover .vello-perf-tooltip { display:block }
.vello-perf-tooltip-row { display:grid;grid-template-columns:105px 1fr;column-gap:10px;margin:3px 0 }
</style>"#,
    );
    write!(
        html,
        r#"<div style="font-size:14px;font-weight:700">{} <span class="vello-perf-info" aria-label="Performance metric definitions">ⓘ<span class="vello-perf-tooltip">
  <span class="vello-perf-tooltip-row"><strong>Frame</strong><span>Elapsed time between rAF callbacks.</span></span>"#,
        panel.backend,
    )
    .unwrap();
    for stage in panel.stages {
        write!(
            html,
            r#"<span class="vello-perf-tooltip-row"><strong>{}</strong><span>{}</span></span>"#,
            stage.label, stage.description,
        )
        .unwrap();
    }
    html.push_str(
        r#"<span class="vello-perf-tooltip-row"><strong>Browser/rAF</strong><span>Browser work, scheduling, and display wait.</span></span>
  <span class="vello-perf-tooltip-row"><strong>Full render</strong><span>Sum of measured stages, on the CPU.</span></span>
  <span class="vello-perf-tooltip-row"><strong>Statistics</strong><span>Rolling average, p95, and maximum.</span></span>
</span></span></div>"#,
    );
    write!(
        html,
        r#"<div style="margin-bottom:9px;opacity:.8">Scene {}/{} · {}×{}px · {fps:.1} FPS</div>
<div style="display:flex;justify-content:space-between;margin-bottom:4px">
  <span>Average rAF frame</span><strong>{:.2} ms</strong>
</div>
<div style="display:flex;height:20px;overflow:hidden;border:1px solid rgba(255,255,255,.45);border-radius:3px">"#,
        panel.scene, panel.scene_count, panel.width, panel.height, panel.frame.average,
    )
    .unwrap();
    for (stage, stats) in panel.stages.iter().zip(panel.stage_stats) {
        let percent = 100.0 * stats.average / timeline_duration;
        write!(
            html,
            r#"<div style="width:{percent:.3}%;background:{}"></div>"#,
            stage.color,
        )
        .unwrap();
    }
    write!(
        html,
        r#"<div style="width:{other_percent:.3}%;background:rgba(255,255,255,.16)"></div>
</div>
<div style="display:grid;grid-template-columns:10px 1fr 58px 58px 58px;column-gap:6px;align-items:center;margin-top:7px">
  <span></span><span style="opacity:.7">Stage (ms)</span>
  <span style="text-align:right;opacity:.7">avg</span>
  <span style="text-align:right;opacity:.7">p95</span>
  <span style="text-align:right;opacity:.7">max</span>"#,
    )
    .unwrap();
    for (stage, stats) in panel.stages.iter().zip(panel.stage_stats) {
        write!(
            html,
            r#"<span style="width:8px;height:8px;background:{};border-radius:2px"></span>
  <span>{}</span>
  <span style="text-align:right">{:.2}</span>
  <span style="text-align:right">{:.2}</span>
  <span style="text-align:right">{:.2}</span>"#,
            stage.color, stage.label, stats.average, stats.p95, stats.max,
        )
        .unwrap();
    }
    write!(
        html,
        r#"<span style="width:8px;height:8px;background:rgba(255,255,255,.35);border-radius:2px"></span>
  <span>Browser/rAF</span>
  <span style="text-align:right">{other_average:.2}</span>
  <span></span><span></span>
</div>
<div style="display:flex;justify-content:space-between;margin-top:8px;padding-top:7px;border-top:1px solid rgba(255,255,255,.25)">
  <strong>Full CPU render</strong><strong>{:.2} ms avg</strong>
</div>
<div style="margin-top:3px;opacity:.65">{} samples</div>
<div style="opacity:.65">{}</div>"#,
        panel.total.average, panel.sample_count, panel.timing_note,
    )
    .unwrap();
    html
}

pub fn now() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

fn push_sample(samples: &mut Vec<f64>, sample: f64) {
    if samples.len() == MAX_SAMPLES {
        samples.remove(0);
    }
    samples.push(sample);
}

struct SampleStats {
    average: f64,
    p95: f64,
    max: f64,
}

impl SampleStats {
    fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let p95_index = ((sorted.len() as f64 * 0.95).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1);

        Some(Self {
            average: samples.iter().sum::<f64>() / samples.len() as f64,
            p95: sorted[p95_index],
            max: *sorted.last().unwrap(),
        })
    }
}
