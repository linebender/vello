// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark harness with warmup calibration and vsync-independent timing.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this benchmark"
)]

use crate::scenes::BenchScene;
use vello_common::kurbo::Affine;
use vello_hybrid::Scene;

/// A predefined benchmark with fixed parameters.
#[derive(Debug, Clone)]
pub(crate) struct BenchDef {
    /// Display name.
    pub(crate) name: &'static str,
    /// Short description of what this benchmark tests.
    pub(crate) description: &'static str,
    /// Which scene index to use.
    pub(crate) scene_idx: usize,
    /// Parameter overrides (speed is always forced to 0 on top of these).
    pub(crate) params: &'static [(&'static str, f64)],
}

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
pub(crate) struct BenchResult {
    /// Benchmark name.
    pub(crate) name: &'static str,
    /// Average time per frame in milliseconds.
    pub(crate) ms_per_frame: f64,
    /// Number of iterations in the run phase.
    pub(crate) iterations: usize,
    /// Total wall-clock time of the run phase in milliseconds.
    #[allow(dead_code, reason = "useful for detailed output")]
    pub(crate) total_ms: f64,
}

/// Events emitted by the harness after each tick.
#[derive(Debug)]
pub(crate) enum HarnessEvent {
    /// The first warmup frame was just rendered — caller should capture a screenshot.
    ScreenshotReady,
    /// A single benchmark finished.
    BenchDone(BenchResult),
    /// All benchmarks finished.
    AllDone,
}

/// Current phase.
#[derive(Debug)]
enum Phase {
    Idle,
    PendingWarmup(usize),
    PendingRun { idx: usize, target_iters: usize },
    Complete,
}

/// Orchestrates running benchmarks.
#[derive(Debug)]
pub(crate) struct BenchHarness {
    phase: Phase,
    pub(crate) warmup_ms: f64,
    pub(crate) run_ms: f64,
    pub(crate) results: Vec<BenchResult>,
    /// Ordered list of bench def indices to run.
    run_order: Vec<usize>,
    /// Current position within `run_order`.
    run_pos: usize,
}

impl BenchHarness {
    pub(crate) fn new() -> Self {
        Self {
            phase: Phase::Idle,
            warmup_ms: 250.0,
            run_ms: 1000.0,
            results: Vec::new(),
            run_order: Vec::new(),
            run_pos: 0,
        }
    }

    /// Start with a specific set of def indices to run (in order).
    pub(crate) fn start(&mut self, selected: Vec<usize>) {
        self.results.clear();
        self.run_order = selected;
        self.run_pos = 0;
        if self.run_order.is_empty() {
            self.phase = Phase::Complete;
        } else {
            self.phase = Phase::PendingWarmup(self.run_order[0]);
        }
    }

    pub(crate) fn is_running(&self) -> bool {
        !matches!(self.phase, Phase::Idle | Phase::Complete)
    }

    pub(crate) fn current_bench_idx(&self) -> Option<usize> {
        match &self.phase {
            Phase::PendingWarmup(i) | Phase::PendingRun { idx: i, .. } => Some(*i),
            _ => None,
        }
    }

    /// Drive one step. Returns events for the caller to act on.
    pub(crate) fn tick(
        &mut self,
        defs: &[BenchDef],
        scenes: &mut [Box<dyn BenchScene>],
        vello_scene: &mut Scene,
        renderer: &mut vello_hybrid::WebGlRenderer,
        width: u32,
        height: u32,
    ) -> Vec<HarnessEvent> {
        let mut events = Vec::new();

        match self.phase {
            Phase::Idle | Phase::Complete => {}
            Phase::PendingWarmup(idx) => {
                let def = &defs[idx];
                let scene = &mut *scenes[def.scene_idx];
                apply_params(scene, def.params);

                let perf = web_sys::window().unwrap().performance().unwrap();

                // First frame: generate geometry + capture screenshot
                let now = perf.now();
                render_one(scene, vello_scene, renderer, width, height, now);
                gpu_sync(renderer);
                events.push(HarnessEvent::ScreenshotReady);

                // Warmup loop
                let start = perf.now();
                let mut count = 0_usize;
                loop {
                    let t = perf.now();
                    render_one(scene, vello_scene, renderer, width, height, t);
                    gpu_sync(renderer);
                    count += 1;
                    if perf.now() - start >= self.warmup_ms {
                        break;
                    }
                }
                let elapsed = perf.now() - start;
                let rate = count as f64 / elapsed;
                let target = (rate * self.run_ms).max(1.0) as usize;

                self.phase = Phase::PendingRun {
                    idx,
                    target_iters: target,
                };
            }
            Phase::PendingRun { idx, target_iters } => {
                let def = &defs[idx];
                let scene = &mut *scenes[def.scene_idx];

                let perf = web_sys::window().unwrap().performance().unwrap();
                let start = perf.now();
                for _ in 0..target_iters {
                    let t = perf.now();
                    render_one(scene, vello_scene, renderer, width, height, t);
                    gpu_sync(renderer);
                }
                let total_ms = perf.now() - start;

                let result = BenchResult {
                    name: def.name,
                    ms_per_frame: total_ms / target_iters as f64,
                    iterations: target_iters,
                    total_ms,
                };
                self.results.push(result.clone());
                events.push(HarnessEvent::BenchDone(result));

                self.run_pos += 1;
                if self.run_pos < self.run_order.len() {
                    self.phase = Phase::PendingWarmup(self.run_order[self.run_pos]);
                } else {
                    self.phase = Phase::Complete;
                    events.push(HarnessEvent::AllDone);
                }
            }
        }

        events
    }
}

fn apply_params(scene: &mut dyn BenchScene, params: &[(&str, f64)]) {
    for &(name, value) in params {
        scene.set_param(name, value);
    }
    // Always force speed=0 for deterministic benchmarks.
    scene.set_param("speed", 0.0);
}

fn render_one(
    bench_scene: &mut dyn BenchScene,
    vello_scene: &mut Scene,
    renderer: &mut vello_hybrid::WebGlRenderer,
    width: u32,
    height: u32,
    time: f64,
) {
    vello_scene.reset();
    bench_scene.render(vello_scene, renderer, width, height, time, Affine::IDENTITY);
    let render_size = vello_hybrid::RenderSize { width, height };
    renderer.render(vello_scene, &render_size).unwrap();
}

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

/// All predefined benchmarks.
pub(crate) fn bench_defs() -> Vec<BenchDef> {
    vec![
        BenchDef {
            name: "200k Rect · 5×5 · Solid",
            description: "Checks how fast we can draw small rectangles",
            scene_idx: 0,
            params: &[
                ("num_rects", 200_000.0),
                ("rect_size", 5.0),
                ("paint_mode", 0.0),
                ("rotated", 0.0),
            ],
        },
        BenchDef {
            name: "50k Rect · 50×50 · Solid",
            description: "Checks how fast we can draw medium-sized rectangles",
            scene_idx: 0,
            params: &[
                ("num_rects", 50_000.0),
                ("rect_size", 50.0),
                ("paint_mode", 0.0),
                ("rotated", 0.0),
            ],
        },
        BenchDef {
            name: "10k Rect · 200×200 · Solid",
            description: "Checks how fast we can draw large rectangles",
            scene_idx: 0,
            params: &[
                ("num_rects", 10_000.0),
                ("rect_size", 200.0),
                ("paint_mode", 0.0),
                ("rotated", 0.0),
            ],
        },
        BenchDef {
            name: "10k Rect · 200×200 · Image · Nearest",
            description: "Checks how fast we can draw transparent image rectangles with nearest-neighbor filtering",
            scene_idx: 0,
            params: &[
                ("num_rects", 10_000.0),
                ("rect_size", 200.0),
                ("paint_mode", 2.0),
                ("rotated", 0.0),
                ("image_filter", 0.0),
                ("image_opaque", 0.0),
            ],
        },
        BenchDef {
            name: "10k Rect · 200×200 · Image · Bilinear",
            description: "Checks how fast we can draw transparent image rectangles with bilinear filtering",
            scene_idx: 0,
            params: &[
                ("num_rects", 10_000.0),
                ("rect_size", 200.0),
                ("paint_mode", 2.0),
                ("rotated", 0.0),
                ("image_filter", 1.0),
                ("image_opaque", 0.0),
            ],
        },
        BenchDef {
            name: "10k Rect · 200×200 · Opaque Image · Nearest",
            description: "Checks how fast we can draw opaque image rectangles with nearest-neighbor filtering",
            scene_idx: 0,
            params: &[
                ("num_rects", 10_000.0),
                ("rect_size", 200.0),
                ("paint_mode", 2.0),
                ("rotated", 0.0),
                ("image_filter", 0.0),
                ("image_opaque", 1.0),
            ],
        },
        BenchDef {
            name: "10k Rect · 200×200 · Opaque Image · Bilinear",
            description: "Checks how fast we can draw opaque image rectangles with bilinear filtering",
            scene_idx: 0,
            params: &[
                ("num_rects", 10_000.0),
                ("rect_size", 200.0),
                ("paint_mode", 2.0),
                ("rotated", 0.0),
                ("image_filter", 1.0),
                ("image_opaque", 1.0),
            ],
        },
        BenchDef {
            name: "Tiger SVG · 1×",
            description: "Ghostscript Tiger SVG fit to viewport",
            scene_idx: 1,
            params: &[],
        },
    ]
}
