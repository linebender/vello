// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark harness that runs predefined benchmarks with warmup calibration.
//!
//! Each benchmark goes through two phases:
//! 1. **Warmup** — render frames in a tight loop for `warmup_ms`, count how many complete.
//! 2. **Run** — use the warmup count to estimate iterations for `run_ms`, execute them,
//!    and divide total time by iteration count for a vsync-independent per-frame time.
//!
//! Multiple iterations run within a single rAF callback (tight loop with `readPixels`
//! GPU sync), so vsync does not affect the measurement.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this benchmark"
)]

use crate::scenes::BenchScene;
use vello_hybrid::Scene;

/// A predefined benchmark with fixed parameters.
#[derive(Debug, Clone)]
pub(crate) struct BenchDef {
    /// Display name.
    pub name: &'static str,
    /// Which scene index to use (into the `all_scenes()` list).
    pub scene_idx: usize,
    /// Parameter overrides to apply before running.
    pub params: &'static [(&'static str, f64)],
}

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
pub(crate) struct BenchResult {
    /// Benchmark name.
    pub name: &'static str,
    /// Average time per frame in milliseconds.
    pub time_per_frame_ms: f64,
    /// Number of iterations in the run phase.
    pub iterations: usize,
    /// Total wall-clock time of the run phase in milliseconds.
    pub total_time_ms: f64,
}

/// Current phase of the harness.
#[derive(Debug)]
enum Phase {
    /// Not running.
    Idle,
    /// About to warmup benchmark at index `bench_idx`.
    PendingWarmup(usize),
    /// About to run benchmark at index `bench_idx` with `target_iters` iterations.
    PendingRun {
        bench_idx: usize,
        target_iters: usize,
    },
    /// All benchmarks finished.
    Complete,
}

/// Orchestrates running a list of benchmarks.
#[derive(Debug)]
pub(crate) struct BenchHarness {
    phase: Phase,
    /// Duration of warmup phase in ms.
    pub warmup_ms: f64,
    /// Target duration of run phase in ms.
    pub run_ms: f64,
    /// Collected results.
    pub results: Vec<BenchResult>,
}

impl BenchHarness {
    /// Create a new harness with default timing.
    pub(crate) fn new() -> Self {
        Self {
            phase: Phase::Idle,
            warmup_ms: 250.0,
            run_ms: 1000.0,
            results: Vec::new(),
        }
    }

    /// Start (or restart) the benchmark suite from the first benchmark.
    pub(crate) fn start(&mut self) {
        self.results.clear();
        self.phase = Phase::PendingWarmup(0);
    }

    /// Whether the harness is currently running.
    pub(crate) fn is_running(&self) -> bool {
        !matches!(self.phase, Phase::Idle | Phase::Complete)
    }

    /// Whether benchmarking has finished.
    pub(crate) fn is_complete(&self) -> bool {
        matches!(self.phase, Phase::Complete)
    }

    /// The index of the benchmark currently being processed, if any.
    pub(crate) fn current_bench_idx(&self) -> Option<usize> {
        match &self.phase {
            Phase::PendingWarmup(i) | Phase::PendingRun { bench_idx: i, .. } => Some(*i),
            _ => None,
        }
    }

    /// Drive one step of the harness. Call this once per rAF.
    ///
    /// Each step does a tight loop (warmup or run) and then yields back
    /// so the browser can repaint and the UI can update.
    ///
    /// Returns `true` if the harness did work this tick.
    pub(crate) fn tick(
        &mut self,
        defs: &[BenchDef],
        scenes: &mut [Box<dyn BenchScene>],
        vello_scene: &mut Scene,
        renderer: &mut vello_hybrid::WebGlRenderer,
        width: u32,
        height: u32,
    ) -> bool {
        match self.phase {
            Phase::Idle | Phase::Complete => false,
            Phase::PendingWarmup(bench_idx) => {
                let def = &defs[bench_idx];
                let scene = &mut *scenes[def.scene_idx];
                for &(name, value) in def.params {
                    scene.set_param(name, value);
                }

                // Force rect generation by rendering one frame
                let perf = web_sys::window().unwrap().performance().unwrap();
                let now = perf.now();
                render_one(scene, vello_scene, renderer, width, height, now);
                gpu_sync(renderer);

                // Warmup: tight loop for warmup_ms
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
                let warmup_elapsed = perf.now() - start;

                // Estimate target iterations for run_ms
                let rate = count as f64 / warmup_elapsed;
                let target = (rate * self.run_ms).max(1.0) as usize;

                self.phase = Phase::PendingRun {
                    bench_idx,
                    target_iters: target,
                };
                true
            }
            Phase::PendingRun {
                bench_idx,
                target_iters,
            } => {
                let def = &defs[bench_idx];
                let scene = &mut *scenes[def.scene_idx];

                let perf = web_sys::window().unwrap().performance().unwrap();
                let start = perf.now();
                for _ in 0..target_iters {
                    let t = perf.now();
                    render_one(scene, vello_scene, renderer, width, height, t);
                    gpu_sync(renderer);
                }
                let total_ms = perf.now() - start;

                self.results.push(BenchResult {
                    name: def.name,
                    time_per_frame_ms: total_ms / target_iters as f64,
                    iterations: target_iters,
                    total_time_ms: total_ms,
                });

                // Move to next benchmark or complete
                let next = bench_idx + 1;
                if next < defs.len() {
                    self.phase = Phase::PendingWarmup(next);
                } else {
                    self.phase = Phase::Complete;
                }
                true
            }
        }
    }
}

/// Render a single frame (scene build + GPU submit).
fn render_one(
    bench_scene: &mut dyn BenchScene,
    vello_scene: &mut Scene,
    renderer: &mut vello_hybrid::WebGlRenderer,
    width: u32,
    height: u32,
    time: f64,
) {
    vello_scene.reset();
    bench_scene.render(vello_scene, width, height, time);
    let render_size = vello_hybrid::RenderSize { width, height };
    renderer.render(vello_scene, &render_size).unwrap();
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

/// The predefined benchmark definitions.
pub(crate) fn bench_defs() -> Vec<BenchDef> {
    vec![BenchDef {
        name: "Solid 5px (200k)",
        scene_idx: 0,
        params: &[
            ("num_rects", 200_000.0),
            ("rect_size", 5.0),
            ("paint_mode", 0.0),
            ("speed", 5.0),
        ],
    }]
}
