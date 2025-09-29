// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG rendering example scene.

use std::fmt;
use vello_common::kurbo::{Affine, Stroke};
use vello_common::pico_svg::{Item, PicoSvg};
use vello_common::recording::{Recorder, Recording};

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use crate::{ExampleScene, RenderingContext};

/// SVG scene that renders an SVG file
pub struct SvgScene {
    transform: Affine,
    svg: PicoSvg,
    /// Whether recording functionality is enabled
    recording_enabled: bool,
    /// The recording to reuse
    recording: CachedRecording,
}

impl fmt::Debug for SvgScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SvgScene")
    }
}

impl ExampleScene for SvgScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        let current_transform = root_transform * self.transform;

        if self.recording_enabled {
            // Try to reuse existing recording if possible
            let render_result = try_reuse_recording(ctx, &mut self.recording, current_transform);
            if render_result.is_reused {
                return;
            }

            // If we get here, we need to record fresh
            record_fresh(self, ctx, current_transform);
        } else {
            // Direct rendering mode (no recording/caching)
            #[cfg(not(target_arch = "wasm32"))]
            let start = std::time::Instant::now();
            render_svg(ctx, &self.svg.items, current_transform);
            #[cfg(not(target_arch = "wasm32"))]
            {
                let elapsed = start.elapsed();
                println!(
                    "Direct    : {:.3}ms | No caching",
                    elapsed.as_secs_f64() * 1000.0
                );
            }
        }
    }

    fn handle_key(&mut self, key: &str) -> bool {
        match key {
            "r" | "R" => {
                self.toggle_recording();
                true
            }
            _ => false,
        }
    }
}

struct RenderResult {
    is_reused: bool,
}

struct CachedRecording {
    // The transform the recording was taken at. Informs if recording can be re-used or if it needs
    // to be re-recorded.
    pub(crate) transform_key: Option<Affine>,
    // The recording absolutely positioned from the `transform_key`.
    recording: Recording,
}

impl CachedRecording {
    fn new() -> Self {
        Self {
            transform_key: None,
            recording: Recording::new(),
        }
    }
}

/// Try to reuse an existing recording, either directly (TODO: or with translation)
fn try_reuse_recording(
    ctx: &mut impl RenderingContext,
    recording: &mut CachedRecording,
    current_transform: Affine,
) -> RenderResult {
    // There is no `transform_key` meaning there is no valid recording to execute.
    let Some(recording_transform) = recording.transform_key else {
        return RenderResult { is_reused: false };
    };
    #[cfg(not(target_arch = "wasm32"))]
    let start = std::time::Instant::now();
    // Case 1: Identical transforms - can reuse directly
    if transforms_are_identical(recording_transform, current_transform) {
        ctx.execute_recording(&recording.recording);
        #[cfg(not(target_arch = "wasm32"))]
        print_render_stats("Identical ", start.elapsed(), &recording.recording);
        return RenderResult { is_reused: true };
    }

    // TODO: Implement "Case 2: Check if we can use with translation"

    // Case 3: Can't optimize, need to record fresh
    RenderResult { is_reused: false }
}

/// Record a fresh scene from scratch
fn record_fresh(
    scene_obj: &mut SvgScene,
    ctx: &mut impl RenderingContext,
    current_transform: Affine,
) {
    #[cfg(not(target_arch = "wasm32"))]
    let start = std::time::Instant::now();
    scene_obj.recording.transform_key = Some(current_transform);
    let new_recording = &mut scene_obj.recording.recording;
    new_recording.clear();
    ctx.record(new_recording, |recorder| {
        render_svg_record(recorder, &scene_obj.svg.items, current_transform);
    });
    ctx.prepare_recording(new_recording);
    ctx.execute_recording(new_recording);
    #[cfg(not(target_arch = "wasm32"))]
    print_render_stats("Fresh     ", start.elapsed(), new_recording);
}

/// Print timing and statistics for a render operation
#[cfg(not(target_arch = "wasm32"))]
fn print_render_stats(render_type: &str, elapsed: std::time::Duration, recording: &Recording) {
    println!(
        "SVG  | {}: {:.3}ms | Strips: {} | Alphas: {}",
        render_type,
        elapsed.as_secs_f64() * 1000.0,
        recording.strip_count(),
        recording.alpha_count()
    );
}

/// Check if two transforms are identical (within tolerance)
fn transforms_are_identical(a: Affine, b: Affine) -> bool {
    let a_coeffs = a.as_coeffs();
    let b_coeffs = b.as_coeffs();
    let tolerance = 1e-6;

    for i in 0..6 {
        if (a_coeffs[i] - b_coeffs[i]).abs() > tolerance {
            return false;
        }
    }
    true
}

impl SvgScene {
    /// Create a new `SvgScene` with the Ghost Tiger SVG
    pub fn tiger() -> Self {
        // Load the ghost tiger SVG by default
        #[cfg(target_arch = "wasm32")]
        let svg_content = include_str!("../../../examples/assets/Ghostscript_Tiger.svg");
        #[cfg(not(target_arch = "wasm32"))]
        let svg_content = {
            let cargo_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .canonicalize()
                .unwrap();
            &std::fs::read_to_string(cargo_dir.join("../../examples/assets/Ghostscript_Tiger.svg"))
                .unwrap()
        };

        let svg = PicoSvg::load(svg_content, 1.0).expect("Failed to parse Ghost Tiger SVG");

        Self {
            transform: Affine::scale(3.0),
            svg,
            recording_enabled: true,
            recording: CachedRecording::new(),
        }
    }

    /// Create a new `SvgScene` with the content from a given file
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_svg_file(path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let svg_content = std::fs::read_to_string(path)?;
        let svg = PicoSvg::load(&svg_content, 1.0)?;

        Ok(Self {
            transform: Affine::scale(3.0),
            svg,
            recording: CachedRecording::new(),
            recording_enabled: true,
        })
    }

    /// Toggle recording functionality on/off
    /// Returns the new state (true = enabled, false = disabled)
    pub fn toggle_recording(&mut self) -> bool {
        self.recording_enabled = !self.recording_enabled;
        self.recording_enabled
    }
}

/// Render SVG to recording
fn render_svg_record(ctx: &mut Recorder<'_>, items: &[Item], transform: Affine) {
    ctx.set_transform(transform);
    for item in items {
        match item {
            Item::Fill(fill_item) => {
                ctx.set_paint(fill_item.color);
                ctx.fill_path(&fill_item.path);
            }
            Item::Stroke(stroke_item) => {
                let style = Stroke::new(stroke_item.width);
                ctx.set_stroke(style);
                ctx.set_paint(stroke_item.color);
                ctx.stroke_path(&stroke_item.path);
            }
            Item::Group(group_item) => {
                render_svg_record(ctx, &group_item.children, transform * group_item.affine);
                ctx.set_transform(transform);
            }
        }
    }
}

/// Render SVG directly to scene without recording
fn render_svg(ctx: &mut impl RenderingContext, items: &[Item], transform: Affine) {
    ctx.set_transform(transform);
    for item in items {
        match item {
            Item::Fill(fill_item) => {
                ctx.set_paint(fill_item.color);
                ctx.fill_path(&fill_item.path);
            }
            Item::Stroke(stroke_item) => {
                let style = Stroke::new(stroke_item.width);
                ctx.set_stroke(style);
                ctx.set_paint(stroke_item.color);
                ctx.stroke_path(&stroke_item.path);
            }
            Item::Group(group_item) => {
                render_svg(ctx, &group_item.children, transform * group_item.affine);
                ctx.set_transform(transform);
            }
        }
    }
}
