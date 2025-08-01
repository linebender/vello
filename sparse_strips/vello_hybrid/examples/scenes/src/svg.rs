// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG rendering example scene.

use std::fmt;
use vello_common::kurbo::{Affine, Stroke};
use vello_common::pico_svg::{Item, PicoSvg};
use vello_hybrid::{Recorder, Recording, Scene};

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use crate::ExampleScene;

/// SVG scene that renders an SVG file
pub struct SvgScene {
    transform: Affine,
    svg: PicoSvg,
    recording: Option<Recording>,
}

impl fmt::Debug for SvgScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SvgScene")
    }
}

impl ExampleScene for SvgScene {
    fn render(&mut self, scene: &mut Scene, root_transform: Affine) {
        println!("=========== Render ===========");
        let current_transform = root_transform * self.transform;

        // Try to reuse existing recording if possible
        if let Some(recording) = &mut self.recording {
            let render_result = try_reuse_recording(scene, recording, current_transform);
            if render_result.is_reused {
                return;
            }
        }

        // If we get here, we need to record fresh
        record_fresh(self, scene, current_transform);
    }
}

struct RenderResult {
    is_reused: bool,
}

/// Try to reuse an existing recording, either directly or with translation
fn try_reuse_recording(
    scene: &mut Scene,
    recording: &mut Recording,
    current_transform: Affine,
) -> RenderResult {
    let start = std::time::Instant::now();

    // Case 1: Identical transforms - can reuse directly
    if transforms_are_identical(recording.transform, current_transform) {
        scene.render_recording(recording);
        print_render_stats("Identical ", start.elapsed(), 0, 0, recording);
        return RenderResult { is_reused: true };
    }

    // Case 2: Check if we can use with translation
    let relative = current_transform * recording.transform.inverse();

    if is_pure_translation(relative) {
        let (dx, dy) = extract_pixel_translation(relative);
        recording.translate(dx, dy);
        recording.set_transform(current_transform);
        scene.render_recording(recording);
        print_render_stats("Translated", start.elapsed(), dx, dy, recording);
        return RenderResult { is_reused: true };
    }

    // Case 3: Can't optimize, need to record fresh
    RenderResult { is_reused: false }
}

/// Record a fresh scene from scratch
fn record_fresh(scene_obj: &mut SvgScene, scene: &mut Scene, current_transform: Affine) {
    let start = std::time::Instant::now();

    let mut new_recording = scene.record_and_render(|recorder| {
        render_svg(recorder, &scene_obj.svg.items, current_transform);
    });
    new_recording.set_transform(current_transform);

    let (dx, dy) = extract_pixel_translation(current_transform);
    print_render_stats("Fresh     ", start.elapsed(), dx, dy, &new_recording);

    scene_obj.recording = Some(new_recording);
}

/// Print timing and statistics for a render operation
fn print_render_stats(
    render_type: &str,
    elapsed: std::time::Duration,
    dx: i32,
    dy: i32,
    recording: &Recording,
) {
    println!(
        "{}: {:.2}ms | ({}, {}) | Strips: {} | Alphas: {}",
        render_type,
        elapsed.as_secs_f64() * 1000.0,
        dx,
        dy,
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

/// Check if a transform is a pure translation (no rotation, scaling, or shearing)
fn is_pure_translation(transform: Affine) -> bool {
    let coeffs = transform.as_coeffs();

    // Check if linear part is identity matrix
    // Coefficients: [xx, yx, xy, yy, x0, y0]
    // For pure translation: [1, 0, 0, 1, dx, dy]
    // Use relaxed tolerances for floating-point precision
    let tolerance = 1e-6;
    (coeffs[0] - 1.0).abs() <= tolerance
        && coeffs[1].abs() <= tolerance
        && coeffs[2].abs() <= tolerance
        && (coeffs[3] - 1.0).abs() <= tolerance
}

/// Extract translation components from a transform and round to i32 pixel coordinates
/// Returns (dx, dy) pixel translation values
fn extract_pixel_translation(transform: Affine) -> (i32, i32) {
    let coeffs = transform.as_coeffs();
    let dx = coeffs[4].round() as i32;
    let dy = coeffs[5].round() as i32;
    (dx, dy)
}

impl SvgScene {
    /// Create a new `SvgScene` with the Ghost Tiger SVG
    pub fn tiger() -> Self {
        // Load the ghost tiger SVG by default
        #[cfg(target_arch = "wasm32")]
        let svg_content = include_str!("../../../../../examples/assets/Ghostscript_Tiger.svg");
        #[cfg(not(target_arch = "wasm32"))]
        let svg_content = {
            let cargo_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                .canonicalize()
                .unwrap();
            &std::fs::read_to_string(
                cargo_dir.join("../../../../examples/assets/Ghostscript_Tiger.svg"),
            )
            .unwrap()
        };

        let svg = PicoSvg::load(svg_content, 1.0).expect("Failed to parse Ghost Tiger SVG");

        Self {
            transform: Affine::scale(3.0),
            svg,
            recording: None,
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
            recording: None,
        })
    }
}

fn render_svg(ctx: &mut Recorder, items: &[Item], transform: Affine) {
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
