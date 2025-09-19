// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Text rendering example scene.

use core::fmt;
use parley::FontFamily;
use parley::{
    Alignment, AlignmentOptions, FontContext, GlyphRun, Layout, LayoutContext,
    PositionedLayoutItem, StyleProperty,
};
use vello_common::color::palette::css::WHITE;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::glyph::Glyph;
use vello_common::kurbo::Affine;
use vello_common::recording::{Recorder, Recording};

use crate::{ExampleScene, RenderingContext};

#[derive(Clone, Copy, Debug, PartialEq)]
struct ColorBrush {
    color: AlphaColor<Srgb>,
}

impl Default for ColorBrush {
    fn default() -> Self {
        Self { color: WHITE }
    }
}

// Wasm doesn't support system fonts, so we need to include the font data directly.
#[cfg(target_arch = "wasm32")]
const ROBOTO_FONT: &[u8] = include_bytes!("../../../examples/assets/roboto/Roboto-Regular.ttf");

/// State for the text example.
pub struct TextScene {
    layout: Layout<ColorBrush>,
    /// Whether recording functionality is enabled
    recording_enabled: bool,
    /// The recording to reuse
    recording: CachedRecording,
}

impl fmt::Debug for TextScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TextScene")
    }
}

impl ExampleScene for TextScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        if self.recording_enabled {
            // Try to reuse existing recording if possible
            let render_result = try_reuse_recording(ctx, &mut self.recording, root_transform);
            if render_result.is_reused {
                return;
            }

            // If we get here, we need to record fresh
            record_fresh(self, ctx, root_transform);
        } else {
            // Direct rendering mode (no recording/caching)
            #[cfg(not(target_arch = "wasm32"))]
            let start = std::time::Instant::now();
            ctx.set_transform(root_transform);
            render_text(self, ctx);
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

impl TextScene {
    /// Create a new `TextScene` with the given text.
    pub fn new(text: &str) -> Self {
        // Typically, you'd want to store 1 `layout_cx` and `font_cx` for the
        // duration of the program (or have an instance per thread).
        let mut layout_cx = LayoutContext::new();

        #[cfg(not(target_arch = "wasm32"))]
        let mut font_cx = FontContext::new();
        #[cfg(target_arch = "wasm32")]
        let mut font_cx = {
            let mut font_cx = FontContext::new();
            font_cx
                .collection
                .register_fonts(ROBOTO_FONT.to_vec().into(), None);
            font_cx
        };

        let mut builder = layout_cx.ranged_builder(&mut font_cx, text, 1.0, true);
        builder.push_default(FontFamily::parse("Roboto").unwrap());
        builder.push_default(StyleProperty::LineHeight(
            parley::LineHeight::FontSizeRelative(1.3),
        ));
        builder.push_default(StyleProperty::FontSize(32.0));

        let mut layout: Layout<ColorBrush> = builder.build(text);
        let max_advance = Some(400.0);
        layout.break_all_lines(max_advance);
        layout.align(max_advance, Alignment::Middle, AlignmentOptions::default());

        Self {
            layout,
            recording_enabled: true,
            recording: CachedRecording::new(),
        }
    }
}

impl Default for TextScene {
    fn default() -> Self {
        Self::new("Hello, Vello!")
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
    scene_obj: &mut TextScene,
    ctx: &mut impl RenderingContext,
    current_transform: Affine,
) {
    #[cfg(not(target_arch = "wasm32"))]
    let start = std::time::Instant::now();
    scene_obj.recording.transform_key = Some(current_transform);
    let recording = &mut scene_obj.recording.recording;
    recording.clear();
    ctx.record(recording, |recorder| {
        render_text_record(&mut scene_obj.layout, recorder, current_transform);
    });
    ctx.prepare_recording(recording);
    ctx.execute_recording(recording);
    #[cfg(not(target_arch = "wasm32"))]
    print_render_stats("Fresh     ", start.elapsed(), recording);
}

/// Print timing and statistics for a render operation
#[cfg(not(target_arch = "wasm32"))]
fn print_render_stats(render_type: &str, elapsed: std::time::Duration, recording: &Recording) {
    println!(
        "Text | {}: {:.3}ms | Strips: {} | Alphas: {}",
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

impl TextScene {
    /// Toggle recording functionality on/off
    /// Returns the new state (true = enabled, false = disabled)
    pub fn toggle_recording(&mut self) -> bool {
        self.recording_enabled = !self.recording_enabled;
        self.recording_enabled
    }
}

fn render_text(state: &mut TextScene, ctx: &mut impl RenderingContext) {
    for line in state.layout.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                render_glyph_run(ctx, &glyph_run, 30);
            }
        }
    }
}

/// Render text to recording
fn render_text_record(layout: &mut Layout<ColorBrush>, ctx: &mut Recorder<'_>, transform: Affine) {
    ctx.set_transform(transform);
    for line in layout.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                render_glyph_run_record(ctx, &glyph_run, 30);
            }
        }
    }
}

fn render_glyph_run(
    ctx: &mut impl RenderingContext,
    glyph_run: &GlyphRun<'_, ColorBrush>,
    padding: u32,
) {
    let mut run_x = glyph_run.offset();
    let run_y = glyph_run.baseline();
    let glyphs = glyph_run.glyphs().map(|glyph| {
        let glyph_x = run_x + glyph.x + padding as f32;
        let glyph_y = run_y - glyph.y + padding as f32;
        run_x += glyph.advance;

        Glyph {
            id: glyph.id as u32,
            x: glyph_x,
            y: glyph_y,
        }
    });

    let run = glyph_run.run();
    let font = run.font();
    let font_size = run.font_size();
    let normalized_coords = bytemuck::cast_slice(run.normalized_coords());

    let style = glyph_run.style();
    ctx.set_paint(style.brush.color);
    ctx.glyph_run(font)
        .font_size(font_size)
        .normalized_coords(normalized_coords)
        .hint(true)
        .fill_glyphs(glyphs);
}

fn render_glyph_run_record(
    ctx: &mut Recorder<'_>,
    glyph_run: &GlyphRun<'_, ColorBrush>,
    padding: u32,
) {
    let mut run_x = glyph_run.offset();
    let run_y = glyph_run.baseline();
    let glyphs = glyph_run.glyphs().map(|glyph| {
        let glyph_x = run_x + glyph.x + padding as f32;
        let glyph_y = run_y - glyph.y + padding as f32;
        run_x += glyph.advance;

        Glyph {
            id: glyph.id as u32,
            x: glyph_x,
            y: glyph_y,
        }
    });

    let run = glyph_run.run();
    let font = run.font();
    let font_size = run.font_size();
    let normalized_coords = bytemuck::cast_slice(run.normalized_coords());

    let style = glyph_run.style();
    ctx.set_paint(style.brush.color);
    ctx.glyph_run(font)
        .font_size(font_size)
        .normalized_coords(normalized_coords)
        .hint(true)
        .fill_glyphs(glyphs);
}
