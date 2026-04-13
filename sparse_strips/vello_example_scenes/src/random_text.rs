// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Random text stress scene.

use core::fmt;
use glifo::Glyph;
use parley::FontFamily;
use parley::{
    Alignment, AlignmentOptions, FontContext, GlyphRun, Layout, LayoutContext,
    PositionedLayoutItem, StyleProperty,
};
use vello_common::color::palette::css::{AQUA, GOLD, LIME, ORANGE, WHITE};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::Affine;

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

#[cfg(target_arch = "wasm32")]
const ROBOTO_FONT: &[u8] = include_bytes!("../../../examples/assets/roboto/Roboto-Regular.ttf");

const INITIAL_SEGMENT_COUNT: usize = 200;
const BATCH_SIZE: usize = 50;
const SEGMENT_MIN_LEN: usize = 4;
const SEGMENT_MAX_LEN: usize = 12;

struct Segment {
    layout: Layout<ColorBrush>,
    x: f32,
    y: f32,
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Segment")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish_non_exhaustive()
    }
}

// TODO: Create a single pseudo-random number generator in `lib.rs` and use that for all scenes.
#[derive(Clone, Copy, Debug)]
struct Lcg {
    state: u64,
}

impl Lcg {
    const MULTIPLIER: u64 = 6364136223846793005;
    const INCREMENT: u64 = 1;

    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(Self::INCREMENT);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    fn range_usize(&mut self, start: usize, end_inclusive: usize) -> usize {
        start + (self.next_u32() as usize % (end_inclusive - start + 1))
    }

    fn range_f32(&mut self, start: f32, end: f32) -> f32 {
        start + (end - start) * self.next_f32()
    }
}

/// Stress-test scene that renders many randomly generated text segments.
pub struct RandomTextScene {
    segments: Vec<Segment>,
    rng: Lcg,
    layout_cx: LayoutContext<ColorBrush>,
    font_cx: FontContext,
    glyph_caching_enabled: bool,
    hinting_enabled: bool,
}

impl fmt::Debug for RandomTextScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RandomTextScene")
    }
}

impl Default for RandomTextScene {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomTextScene {
    /// Create a random-text scene with an initial set of generated segments.
    pub fn new() -> Self {
        let mut scene = Self {
            segments: Vec::new(),
            rng: Lcg::new(0x5eed_cafe_d00d_f00d),
            layout_cx: LayoutContext::new(),
            font_cx: new_font_context(),
            glyph_caching_enabled: false,
            hinting_enabled: false,
        };
        scene.add_segments(INITIAL_SEGMENT_COUNT);
        scene
    }

    fn add_segments(&mut self, count: usize) {
        for _ in 0..count {
            self.segments.push(build_segment(
                &mut self.rng,
                &mut self.layout_cx,
                &mut self.font_cx,
            ));
        }
    }

    fn remove_segments(&mut self, count: usize) {
        let new_len = self.segments.len().saturating_sub(count);
        self.segments.truncate(new_len);
    }
}

impl ExampleScene for RandomTextScene {
    fn render<T: RenderingContext>(
        &mut self,
        ctx: &mut T,
        resources: &mut T::Resources,
        root_transform: Affine,
    ) {
        ctx.set_transform(root_transform);
        for segment in &self.segments {
            render_segment(
                ctx,
                resources,
                &segment.layout,
                segment.x,
                segment.y,
                self.glyph_caching_enabled,
                self.hinting_enabled,
            );
        }
    }

    fn handle_key(&mut self, key: &str) -> bool {
        match key {
            "+" => {
                self.add_segments(BATCH_SIZE);
                true
            }
            "-" => {
                self.remove_segments(BATCH_SIZE);
                true
            }
            "c" | "C" => {
                self.glyph_caching_enabled = !self.glyph_caching_enabled;
                true
            }
            "h" | "H" => {
                self.hinting_enabled = !self.hinting_enabled;
                true
            }
            _ => false,
        }
    }

    fn status(&self) -> Option<String> {
        Some(format!(
            "Random text: {} segments | cache: {} | hinting: {} | +/-: {} | c/h: toggles",
            self.segments.len(),
            if self.glyph_caching_enabled {
                "on"
            } else {
                "off"
            },
            if self.hinting_enabled { "on" } else { "off" },
            BATCH_SIZE
        ))
    }
}

fn new_font_context() -> FontContext {
    #[cfg(not(target_arch = "wasm32"))]
    {
        FontContext::new()
    }

    #[cfg(target_arch = "wasm32")]
    {
        let mut font_cx = FontContext::new();
        font_cx
            .collection
            .register_fonts(ROBOTO_FONT.to_vec().into(), None);
        font_cx
    }
}

fn build_segment(
    rng: &mut Lcg,
    layout_cx: &mut LayoutContext<ColorBrush>,
    font_cx: &mut FontContext,
) -> Segment {
    let text = random_text(rng);
    let color = random_color(rng);
    let font_size = rng.range_f32(9.0, 18.0);
    let x = rng.range_f32(10.0, 1500.0);
    let y = rng.range_f32(20.0, 1000.0);

    let mut builder = layout_cx.ranged_builder(font_cx, &text, 1.0, true);
    builder.push_default(FontFamily::parse("Roboto").unwrap());
    builder.push_default(StyleProperty::FontSize(font_size));
    builder.push_default(StyleProperty::Brush(ColorBrush { color }));

    let mut layout: Layout<ColorBrush> = builder.build(&text);
    let max_advance = Some(600.0);
    layout.break_all_lines(max_advance);
    layout.align(max_advance, Alignment::Start, AlignmentOptions::default());

    Segment { layout, x, y }
}

fn random_text(rng: &mut Lcg) -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let len = rng.range_usize(SEGMENT_MIN_LEN, SEGMENT_MAX_LEN);
    let mut text = String::with_capacity(len);
    for _ in 0..len {
        let idx = rng.range_usize(0, ALPHABET.len() - 1);
        text.push(ALPHABET[idx] as char);
    }
    text
}

fn random_color(rng: &mut Lcg) -> AlphaColor<Srgb> {
    match rng.range_usize(0, 4) {
        0 => WHITE,
        1 => AQUA,
        2 => LIME,
        3 => GOLD,
        _ => ORANGE,
    }
}

fn render_segment<T: RenderingContext>(
    ctx: &mut T,
    resources: &mut T::Resources,
    layout: &Layout<ColorBrush>,
    offset_x: f32,
    offset_y: f32,
    glyph_caching_enabled: bool,
    hinting_enabled: bool,
) {
    for line in layout.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                render_glyph_run(
                    ctx,
                    resources,
                    &glyph_run,
                    offset_x,
                    offset_y,
                    glyph_caching_enabled,
                    hinting_enabled,
                );
            }
        }
    }
}

fn render_glyph_run<T: RenderingContext>(
    ctx: &mut T,
    resources: &mut T::Resources,
    glyph_run: &GlyphRun<'_, ColorBrush>,
    offset_x: f32,
    offset_y: f32,
    glyph_caching_enabled: bool,
    hinting_enabled: bool,
) {
    let mut run_x = glyph_run.offset();
    let run_y = glyph_run.baseline();
    let glyphs = glyph_run.glyphs().map(move |glyph| {
        let glyph_x = offset_x + run_x + glyph.x;
        let glyph_y = offset_y + run_y - glyph.y;
        run_x += glyph.advance;

        Glyph {
            id: glyph.id as u32,
            x: glyph_x,
            y: glyph_y,
        }
    });

    let run = glyph_run.run();
    let style = glyph_run.style();
    ctx.set_paint(style.brush.color);
    ctx.glyph_run(resources, run.font())
        .font_size(run.font_size())
        .hint(hinting_enabled)
        .atlas_cache(glyph_caching_enabled)
        .fill_glyphs(glyphs);
}
