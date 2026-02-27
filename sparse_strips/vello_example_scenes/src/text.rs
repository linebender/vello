// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Text rendering example scene.
//!
//! Closely follows the parley example at `parley/examples/common/src/lib.rs`
//! and `parley/examples/vello_hybrid_render/src/main.rs`.

use core::any::Any;
use core::fmt;

use parley::{
    Alignment, AlignmentOptions, FontContext, FontFamily, FontWeight, GenericFamily, GlyphRun,
    Layout, LayoutContext, LineHeight, PositionedLayoutItem, StyleProperty,
};
use parley_draw::{Glyph, ImageCache};
use vello_common::kurbo::{Affine, Rect, Vec2};
use vello_common::peniko::Color;

use crate::{ExampleScene, RenderingContext, TextConfig};

const PADDING: u32 = 100;
const MAX_ADVANCE: f32 = 1780.0;
const FONT_SIZE: f32 = 32.0;

const SIMPLE_TEXT: &str = "Some text here. Let's make it a bit longer so that \
    line wrapping kicks in easily. This demonstrates basic glyph caching with \
    plain Latin text and common punctuation??? The quick brown fox jumps over \
    the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick \
    daft zebras jump! The five boxing wizards jump quickly. Sphinx of black \
    quartz, judge my vow. Two driven jocks help fax my big quiz. The jay, pig, \
    fox, zebra and my wolves quack! Crazy Frederick bought many very exquisite \
    opal jewels. We promptly judged antique ivory buckles for the next prize. \
    A mad boxer shot a quick, gloved jab to the jaw of his dizzy opponent. \
    Jived fox nymph grabs quick waltz. Glib jocks quiz nymph to vex dwarf. \
    How quickly daft jumping zebras vex! Jackdaws love my big sphinx of quartz. \
    The quick brown fox jumps over the lazy dog again and again and again. \
    Amazingly few discotheques provide jukeboxes. My girl wove six dozen plaid \
    jackets before she quit. Six big devils from Japan quickly forgot how to \
    waltz. Big July earthquakes confound zany experimental vow. Foxy parsons \
    quiz and cajole the lovably dim wiki-Loss. Have a pick: twenty-six letters, \
    no more, no less. Each sentence is a pangram, using every letter at least \
    once. This block of text is designed to stress test glyph caching by \
    exercising the full Latin alphabet repeatedly across many lines of wrapped \
    text at a small font size, ensuring the atlas must handle hundreds of glyph \
    instances with varying subpixel positions. Some text here. Let's make it a bit longer so that \
    line wrapping kicks in easily. This demonstrates basic glyph caching with \
    plain Latin text and common punctuation??? The quick brown fox jumps over \
    the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick \
    daft zebras jump! The five boxing wizards jump quickly. Sphinx of black \
    quartz, judge my vow. Two driven jocks help fax my big quiz. The jay, pig, \
    fox, zebra and my wolves quack! Crazy Frederick bought many very exquisite \
    opal jewels. We promptly judged antique ivory buckles for the next prize. \
    A mad boxer shot a quick, gloved jab to the jaw of his dizzy opponent. \
    Jived fox nymph grabs quick waltz. Glib jocks quiz nymph to vex dwarf. \
    How quickly daft jumping zebras vex! Jackdaws love my big sphinx of quartz. \
    The quick brown fox jumps over the lazy dog again and again and again. \
    Amazingly few discotheques provide jukeboxes. My girl wove six dozen plaid \
    jackets before she quit. Six big devils from Japan quickly forgot how to \
    waltz. Big July earthquakes confound zany experimental vow.";

// const SIMPLE_TEXT: &str = "Some text here. Let's make it a bit longer so that \
//     line wrapping kicks in easily. This demonstrates basic glyph caching with \
//     plain Latin text and common punctuation???";

const RICH_TEXT: &str = "Some text here. Let's make it a bit longer so that \
    line wrapping kicks in. Bitmap emoji 😊 and COLR emoji 🎉.\n\
    And also some اللغة العربية arabic text.\n\
    This is underlining pq and strikethrough text.";

/// Minimal brush carrying only a solid color, matching the parley examples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorBrush {
    /// The solid color for this brush.
    pub color: Color,
}

impl Default for ColorBrush {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
        }
    }
}

#[cfg(target_arch = "wasm32")]
const ROBOTO_FONT: &[u8] = include_bytes!("../../../examples/assets/roboto/Roboto-Regular.ttf");

/// State for the text example.
pub struct TextScene {
    layout: Layout<ColorBrush>,
    repeat_count: u32,
}

impl fmt::Debug for TextScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TextScene")
    }
}

impl ExampleScene for TextScene {
    fn render(
        &mut self,
        ctx: &mut impl RenderingContext,
        root_transform: Affine,
        glyph_caches: &mut dyn Any,
        image_cache: &mut ImageCache,
        text_config: &TextConfig,
    ) {
        for i in 0..self.repeat_count {
            let offset = i as f64 * 10.0;
            let content_transform = root_transform
                * Affine::translate(Vec2::new(PADDING as f64 + offset, PADDING as f64 + offset));
            ctx.set_transform(content_transform);

            for line in self.layout.lines() {
                for item in line.items() {
                    if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                        render_glyph_run(ctx, &glyph_run, glyph_caches, image_cache, text_config);
                    }
                }
            }
        }
    }

    fn handle_key(&mut self, key: &str) -> bool {
        match key {
            "+" | "=" => {
                self.repeat_count += 1;
                true
            }
            "-" | "_" => {
                self.repeat_count = self.repeat_count.saturating_sub(1).max(1);
                true
            }
            _ => false,
        }
    }

    fn status(&self) -> Option<String> {
        Some(format!("repeats: {}", self.repeat_count))
    }
}

impl TextScene {
    /// Create a new `TextScene` with the given text using the simple layout.
    pub fn new(text: &str) -> Self {
        let mut font_cx = FontContext::new();
        let mut layout_cx = LayoutContext::new();

        #[cfg(target_arch = "wasm32")]
        font_cx
            .collection
            .register_fonts(ROBOTO_FONT.to_vec().into(), None);

        let layout = build_simple_layout(&mut font_cx, &mut layout_cx, text);

        Self {
            layout,
            repeat_count: 1,
        }
    }

    /// Create a new `TextScene` with the rich layout (emoji, underline, etc.).
    pub fn rich() -> Self {
        let mut font_cx = FontContext::new();
        let mut layout_cx = LayoutContext::new();

        #[cfg(target_arch = "wasm32")]
        font_cx
            .collection
            .register_fonts(ROBOTO_FONT.to_vec().into(), None);

        let layout = build_rich_layout(&mut font_cx, &mut layout_cx, RICH_TEXT);

        Self {
            layout,
            repeat_count: 1,
        }
    }
}

impl Default for TextScene {
    fn default() -> Self {
        Self::new(SIMPLE_TEXT)
    }
}

fn build_simple_layout(
    font_cx: &mut FontContext,
    layout_cx: &mut LayoutContext<ColorBrush>,
    text: &str,
) -> Layout<ColorBrush> {
    let mut builder = layout_cx.ranged_builder(font_cx, text, 1.0, true);

    let foreground = ColorBrush {
        color: Color::WHITE,
    };
    builder.push_default(StyleProperty::Brush(foreground));
    builder.push_default(GenericFamily::SystemUi);
    builder.push_default(LineHeight::FontSizeRelative(1.3));
    builder.push_default(StyleProperty::FontSize(FONT_SIZE));

    let bold = FontWeight::new(600.0);
    builder.push(StyleProperty::FontWeight(bold), 0..4);

    if let Some((start, matched)) = text.match_indices("here").next() {
        let purple = ColorBrush {
            color: Color::from_rgb8(200, 130, 255),
        };
        builder.push(StyleProperty::Brush(purple), start..start + matched.len());
    }

    let mut layout = builder.build(text);
    let max_advance = Some(MAX_ADVANCE);
    layout.break_all_lines(max_advance);
    layout.align(max_advance, Alignment::Start, AlignmentOptions::default());

    layout
}

fn build_rich_layout(
    font_cx: &mut FontContext,
    layout_cx: &mut LayoutContext<ColorBrush>,
    text: &str,
) -> Layout<ColorBrush> {
    let mut builder = layout_cx.ranged_builder(font_cx, text, 1.0, true);

    let foreground = ColorBrush {
        color: Color::WHITE,
    };
    builder.push_default(StyleProperty::Brush(foreground));
    builder.push_default(GenericFamily::SystemUi);
    builder.push_default(LineHeight::FontSizeRelative(1.3));
    builder.push_default(StyleProperty::FontSize(FONT_SIZE));

    let bold = FontWeight::new(600.0);
    builder.push(StyleProperty::FontWeight(bold), 0..4);

    if let Some((start, matched)) = text.match_indices("here").next() {
        let purple = ColorBrush {
            color: Color::from_rgb8(200, 130, 255),
        };
        builder.push(StyleProperty::Brush(purple), start..start + matched.len());
    }

    if let Some((start, matched)) = text.match_indices("underlining pq").next() {
        builder.push(StyleProperty::Underline(true), start..start + matched.len());
    }
    if let Some((start, matched)) = text.match_indices("strikethrough").next() {
        builder.push(
            StyleProperty::Strikethrough(true),
            start..start + matched.len(),
        );
    }
    if let Some((start, matched)) = text.match_indices("🎉").next() {
        builder.push(
            FontFamily::parse("Noto Color Emoji").unwrap(),
            start..start + matched.len(),
        );
    }

    let mut layout = builder.build(text);
    let max_advance = Some(MAX_ADVANCE);
    layout.break_all_lines(max_advance);
    layout.align(max_advance, Alignment::Start, AlignmentOptions::default());

    layout
}

fn render_glyph_run(
    ctx: &mut impl RenderingContext,
    glyph_run: &GlyphRun<'_, ColorBrush>,
    glyph_caches: &mut dyn Any,
    image_cache: &mut ImageCache,
    text_config: &TextConfig,
) {
    let style = glyph_run.style();
    ctx.set_paint(style.brush.color);

    let run = glyph_run.run();
    let font = run.font();
    let font_size = run.font_size();
    let normalized_coords = run.normalized_coords();

    ctx.fill_glyphs(
        font,
        font_size,
        normalized_coords,
        glyph_run.positioned_glyphs().map(|g| Glyph {
            id: u32::from(g.id),
            x: g.x,
            y: g.y,
        }),
        glyph_caches,
        image_cache,
        text_config,
    );

    if let Some(decoration) = &style.underline {
        let offset = decoration.offset.unwrap_or(run.metrics().underline_offset);
        let size = decoration.size.unwrap_or(run.metrics().underline_size);
        ctx.set_paint(decoration.brush.color);
        render_decoration(ctx, glyph_run, offset, size);
    }
    if let Some(decoration) = &style.strikethrough {
        let offset = decoration
            .offset
            .unwrap_or(run.metrics().strikethrough_offset);
        let size = decoration.size.unwrap_or(run.metrics().strikethrough_size);
        ctx.set_paint(decoration.brush.color);
        render_strikethrough(ctx, glyph_run, offset, size);
    }
}

fn render_decoration(
    ctx: &mut impl RenderingContext,
    glyph_run: &GlyphRun<'_, ColorBrush>,
    offset: f32,
    size: f32,
) {
    let y = glyph_run.baseline() - offset;
    let x = glyph_run.offset();
    let x1 = x + glyph_run.advance();
    let y1 = y + size;
    ctx.fill_rect(&Rect::new(x as f64, y as f64, x1 as f64, y1 as f64));
}

fn render_strikethrough(
    ctx: &mut impl RenderingContext,
    glyph_run: &GlyphRun<'_, ColorBrush>,
    offset: f32,
    size: f32,
) {
    let y = glyph_run.baseline() - offset;
    let x = glyph_run.offset();
    let x1 = x + glyph_run.advance();
    let y1 = y + size;
    ctx.fill_rect(&Rect::new(x as f64, y as f64, x1 as f64, y1 as f64));
}
