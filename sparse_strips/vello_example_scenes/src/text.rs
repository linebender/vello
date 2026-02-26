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
use parley_draw::Glyph;
use vello_common::kurbo::{Affine, Rect, Vec2};
use vello_common::peniko::Color;

use crate::{ExampleScene, RenderingContext};

const PADDING: u32 = 20;
const MAX_ADVANCE: f32 = 200.0;
const FONT_SIZE: f32 = 16.0;

const SIMPLE_TEXT: &str = "Some text here. Let's make it a bit longer so that \
    line wrapping kicks in easily. This demonstrates basic glyph caching with \
    plain Latin text and common punctuation???";

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
    /// Type-erased glyph caches, lazily initialized per backend.
    glyph_caches: Option<Box<dyn Any>>,
}

impl fmt::Debug for TextScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TextScene")
    }
}

impl ExampleScene for TextScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        if self.glyph_caches.is_none() {
            self.glyph_caches = Some(ctx.create_glyph_caches());
        }

        let content_transform =
            root_transform * Affine::translate(Vec2::new(PADDING as f64, PADDING as f64));
        ctx.set_transform(content_transform);

        let glyph_caches = self
            .glyph_caches
            .as_mut()
            .expect("glyph caches not initialized");

        for line in self.layout.lines() {
            for item in line.items() {
                if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                    render_glyph_run(ctx, &glyph_run, glyph_caches.as_mut());
                }
            }
        }
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
            glyph_caches: None,
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
            glyph_caches: None,
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
) {
    let style = glyph_run.style();
    ctx.set_paint(style.brush.color);

    let run = glyph_run.run();
    let font = run.font();
    let font_size = run.font_size();
    let normalized_coords = run.normalized_coords();

    let glyphs: Vec<Glyph> = glyph_run
        .positioned_glyphs()
        .map(|g| Glyph {
            id: u32::from(g.id),
            x: g.x,
            y: g.y,
        })
        .collect();

    ctx.fill_glyphs(
        font,
        font_size,
        true,
        normalized_coords,
        glyphs.into_iter(),
        glyph_caches,
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
