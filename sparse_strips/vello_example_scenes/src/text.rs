// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Text rendering example scene.

use core::fmt;
use glifo::Glyph;
use parley::FontFamily;
use parley::{
    Alignment, AlignmentOptions, FontContext, GlyphRun, Layout, LayoutContext,
    PositionedLayoutItem, StyleProperty,
};
use vello_common::color::palette::css::WHITE;
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

// Wasm doesn't support system fonts, so we need to include the font data directly.
#[cfg(target_arch = "wasm32")]
const ROBOTO_FONT: &[u8] = include_bytes!("../../../examples/assets/roboto/Roboto-Regular.ttf");

/// State for the text example.
pub struct TextScene {
    layout: Layout<ColorBrush>,
}

impl fmt::Debug for TextScene {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TextScene")
    }
}

impl ExampleScene for TextScene {
    fn render<T: RenderingContext>(
        &mut self,
        ctx: &mut T,
        resources: &mut T::Resources,
        root_transform: Affine,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        let start = std::time::Instant::now();
        ctx.set_transform(root_transform);
        render_text(self, ctx, resources);
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

        Self { layout }
    }
}

impl Default for TextScene {
    fn default() -> Self {
        Self::new("Hello, Vello!")
    }
}

fn render_text<T: RenderingContext>(
    state: &mut TextScene,
    ctx: &mut T,
    resources: &mut T::Resources,
) {
    for line in state.layout.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                render_glyph_run(ctx, resources, &glyph_run, 30);
            }
        }
    }
}

fn render_glyph_run<T: RenderingContext>(
    ctx: &mut T,
    resources: &mut T::Resources,
    glyph_run: &GlyphRun<'_, ColorBrush>,
    padding: u32,
) {
    let mut run_x = glyph_run.offset();
    let run_y = glyph_run.baseline();
    let glyphs = glyph_run.glyphs().map(move |glyph| {
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
    ctx.glyph_run(resources, font)
        .font_size(font_size)
        .normalized_coords(normalized_coords)
        .hint(true)
        .fill_glyphs(glyphs);
}
