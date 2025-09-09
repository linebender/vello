// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use criterion::{Criterion, black_box};
use parley::{
    Alignment, AlignmentOptions, Font, FontContext, FontFamily, GlyphRun, Layout, LayoutContext,
    PositionedLayoutItem,
};
use vello_common::fearless_simd::Level;
use vello_common::glyph::{Glyph, GlyphCache, GlyphRunBuilder, HintCache};
use vello_common::glyph::{GlyphRenderer, GlyphType};
use vello_common::kurbo::Affine;
use vello_common::peniko::Fill;
use vello_common::strip_generator::StripGenerator;

pub fn glyph(c: &mut Criterion) {
    let mut g = c.benchmark_group("glyph");

    const WIDTH: u16 = 256;
    const HEIGHT: u16 = 256;

    let mut renderer = GlyphBenchRenderer {
        strip_generator: StripGenerator::new(
            WIDTH,
            HEIGHT,
            Level::try_detect().unwrap_or(Level::fallback()),
        ),
        hint_cache: Some(Default::default()),
        glyph_cache: Some(Default::default()),
    };

    const LATIN: &str = "The quick brown fox jumps over the lazy dog 0123456789";

    let mut layout_cx = LayoutContext::new();
    let mut font_cx = FontContext::new();
    let mut builder = layout_cx.ranged_builder(&mut font_cx, LATIN, 1.0, true);
    builder.push_default(FontFamily::parse("Roboto").unwrap());
    let mut layout: Layout<Brush> = builder.build(LATIN);
    let max_advance = Some(WIDTH as f32);
    layout.break_all_lines(max_advance);
    layout.align(max_advance, Alignment::Start, AlignmentOptions::default());

    g.bench_function("latin", |b| {
        b.iter(|| {
            for line in layout.lines() {
                for item in line.items() {
                    if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                        render_glyph_run(&mut renderer, &glyph_run);
                    }
                }
            }
        })
    });
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Brush {}

struct GlyphBenchRenderer {
    strip_generator: StripGenerator,
}

impl GlyphBenchRenderer {
    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self> {
        GlyphRunBuilder::new(font.clone(), Affine::IDENTITY, self)
    }
}

impl GlyphRenderer for GlyphBenchRenderer {
    fn fill_glyph(&mut self, glyph: vello_common::glyph::PreparedGlyph<'_>) {
        match glyph.glyph_type {
            GlyphType::Outline(outline_glyph) => {
                self.strip_generator.generate_filled_path(
                    outline_glyph.path,
                    Fill::NonZero,
                    glyph.transform,
                    Some(128),
                    |strips| {
                        black_box(strips);
                    },
                );
            }
            GlyphType::Bitmap(_) => {}
            GlyphType::Colr(_) => {}
        }
    }

    fn stroke_glyph(&mut self, _glyph: vello_common::glyph::PreparedGlyph<'_>) {
        // We only care about filled glyphs for now.
        unimplemented!()
    }
}

fn render_glyph_run(renderer: &mut GlyphBenchRenderer, glyph_run: &GlyphRun<'_, Brush>) {
    let mut run_x = glyph_run.offset();
    let run_y = glyph_run.baseline();
    let glyphs = glyph_run.glyphs().map(|glyph| {
        let glyph_x = run_x + glyph.x;
        let glyph_y = run_y - glyph.y;
        run_x += glyph.advance;

        Glyph {
            id: glyph.id as u32,
            x: glyph_x,
            y: glyph_y,
        }
    });

    let run = glyph_run.run();
    renderer
        .glyph_run(run.font())
        .font_size(run.font_size())
        .hint(true)
        .fill_glyphs(glyphs);
}
