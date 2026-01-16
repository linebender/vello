// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::time::{Duration, Instant};

use criterion::Criterion;
use parley::{
    Alignment, AlignmentOptions, Font, FontContext, FontFamily, GlyphRun, Layout, LayoutContext,
    PositionedLayoutItem,
};
use vello_common::fearless_simd::Level;
use vello_common::glyph::{Glyph, GlyphCaches, GlyphRunBuilder};
use vello_common::glyph::{GlyphRenderer, GlyphType};
use vello_common::kurbo::Affine;
use vello_common::peniko::Fill;
use vello_common::strip_generator::{StripGenerator, StripStorage};

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
        strip_storage: StripStorage::default(),
        glyph_caches: None,
    };

    const TEXT: &str = "The quick brown fox jumps over the lazy dog 0123456789";

    let layout_for = |text: &str, scale: f32| {
        let mut layout_cx = LayoutContext::new();
        let mut font_cx = FontContext::new();
        let mut builder = layout_cx.ranged_builder(&mut font_cx, text, scale, true);
        builder.push_default(FontFamily::parse("Roboto").unwrap());
        let mut layout: Layout<Brush> = builder.build(text);
        let max_advance = Some(WIDTH as f32);
        layout.break_all_lines(max_advance);
        layout.align(max_advance, Alignment::Start, AlignmentOptions::default());
        layout
    };

    for (hint_name, hint) in [("hinted", true), ("unhinted", false)] {
        g.bench_function(format!("cached_{hint_name}"), |b| {
            let layout = layout_for(TEXT, 1.0);
            render_layout(&mut renderer, &layout, hint);

            b.iter_custom(|iters| {
                let mut total_time = Duration::from_nanos(0);
                for _ in 0..iters {
                    // Don't include `clear` time in the benchmark.
                    renderer.strip_storage.clear();

                    let start = Instant::now();
                    render_layout(&mut renderer, &layout, hint);
                    total_time += start.elapsed();
                }
                total_time
            });
        });

        g.bench_function(format!("uncached_{hint_name}"), |b| {
            let layout = layout_for(TEXT, 1.0);

            b.iter_custom(|iters| {
                let mut total_time = Duration::from_nanos(0);
                for _ in 0..iters {
                    // Don't include `clear` time in the benchmark.
                    renderer.glyph_caches.as_mut().unwrap().clear();
                    renderer.strip_storage.clear();

                    let start = Instant::now();
                    render_layout(&mut renderer, &layout, hint);
                    total_time += start.elapsed();
                }
                total_time
            });
        });
    }

    g.bench_function("maintain", |b| {
        let layouts = (0..10)
            .map(|i| layout_for(TEXT, 1.0 + i as f32 * 0.1))
            .collect::<Vec<_>>();

        b.iter_custom(|iters| {
            let mut total_time = Duration::from_nanos(0);
            for _ in 0..iters {
                // Prepopulate cache with enough glyphs to overflow cache bounds.
                // Don't include prepopulate time in the benchmark.
                for layout in layouts.iter() {
                    render_layout(&mut renderer, layout, true);
                }

                let start = Instant::now();
                renderer.glyph_caches.as_mut().unwrap().maintain();
                total_time += start.elapsed();
            }
            total_time
        });
    });
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Brush {}

struct GlyphBenchRenderer {
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    glyph_caches: Option<GlyphCaches>,
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
                    &mut self.strip_storage,
                    None,
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

    fn take_glyph_caches(&mut self) -> GlyphCaches {
        self.glyph_caches.take().unwrap_or_default()
    }
    fn restore_glyph_caches(&mut self, cache: GlyphCaches) {
        self.glyph_caches = Some(cache);
    }
}

fn render_layout(renderer: &mut GlyphBenchRenderer, layout: &Layout<Brush>, hint: bool) {
    for line in layout.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                render_glyph_run(renderer, &glyph_run, hint);
            }
        }
    }
}

fn render_glyph_run(
    renderer: &mut GlyphBenchRenderer,
    glyph_run: &GlyphRun<'_, Brush>,
    hint: bool,
) {
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
        .hint(hint)
        .fill_glyphs(glyphs);
}
