// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::time::{Duration, Instant};

use criterion::Criterion;
use parley::{
    Alignment, AlignmentOptions, FontContext, FontFamily, GlyphRun, Layout, LayoutContext,
    PositionedLayoutItem,
};
use vello_common::pixmap::Pixmap;
use vello_cpu::{Glyph, RenderContext, RenderMode, RenderSettings, Resources};

pub fn glyph(c: &mut Criterion) {
    let mut g = c.benchmark_group("glyph");

    const WIDTH: u16 = 256;
    const HEIGHT: u16 = 256;
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

    let settings = RenderSettings {
        render_mode: RenderMode::OptimizeSpeed,
        ..Default::default()
    };
    let layout = layout_for(TEXT, 1.0);

    for (hint_name, hint) in [("hinted", true), ("unhinted", false)] {
        g.bench_function(format!("cached_{hint_name}"), |b| {
            let mut renderer = GlyphBenchRenderer::new(WIDTH, HEIGHT, settings);
            render_layout(&mut renderer, &layout, hint, true);

            b.iter_custom(|iters| {
                let mut total_time = Duration::from_nanos(0);
                for _ in 0..iters {
                    // Don't include `reset` time in the benchmark.
                    renderer.reset();

                    let start = Instant::now();
                    render_layout(&mut renderer, &layout, hint, true);
                    total_time += start.elapsed();
                }
                total_time
            });
        });

        // Note that even for `uncached`, the outline and hint cache will still be used. This benchmark
        // is only for testing the difference between having atlas caching enabled and disabled.

        g.bench_function(format!("uncached_{hint_name}"), |b| {
            let mut renderer = GlyphBenchRenderer::new(WIDTH, HEIGHT, settings);
            render_layout(&mut renderer, &layout, hint, false);

            b.iter_custom(|iters| {
                let mut total_time = Duration::from_nanos(0);
                for _ in 0..iters {
                    renderer.reset();

                    let start = Instant::now();
                    render_layout(&mut renderer, &layout, hint, false);
                    total_time += start.elapsed();
                }
                total_time
            });
        });
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Brush;

struct GlyphBenchRenderer {
    ctx: RenderContext,
    resources: Resources,
    pixmap: Pixmap,
}

impl GlyphBenchRenderer {
    fn new(width: u16, height: u16, settings: RenderSettings) -> Self {
        Self {
            ctx: RenderContext::new_with(width, height, settings),
            resources: Resources::new(),
            pixmap: Pixmap::new(width, height),
        }
    }

    fn reset(&mut self) {
        self.ctx.reset();
    }
}

fn render_layout(
    renderer: &mut GlyphBenchRenderer,
    layout: &Layout<Brush>,
    hint: bool,
    atlas_cache: bool,
) {
    for line in layout.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                render_glyph_run(renderer, &glyph_run, hint, atlas_cache);
            }
        }
    }

    renderer
        .ctx
        .render_to_pixmap(&mut renderer.resources, &mut renderer.pixmap);
}

fn render_glyph_run(
    renderer: &mut GlyphBenchRenderer,
    glyph_run: &GlyphRun<'_, Brush>,
    hint: bool,
    atlas_cache: bool,
) {
    let mut run_x = glyph_run.offset();
    let run_y = glyph_run.baseline();
    let glyphs = glyph_run.glyphs().map(move |glyph| {
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
        .ctx
        .glyph_run(&mut renderer.resources, run.font())
        .font_size(run.font_size())
        .hint(hint)
        .atlas_cache(atlas_cache)
        .fill_glyphs(glyphs);
}
