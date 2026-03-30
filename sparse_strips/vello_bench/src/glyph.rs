// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::time::{Duration, Instant};

use criterion::Criterion;
use glifo::{Glyph, GlyphRenderer, GlyphRunBuilder, GlyphType, ImageCache};
use parley::{
    Alignment, AlignmentOptions, Font, FontContext, FontFamily, GlyphRun, Layout, LayoutContext,
    PositionedLayoutItem,
};
use vello_common::fearless_simd::Level;
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
            Level::try_detect().unwrap_or(Level::baseline()),
        ),
        strip_storage: StripStorage::default(),
        image_cache: ImageCache::new_with_config(Default::default()),
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
        g.bench_function(format!("direct_{hint_name}"), |b| {
            let layout = layout_for(TEXT, 1.0);

            b.iter_custom(|iters| {
                let mut total_time = Duration::from_nanos(0);
                for _ in 0..iters {
                    renderer.strip_storage.clear();

                    let start = Instant::now();
                    render_layout(&mut renderer, &layout, hint);
                    total_time += start.elapsed();
                }
                total_time
            });
        });
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Brush {}

#[derive(Default)]
struct NoOpGlyphCache;

impl glifo::GlyphCache for NoOpGlyphCache {
    fn get(&mut self, _key: &glifo::GlyphCacheKey) -> Option<glifo::AtlasSlot> {
        None
    }

    fn insert(
        &mut self,
        _image_cache: &mut ImageCache,
        _key: glifo::GlyphCacheKey,
        _raster_metrics: glifo::RasterMetrics,
    ) -> Option<(u16, u16, glifo::AtlasSlot, &mut glifo::AtlasCommandRecorder)> {
        unreachable!("atlas cache is disabled in direct glyph benchmarks")
    }

    fn push_pending_upload(
        &mut self,
        _image_id: vello_common::paint::ImageId,
        _pixmap: std::sync::Arc<vello_common::pixmap::Pixmap>,
        _atlas_slot: glifo::AtlasSlot,
    ) {
    }

    fn drain_pending_uploads(&mut self) -> impl Iterator<Item = glifo::atlas::PendingBitmapUpload> + '_ {
        std::iter::empty()
    }

    fn replay_pending_atlas_commands(&mut self, _f: impl FnMut(&mut glifo::AtlasCommandRecorder)) {}

    fn drain_pending_clear_rects(&mut self) -> impl Iterator<Item = glifo::PendingClearRect> + '_ {
        std::iter::empty()
    }

    fn maintain(&mut self, _image_cache: &mut ImageCache) {}

    fn clear(&mut self) {}

    fn len(&self) -> usize {
        0
    }

    fn is_empty(&self) -> bool {
        true
    }

    fn cache_hits(&self) -> u64 {
        0
    }

    fn cache_misses(&self) -> u64 {
        0
    }

    fn clear_stats(&mut self) {}

    fn config(&self) -> &glifo::GlyphCacheConfig {
        static CONFIG: glifo::GlyphCacheConfig = glifo::GlyphCacheConfig {
            max_entry_age: 64,
            eviction_frequency: 64,
            max_cached_font_size: 128.0,
        };
        &CONFIG
    }
}

struct GlyphBenchRenderer {
    strip_generator: StripGenerator,
    strip_storage: StripStorage,
    image_cache: ImageCache,
}

impl GlyphBenchRenderer {
    fn glyph_run(font: &Font) -> GlyphRunBuilder<'_> {
        GlyphRunBuilder::new(font.clone(), Affine::IDENTITY).atlas_cache(false)
    }
}

impl GlyphRenderer<NoOpGlyphCache> for GlyphBenchRenderer {
    fn fill_glyph(
        &mut self,
        glyph: glifo::PreparedGlyph<'_>,
        _glyph_atlas: &mut NoOpGlyphCache,
        _image_cache: &mut ImageCache,
    ) {
        match glyph.glyph_type {
            GlyphType::Outline(outline_glyph) => {
                self.strip_generator.generate_filled_path(
                    outline_glyph.path.elements().iter().copied(),
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

    fn stroke_glyph(
        &mut self,
        _glyph: glifo::PreparedGlyph<'_>,
        _glyph_atlas: &mut NoOpGlyphCache,
        _image_cache: &mut ImageCache,
    ) {
        unimplemented!()
    }

    fn render_cached_glyph(
        &mut self,
        _cached_slot: glifo::AtlasSlot,
        _transform: Affine,
        _glyph_type: glifo::CachedGlyphType,
    ) {
    }

    fn fill_rect(&mut self, _rect: vello_common::kurbo::Rect) {}

    fn get_context_color(&self) -> vello_common::color::AlphaColor<vello_common::color::Srgb> {
        vello_common::peniko::color::palette::css::BLACK
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
    let glyphs: Vec<_> = glyph_run
        .glyphs()
        .map(|glyph| {
            let glyph_x = run_x + glyph.x;
            let glyph_y = run_y - glyph.y;
            run_x += glyph.advance;

            Glyph {
                id: glyph.id as u32,
                x: glyph_x,
                y: glyph_y,
            }
        })
        .collect();

    let run = glyph_run.run();
    let mut glyph_caches = glifo::GlyphCaches {
        hinting_cache: Default::default(),
        outline_cache: Default::default(),
        underline_exclusions: Default::default(),
        glyph_atlas: NoOpGlyphCache,
    };
    let mut image_cache =
        std::mem::replace(&mut renderer.image_cache, ImageCache::new_with_config(Default::default()));
    GlyphBenchRenderer::glyph_run(run.font())
        .font_size(run.font_size())
        .hint(hint)
        .build(glyphs.into_iter(), &mut glyph_caches, &mut image_cache)
        .fill_glyphs(renderer);
    renderer.image_cache = image_cache;
}
