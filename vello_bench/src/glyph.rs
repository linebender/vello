// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::rc::Rc;
use std::sync::Arc;

use crate::harness::Registry;
use parley::fontique::Blob;
use parley::{
    Alignment, AlignmentOptions, FontContext, FontFamily, GlyphRun, Layout, LayoutContext,
    PositionedLayoutItem,
};
use vello_common::pixmap::Pixmap;
use vello_cpu::{Glyph, RenderContext, RenderSettings, Resources};

const WIDTH: u16 = 256;
const HEIGHT: u16 = 256;

pub fn register(registry: &mut Registry) {
    registry.extended(|registry| {
        const TEXT: &str = "The quick brown fox jumps over the lazy dog 0123456789";

        let mut layout_cx = LayoutContext::new();
        let mut font_cx = FontContext::new();
        font_cx.collection.register_fonts(
            Blob::new(Arc::new(
                include_bytes!("../../assets/roboto/Roboto-Regular.ttf").to_vec(),
            )),
            None,
        );
        let mut builder = layout_cx.ranged_builder(&mut font_cx, TEXT, 1.0, true);
        builder.push_default(FontFamily::named("Roboto"));
        let mut layout: Layout<Brush> = builder.build(TEXT);
        layout.break_all_lines(Some(WIDTH as f32));
        layout.align(Alignment::Start, AlignmentOptions::default());

        let settings = RenderSettings::default();
        let layout = Rc::new(layout);
        for (hint_name, hint) in [("hinted", true), ("unhinted", false)] {
            register_glyph_case(
                registry,
                format!("glyph/cached_{hint_name}"),
                Rc::clone(&layout),
                settings,
                hint,
                true,
            );

            // "uncached" disables atlas caching; outline and hint caches remain active.
            register_glyph_case(
                registry,
                format!("glyph/uncached_{hint_name}"),
                Rc::clone(&layout),
                settings,
                hint,
                false,
            );
        }
    });
}

fn register_glyph_case(
    registry: &mut Registry,
    id: String,
    layout: Rc<Layout<Brush>>,
    settings: RenderSettings,
    hint: bool,
    atlas_cache: bool,
) {
    registry.add(id, move |b| {
        let mut renderer = GlyphBenchRenderer::new(WIDTH, HEIGHT, settings);
        render_layout(&mut renderer, &layout, hint, atlas_cache);

        b.iter(|| {
            renderer.ctx.reset();
            render_layout(&mut renderer, &layout, hint, atlas_cache);
            std::hint::black_box(&renderer.pixmap);
        });
    });
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
        .render(&mut renderer.pixmap, &mut renderer.resources);
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
            id: glyph.id,
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
        .fill_glyphs(glyphs)
        .unwrap();
}
