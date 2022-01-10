use piet::kurbo::{Point, Rect};
use piet::RenderContext;
use piet_parley::{swash, ParleyTextLayout};
use swash::scale::{ScaleContext, Scaler, StrikeWith};
use swash::zeno::{Vector, Verb};
use swash::{FontRef, GlyphId};

use crate::encoder::GlyphEncoder;
use crate::render_ctx;
use crate::stages::Transform;
use crate::PietGpuRenderContext;

pub fn draw_text(ctx: &mut PietGpuRenderContext, layout: &ParleyTextLayout, pos: Point) {
    let mut scale_ctx = ScaleContext::new();
    let tpos = render_ctx::to_f32_2(pos);
    for line in layout.layout.lines() {
        let mut last_x = 0.0;
        let mut last_y = 0.0;
        ctx.encode_transform(Transform {
            mat: [1.0, 0.0, 0.0, -1.0],
            translate: tpos,
        });
        for glyph_run in line.glyph_runs() {
            let run = glyph_run.run();
            let color = &glyph_run.style().brush.0;
            let font = run.font();
            let font = font.as_ref();
            let mut first = true;
            let mut scaler = scale_ctx.builder(font).size(run.font_size()).build();
            for glyph in glyph_run.positioned_glyphs() {
                let delta_x = glyph.x - last_x;
                let delta_y = glyph.y - last_y;
                let transform = Transform {
                    mat: [1.0, 0.0, 0.0, 1.0],
                    translate: [delta_x, -delta_y],
                };
                last_x = glyph.x;
                last_y = glyph.y;
                if first {
                    if let Some(deco) = glyph_run.style().underline.as_ref() {
                        let offset = deco.offset.unwrap_or(run.metrics().underline_offset);
                        let size = deco.size.unwrap_or(run.metrics().underline_size);
                        ctx.encode_transform(Transform {
                            mat: [1.0, 0.0, 0.0, 1.0],
                            translate: [delta_x, -(delta_y - offset)],
                        });
                        let width = glyph_run.advance();
                        let brush = ctx.solid_brush(deco.brush.0.clone());
                        ctx.fill(Rect::new(0.0, 0.0, width as _, -size as _), &brush);
                        ctx.encode_transform(Transform {
                            mat: [1.0, 0.0, 0.0, 1.0],
                            translate: [-delta_x, delta_y - offset],
                        });
                    }
                    if let Some(deco) = glyph_run.style().strikethrough.as_ref() {
                        let offset = deco.offset.unwrap_or(run.metrics().strikethrough_offset);
                        let size = deco.size.unwrap_or(run.metrics().strikethrough_size);
                        ctx.encode_transform(Transform {
                            mat: [1.0, 0.0, 0.0, 1.0],
                            translate: [delta_x, -(delta_y - offset)],
                        });
                        let width = glyph_run.advance();
                        let brush = ctx.solid_brush(deco.brush.0.clone());
                        ctx.fill(Rect::new(0.0, 0.0, width as _, -size as _), &brush);
                        ctx.encode_transform(Transform {
                            mat: [1.0, 0.0, 0.0, 1.0],
                            translate: [-delta_x, delta_y - offset],
                        });
                    }
                }
                first = false;
                ctx.encode_transform(transform);
                if let Some(glyph) = make_glyph(&font, &mut scaler, glyph.id) {
                    ctx.encode_glyph(&glyph);
                    if !glyph.is_color() {
                        ctx.fill_glyph(color.as_rgba_u32());
                    }
                }
            }
        }
        ctx.encode_transform(Transform {
            mat: [1.0, 0.0, 0.0, -1.0],
            translate: [-(tpos[0] + last_x), tpos[1] + last_y],
        });
    }
}

fn make_glyph(font: &FontRef, scaler: &mut Scaler, glyph_id: GlyphId) -> Option<GlyphEncoder> {
    let mut encoder = GlyphEncoder::default();
    if let Some(_bitmap) = scaler.scale_color_bitmap(glyph_id, StrikeWith::BestFit) {
        // TODO
    }
    if let Some(_bitmap) = scaler.scale_bitmap(glyph_id, StrikeWith::BestFit) {
        // TODO
    }
    if let Some(outline) = scaler.scale_color_outline(glyph_id) {
        // TODO: be more sophisticated choosing a palette
        let palette = font.color_palettes().next().unwrap();
        let mut i = 0;
        while let Some(layer) = outline.get(i) {
            if let Some(color_ix) = layer.color_index() {
                let color = palette.get(color_ix);
                append_outline(&mut encoder, layer.verbs(), layer.points());
                encoder.fill_color(*bytemuck::from_bytes(&color));
            }
            i += 1;
        }
        return Some(encoder);
    }
    if let Some(outline) = scaler.scale_outline(glyph_id) {
        append_outline(&mut encoder, outline.verbs(), outline.points());
        return Some(encoder);
    }
    None
}

fn append_outline(encoder: &mut GlyphEncoder, verbs: &[Verb], points: &[Vector]) {
    let mut path_encoder = encoder.path_encoder();
    let mut i = 0;
    for verb in verbs {
        match verb {
            Verb::MoveTo => {
                let p = points[i];
                path_encoder.move_to(p.x, p.y);
                i += 1;
            }
            Verb::LineTo => {
                let p = points[i];
                path_encoder.line_to(p.x, p.y);
                i += 1;
            }
            Verb::QuadTo => {
                let p1 = points[i];
                let p2 = points[i + 1];
                path_encoder.quad_to(p1.x, p1.y, p2.x, p2.y);
                i += 2;
            }
            Verb::CurveTo => {
                let p1 = points[i];
                let p2 = points[i + 1];
                let p3 = points[i + 2];
                path_encoder.cubic_to(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
                i += 3;
            }
            Verb::Close => path_encoder.close_path(),
        }
    }
    path_encoder.path();
    let n_pathseg = path_encoder.n_pathseg();
    encoder.finish_path(n_pathseg);
}
