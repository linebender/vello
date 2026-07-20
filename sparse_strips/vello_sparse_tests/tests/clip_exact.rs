// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Byte-exactness pins for `vello_hybrid`'s integer-rectangle clip fast path
//! and `Scene::set_cull_hint`.
//!
//! An axis-aligned clip rectangle on integer device coordinates has a 0/255
//! anti-aliasing step at its edges, so clamping fast-path rects to it must be
//! byte-identical to the mask-intersection path — and content strictly inside
//! the clip must be byte-identical to the unclipped scene. These tests render
//! the same content with and without such clips and compare raw pixels
//! (fractional content edges included: those are the bytes that would drift
//! if clipping demoted rects from the GPU analytic path to CPU strips).

#[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
mod tests {
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::{Affine, BezPath, Rect, Shape, Stroke};
    use vello_common::peniko::{Color, ImageQuality};
    use vello_common::pixmap::Pixmap;
    use vello_hybrid::SampleRect;

    use crate::load_image;
    use crate::renderer::{HybridRenderer, Renderer};
    use crate::util::{get_ctx, render_pixmap};
    use vello_cpu::RenderMode;

    const W: u16 = 128;
    const H: u16 = 64;

    fn hybrid() -> HybridRenderer {
        get_ctx::<HybridRenderer>(
            W,
            H,
            false,
            0,
            "fallback",
            RenderMode::OptimizeQuality,
            false,
        )
    }

    fn rect_path(x0: f64, y0: f64, x1: f64, y1: f64) -> BezPath {
        Rect::new(x0, y0, x1, y1).to_path(0.1)
    }

    fn paint_backdrop(ctx: &mut HybridRenderer) {
        ctx.set_paint(Color::new([0.086, 0.106, 0.133, 1.0]));
        ctx.fill_rect(&Rect::new(0.0, 0.0, W as f64, H as f64));
    }

    /// Fractional-edge content of every fast-path class: opaque rect,
    /// translucent rect, non-rect path, stroke.
    fn paint_content(ctx: &mut HybridRenderer) {
        ctx.set_paint(Color::new([0.941, 0.533, 0.243, 1.0]));
        ctx.fill_rect(&Rect::new(10.3, 8.7, 40.6, 30.2));
        ctx.set_paint(Color::new([0.2, 0.8, 0.4, 0.5]));
        ctx.fill_rect(&Rect::new(60.5, 12.25, 95.75, 40.5));
        let mut p = BezPath::new();
        p.move_to((100.3, 10.4));
        p.line_to((120.7, 18.9));
        p.line_to((104.2, 44.6));
        p.close_path();
        ctx.set_paint(Color::new([0.4, 0.4, 0.9, 1.0]));
        ctx.fill_path(&p);
        ctx.set_paint(Color::new([0.9, 0.2, 0.3, 0.8]));
        ctx.set_stroke(Stroke::new(2.5));
        ctx.stroke_path(&rect_path(30.6, 35.3, 70.4, 55.8));
    }

    fn pixel(pixmap: &Pixmap, x: u16, y: u16) -> &[u8] {
        let i = (usize::from(y) * usize::from(W) + usize::from(x)) * 4;
        &pixmap.data_as_u8_slice()[i..i + 4]
    }

    fn assert_identical(a: &Pixmap, b: &Pixmap, what: &str) {
        let diffs = a
            .data_as_u8_slice()
            .iter()
            .zip(b.data_as_u8_slice())
            .filter(|(x, y)| x != y)
            .count();
        assert_eq!(diffs, 0, "{what}: {diffs} differing bytes");
    }

    /// A full-viewport integer-aligned rect clip removes nothing, so it must
    /// change nothing.
    #[test]
    fn clip_exact_full_viewport_integer_rect_is_byte_identical() {
        let mut plain = hybrid();
        paint_backdrop(&mut plain);
        paint_content(&mut plain);
        let plain = render_pixmap(&mut plain);

        let mut clipped = hybrid();
        paint_backdrop(&mut clipped);
        clipped.push_clip_path(&rect_path(0.0, 0.0, W as f64, H as f64));
        paint_content(&mut clipped);
        clipped.pop_clip_path();
        let clipped = render_pixmap(&mut clipped);

        assert_identical(&plain, &clipped, "full-viewport integer clip");
    }

    /// An interior integer-rect clip: bytes inside the clip equal the
    /// unclipped render, bytes outside are exactly the backdrop (content
    /// removed — including the straddling shapes' out-of-clip parts).
    #[test]
    fn clip_exact_interior_integer_rect_confines_and_preserves() {
        const CLIP: [u16; 4] = [16, 12, 112, 56]; // x0, y0, x1, y1

        let mut reference = hybrid();
        paint_backdrop(&mut reference);
        paint_content(&mut reference);
        let reference = render_pixmap(&mut reference);

        let mut backdrop_only = hybrid();
        paint_backdrop(&mut backdrop_only);
        let backdrop_only = render_pixmap(&mut backdrop_only);

        let mut clipped = hybrid();
        paint_backdrop(&mut clipped);
        clipped.push_clip_path(&rect_path(
            CLIP[0] as f64,
            CLIP[1] as f64,
            CLIP[2] as f64,
            CLIP[3] as f64,
        ));
        paint_content(&mut clipped);
        clipped.pop_clip_path();
        let clipped = render_pixmap(&mut clipped);

        for y in 0..H {
            for x in 0..W {
                let inside = x >= CLIP[0] && x < CLIP[2] && y >= CLIP[1] && y < CLIP[3];
                let want = if inside {
                    pixel(&reference, x, y)
                } else {
                    pixel(&backdrop_only, x, y)
                };
                assert_eq!(
                    pixel(&clipped, x, y),
                    want,
                    "({x},{y}) inside={inside}: clip must {}",
                    if inside {
                        "preserve the unclipped bytes"
                    } else {
                        "remove the content"
                    }
                );
            }
        }
    }

    /// A clip path made of two disjoint integer rectangles: content clamps to
    /// each rect (a straddling shape splits, without double-blending), and
    /// the gap between the rects keeps the backdrop.
    #[test]
    fn clip_exact_disjoint_rect_set_confines_and_preserves() {
        const LEFT: [u16; 4] = [8, 4, 52, 60];
        const RIGHT: [u16; 4] = [70, 4, 124, 48];
        let inside = |x: u16, y: u16| {
            (x >= LEFT[0] && x < LEFT[2] && y >= LEFT[1] && y < LEFT[3])
                || (x >= RIGHT[0] && x < RIGHT[2] && y >= RIGHT[1] && y < RIGHT[3])
        };

        let mut reference = hybrid();
        paint_backdrop(&mut reference);
        paint_content(&mut reference);
        let reference = render_pixmap(&mut reference);

        let mut backdrop_only = hybrid();
        paint_backdrop(&mut backdrop_only);
        let backdrop_only = render_pixmap(&mut backdrop_only);

        let mut clip = rect_path(
            LEFT[0] as f64,
            LEFT[1] as f64,
            LEFT[2] as f64,
            LEFT[3] as f64,
        );
        clip.extend(
            rect_path(
                RIGHT[0] as f64,
                RIGHT[1] as f64,
                RIGHT[2] as f64,
                RIGHT[3] as f64,
            )
            .iter(),
        );
        let mut clipped = hybrid();
        paint_backdrop(&mut clipped);
        clipped.push_clip_path(&clip);
        paint_content(&mut clipped);
        clipped.pop_clip_path();
        let clipped = render_pixmap(&mut clipped);

        for y in 0..H {
            for x in 0..W {
                let want = if inside(x, y) {
                    pixel(&reference, x, y)
                } else {
                    pixel(&backdrop_only, x, y)
                };
                assert_eq!(
                    pixel(&clipped, x, y),
                    want,
                    "({x},{y}) inside={}",
                    inside(x, y)
                );
            }
        }
    }

    /// Texture rects at a fractional offset under an integer clip: the
    /// fast-arm clamp must not shift sampling (paints sample by absolute
    /// position), so in-clip bytes equal the unclipped render.
    #[test]
    fn clip_exact_texture_rects_are_byte_identical_inside() {
        const CLIP: [u16; 4] = [16, 12, 112, 56];
        let sample = SampleRect {
            source_region: RectU16::new(0, 0, 56, 56),
            transform: Affine::translate((10.3, 8.7)),
        };

        let mut reference = hybrid();
        paint_backdrop(&mut reference);
        let id = reference.register_external_texture(load_image!("glyphs_colr_noto"));
        reference.draw_texture_rects(id, ImageQuality::Low, [sample]);
        let reference = render_pixmap(&mut reference);

        let mut backdrop_only = hybrid();
        paint_backdrop(&mut backdrop_only);
        let backdrop_only = render_pixmap(&mut backdrop_only);

        let mut clipped = hybrid();
        paint_backdrop(&mut clipped);
        let id = clipped.register_external_texture(load_image!("glyphs_colr_noto"));
        clipped.push_clip_path(&rect_path(
            CLIP[0] as f64,
            CLIP[1] as f64,
            CLIP[2] as f64,
            CLIP[3] as f64,
        ));
        clipped.draw_texture_rects(id, ImageQuality::Low, [sample]);
        clipped.pop_clip_path();
        let clipped = render_pixmap(&mut clipped);

        for y in 0..H {
            for x in 0..W {
                let inside = x >= CLIP[0] && x < CLIP[2] && y >= CLIP[1] && y < CLIP[3];
                let want = if inside {
                    pixel(&reference, x, y)
                } else {
                    pixel(&backdrop_only, x, y)
                };
                assert_eq!(pixel(&clipped, x, y), want, "({x},{y}) inside={inside}");
            }
        }
    }

    /// `Scene::set_cull_hint` changes no in-hint pixel — including content
    /// under a (non-rect) clip inside the hint, which exercises the
    /// hint ∩ clip-bbox composition. Out-of-hint pixels are undefined by
    /// contract and deliberately not compared.
    #[test]
    fn clip_exact_cull_hint_preserves_in_hint_bytes() {
        const HINT: RectU16 = RectU16::new(16, 8, 96, 56);

        let paint = |ctx: &mut HybridRenderer| {
            paint_backdrop(ctx);
            paint_content(ctx);
            let mut tri = BezPath::new();
            tri.move_to((20.0, 15.0));
            tri.line_to((90.0, 20.0));
            tri.line_to((30.0, 50.0));
            tri.close_path();
            ctx.push_clip_path(&tri);
            ctx.set_paint(Color::new([0.95, 0.85, 0.2, 0.7]));
            ctx.fill_rect(&Rect::new(18.4, 14.6, 88.2, 48.9));
            ctx.pop_clip_path();
        };

        let mut unhinted = hybrid();
        paint(&mut unhinted);
        let unhinted = render_pixmap(&mut unhinted);

        let mut hinted = hybrid();
        hinted.scene_mut().set_cull_hint(Some(HINT));
        paint(&mut hinted);
        let hinted = render_pixmap(&mut hinted);

        for y in HINT.y0..HINT.y1 {
            for x in HINT.x0..HINT.x1 {
                assert_eq!(
                    pixel(&hinted, x, y),
                    pixel(&unhinted, x, y),
                    "({x},{y}): in-hint pixels must be bit-identical"
                );
            }
        }
    }
}
