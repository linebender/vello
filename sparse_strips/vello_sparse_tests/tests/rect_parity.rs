// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Byte-exactness pins for the two rectangle rasterizers in `vello_hybrid`.
//!
//! An axis-aligned `fill_rect` normally renders as a GPU quad (the rect fast path). As soon as a
//! clip is active it renders through the CPU strip path instead. These are two different
//! rasterizers for the same rectangle, and both quantize anti-aliased coverage to 8 bits — if
//! they round differently, pushing a clip changes pixels the clip does not even touch.
//!
//! The GPU quad reconstructs per-pixel alpha from the exact bytes the CPU strip renderer would
//! write (see `vello_common::rect` and `RectPart` in `vello_hybrid`), so a clip that removes
//! nothing must change nothing. These tests pin that: they render fractional-edge content with
//! and without integer-rectangle clips and compare raw bytes. An integer clip edge is a 0/255
//! coverage step, so inside the clip the strip path is a pure pass-through and any byte
//! difference is a rasterizer divergence.

#[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
mod tests {
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::{BezPath, Rect, Shape, Stroke};
    use vello_common::peniko::{Color, Extend, ImageQuality, ImageSampler};
    use vello_common::pixmap::Pixmap;
    use vello_hybrid::ExternalTextureRect;

    use crate::load_image;
    use crate::renderer::{HybridRenderer, Renderer};
    use crate::util::{get_ctx, render_pixmap};
    use vello_cpu::RenderMode;

    const W: u16 = 128;
    const H: u16 = 64;

    fn hybrid(width: u16, height: u16) -> HybridRenderer {
        get_ctx::<HybridRenderer>(
            width,
            height,
            false,
            0,
            "fallback",
            RenderMode::OptimizeQuality,
        )
    }

    fn rect_path(x0: f64, y0: f64, x1: f64, y1: f64) -> BezPath {
        Rect::new(x0, y0, x1, y1).to_path(0.1)
    }

    fn paint_backdrop(ctx: &mut HybridRenderer) {
        ctx.set_paint(Color::new([0.086, 0.106, 0.133, 1.0]));
        ctx.fill_rect(&Rect::new(
            0.0,
            0.0,
            f64::from(ctx.width()),
            f64::from(ctx.height()),
        ));
    }

    /// Fractional-edge content of every rect-fast-path class plus a non-rect path and a stroke:
    /// opaque rect, translucent rect, blurred rounded rect, filled path, stroked rect. The
    /// near-integer rect's boundary bytes all quantize to 255 while its exact top-left corner
    /// alpha is 254 — the case where a "fully covered" shortcut would diverge from strips.
    fn paint_content(ctx: &mut HybridRenderer) {
        ctx.set_paint(Color::new([0.941, 0.533, 0.243, 1.0]));
        ctx.fill_rect(&Rect::new(10.3, 8.7, 40.6, 30.2));
        ctx.set_paint(Color::new([0.15, 0.45, 0.85, 1.0]));
        ctx.fill_rect(&Rect::new(102.001, 44.001, 122.0, 60.0));
        ctx.set_paint(Color::new([0.2, 0.8, 0.4, 0.5]));
        ctx.fill_rect(&Rect::new(60.5, 12.25, 95.75, 40.5));
        ctx.set_paint(Color::new([0.3, 0.2, 0.7, 0.9]));
        ctx.fill_blurred_rounded_rect(&Rect::new(44.4, 34.6, 74.2, 58.9), 4.0, 3.0, false);
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

    fn pixel(pixmap: &Pixmap, width: u16, x: u16, y: u16) -> &[u8] {
        let i = (usize::from(y) * usize::from(width) + usize::from(x)) * 4;
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

    /// A full-viewport integer-aligned rect clip removes nothing, so it must change nothing —
    /// even though it moves every rect from the GPU quad to the CPU strip path.
    #[test]
    fn rect_parity_full_viewport_integer_clip_is_byte_identical() {
        let mut plain = hybrid(W, H);
        paint_backdrop(&mut plain);
        paint_content(&mut plain);
        let plain = render_pixmap(&mut plain);

        let mut clipped = hybrid(W, H);
        paint_backdrop(&mut clipped);
        clipped.push_clip_path(&rect_path(0.0, 0.0, f64::from(W), f64::from(H)));
        paint_content(&mut clipped);
        clipped.pop_clip_path();
        let clipped = render_pixmap(&mut clipped);

        assert_identical(&plain, &clipped, "full-viewport integer clip");
    }

    /// An interior integer-rect clip: bytes inside the clip equal the unclipped render, bytes
    /// outside are exactly the backdrop.
    #[test]
    fn rect_parity_interior_integer_clip_confines_and_preserves() {
        const CLIP: [u16; 4] = [16, 12, 112, 56]; // x0, y0, x1, y1

        let mut reference = hybrid(W, H);
        paint_backdrop(&mut reference);
        paint_content(&mut reference);
        let reference = render_pixmap(&mut reference);

        let mut backdrop_only = hybrid(W, H);
        paint_backdrop(&mut backdrop_only);
        let backdrop_only = render_pixmap(&mut backdrop_only);

        let mut clipped = hybrid(W, H);
        paint_backdrop(&mut clipped);
        clipped.push_clip_path(&rect_path(
            f64::from(CLIP[0]),
            f64::from(CLIP[1]),
            f64::from(CLIP[2]),
            f64::from(CLIP[3]),
        ));
        paint_content(&mut clipped);
        clipped.pop_clip_path();
        let clipped = render_pixmap(&mut clipped);

        for y in 0..H {
            for x in 0..W {
                let inside = x >= CLIP[0] && x < CLIP[2] && y >= CLIP[1] && y < CLIP[3];
                let want = if inside {
                    pixel(&reference, W, x, y)
                } else {
                    pixel(&backdrop_only, W, x, y)
                };
                assert_eq!(
                    pixel(&clipped, W, x, y),
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

    /// Texture rects at a fractional offset under an integer clip: in-clip bytes equal the
    /// unclipped render (sampling positions must not shift when the rect goes through strips).
    #[test]
    fn rect_parity_texture_rects_byte_identical_inside_clip() {
        const CLIP: [u16; 4] = [16, 12, 112, 56];
        let sample = |texture_id| ExternalTextureRect {
            texture_id,
            src_rect: RectU16::new(0, 0, 56, 56),
            dest_rect: Rect::new(10.3, 8.7, 10.3 + 56., 8.7 + 56.),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
            may_have_transparency: true,
        };

        let mut reference = hybrid(W, H);
        paint_backdrop(&mut reference);
        let id = reference.register_external_texture(load_image!("glyphs_colr_noto"));
        reference.draw_texture_rect(sample(id));
        let reference = render_pixmap(&mut reference);

        let mut backdrop_only = hybrid(W, H);
        paint_backdrop(&mut backdrop_only);
        let backdrop_only = render_pixmap(&mut backdrop_only);

        let mut clipped = hybrid(W, H);
        paint_backdrop(&mut clipped);
        let id = clipped.register_external_texture(load_image!("glyphs_colr_noto"));
        clipped.push_clip_path(&rect_path(
            f64::from(CLIP[0]),
            f64::from(CLIP[1]),
            f64::from(CLIP[2]),
            f64::from(CLIP[3]),
        ));
        clipped.draw_texture_rect(sample(id));
        clipped.pop_clip_path();
        let clipped = render_pixmap(&mut clipped);

        for y in 0..H {
            for x in 0..W {
                let inside = x >= CLIP[0] && x < CLIP[2] && y >= CLIP[1] && y < CLIP[3];
                let want = if inside {
                    pixel(&reference, W, x, y)
                } else {
                    pixel(&backdrop_only, W, x, y)
                };
                assert_eq!(pixel(&clipped, W, x, y), want, "({x},{y}) inside={inside}");
            }
        }
    }

    /// A deterministic sweep of sub-pixel rect geometries (sizes from sub-pixel slivers to
    /// multi-tile, opaque and translucent), rendered once plain and once under a full-viewport
    /// integer clip.
    #[test]
    fn rect_parity_randomized_fractional_rects() {
        // Simple LCG so the scene is deterministic without a rand dependency.
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / f64::from(u32::MAX >> 1)
        };

        let mut rects = Vec::new();
        for _ in 0..40 {
            let x0 = next() * f64::from(W - 12);
            let y0 = next() * f64::from(H - 12);
            let w = 0.3 + next() * 11.0;
            let h = 0.3 + next() * 11.0;
            let color = Color::new([
                next() as f32,
                next() as f32,
                next() as f32,
                (0.3 + 0.7 * next()) as f32,
            ]);
            rects.push((Rect::new(x0, y0, x0 + w, y0 + h), color));
        }

        let paint = |ctx: &mut HybridRenderer| {
            paint_backdrop(ctx);
            for (rect, color) in &rects {
                ctx.set_paint(*color);
                ctx.fill_rect(rect);
            }
        };

        let mut plain = hybrid(W, H);
        paint(&mut plain);
        let plain = render_pixmap(&mut plain);

        let mut clipped = hybrid(W, H);
        clipped.push_clip_path(&rect_path(0.0, 0.0, f64::from(W), f64::from(H)));
        paint(&mut clipped);
        clipped.pop_clip_path();
        let clipped = render_pixmap(&mut clipped);

        assert_identical(&plain, &clipped, "randomized fractional rects");
    }

    /// Same pin far from the origin. Converting an `f64` edge to `f32` moves it by up to half
    /// an `f32` ulp (about 2.4e-4 px at x ~ 4400), which crosses a byte-decision boundary for a
    /// few percent of fractional positions — so the quad's coverage bytes must be computed from
    /// the same `f32`-converted edges the strip renderer uses, or the two paths disagree here.
    #[test]
    fn rect_parity_holds_at_large_coordinates() {
        const BIG_W: u16 = 4400;
        const BIG_H: u16 = 16;

        let paint = |ctx: &mut HybridRenderer| {
            paint_backdrop(ctx);
            ctx.set_paint(Color::new([0.941, 0.533, 0.243, 1.0]));
            ctx.fill_rect(&Rect::new(4300.3, 2.7, 4390.6, 12.2));
            ctx.set_paint(Color::new([0.2, 0.8, 0.4, 0.5]));
            ctx.fill_rect(&Rect::new(4210.55, 5.25, 4285.75, 9.5));
        };

        let mut plain = hybrid(BIG_W, BIG_H);
        paint(&mut plain);
        let plain = render_pixmap(&mut plain);

        let mut clipped = hybrid(BIG_W, BIG_H);
        clipped.push_clip_path(&rect_path(0.0, 0.0, f64::from(BIG_W), f64::from(BIG_H)));
        paint(&mut clipped);
        clipped.pop_clip_path();
        let clipped = render_pixmap(&mut clipped);

        assert_identical(&plain, &clipped, "large-coordinate rects");
    }
}
