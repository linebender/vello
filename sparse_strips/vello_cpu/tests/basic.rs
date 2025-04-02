// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for basic functionality.

use crate::util::{check_ref, get_ctx, render_pixmap};
use skrifa::MetadataProvider;
use skrifa::raw::FileRef;
use std::f64::consts::PI;
use std::sync::Arc;
use vello_common::color::palette::css::{
    BEIGE, BLUE, DARK_BLUE, DARK_GREEN, GREEN, LIME, MAROON, REBECCA_PURPLE, RED, YELLOW,
};
use vello_common::glyph::Glyph;
use vello_common::kurbo::{Affine, BezPath, Circle, Join, Point, Rect, Shape, Stroke, Vec2};
use vello_common::peniko::{self, Blob, Fill};
use vello_common::peniko::{Compose, Font};
use vello_cpu::RenderContext;

mod util;

#[test]
fn empty_1x1() {
    let ctx = get_ctx(1, 1, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_5x1() {
    let ctx = get_ctx(5, 1, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_1x5() {
    let ctx = get_ctx(1, 5, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_3x10() {
    let ctx = get_ctx(3, 10, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_23x45() {
    let ctx = get_ctx(23, 45, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_50x50() {
    let ctx = get_ctx(50, 50, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_463x450() {
    let ctx = get_ctx(463, 450, true);
    render_pixmap(&ctx);
}

#[test]
fn empty_1134x1376() {
    let ctx = get_ctx(1134, 1376, true);
    render_pixmap(&ctx);
}

#[test]
fn full_cover_1() {
    let mut ctx = get_ctx(8, 8, true);

    ctx.set_paint(BEIGE.into());
    ctx.fill_path(&Rect::new(0.0, 0.0, 8.0, 8.0).to_path(0.1));

    check_ref(&ctx, "full_cover_1");
}

#[test]
fn filled_triangle() {
    let mut ctx = get_ctx(100, 100, false);

    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((95.0, 50.0));
        path.line_to((5.0, 95.0));
        path.close_path();

        path
    };

    ctx.set_paint(LIME.into());
    ctx.fill_path(&path);

    check_ref(&ctx, "filled_triangle");
}

#[test]
fn stroked_triangle() {
    let mut ctx = get_ctx(100, 100, false);
    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, 5.0));
        path.line_to((95.0, 50.0));
        path.line_to((5.0, 95.0));
        path.close_path();

        path
    };

    ctx.set_stroke(Stroke::new(3.0));
    ctx.set_paint(LIME.into());
    ctx.stroke_path(&path);

    check_ref(&ctx, "stroked_triangle");
}

#[test]
fn filled_circle() {
    let mut ctx = get_ctx(100, 100, false);
    let circle = Circle::new((50.0, 50.0), 45.0);
    ctx.set_paint(LIME.into());
    ctx.fill_path(&circle.to_path(0.1));

    check_ref(&ctx, "filled_circle");
}

#[test]
fn filled_overflowing_circle() {
    let mut ctx = get_ctx(100, 100, false);
    let circle = Circle::new((50.0, 50.0), 50.0 + 1.0);

    ctx.set_paint(LIME.into());
    ctx.fill_path(&circle.to_path(0.1));

    check_ref(&ctx, "filled_overflowing_circle");
}

#[test]
fn filled_fully_overflowing_circle() {
    let mut ctx = get_ctx(100, 100, false);
    let circle = Circle::new((50.0, 50.0), 80.0);

    ctx.set_paint(LIME.into());
    ctx.fill_path(&circle.to_path(0.1));

    check_ref(&ctx, "filled_fully_overflowing_circle");
}

#[test]
fn filled_circle_with_opacity() {
    let mut ctx = get_ctx(100, 100, false);
    let circle = Circle::new((50.0, 50.0), 45.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_path(&circle.to_path(0.1));

    check_ref(&ctx, "filled_circle_with_opacity");
}

#[test]
fn filled_overlapping_circles() {
    let mut ctx = get_ctx(100, 100, false);

    for e in [(35.0, 35.0, RED), (65.0, 35.0, GREEN), (50.0, 65.0, BLUE)] {
        let circle = Circle::new((e.0, e.1), 30.0);
        ctx.set_paint(e.2.with_alpha(0.5).into());
        ctx.fill_path(&circle.to_path(0.1));
    }

    check_ref(&ctx, "filled_overlapping_circles");
}

#[test]
fn stroked_circle() {
    let mut ctx = get_ctx(100, 100, false);
    let circle = Circle::new((50.0, 50.0), 45.0);
    let stroke = Stroke::new(3.0);

    ctx.set_paint(LIME.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(&circle.to_path(0.1));

    check_ref(&ctx, "stroked_circle");
}

/// Requires winding of the first row of tiles to be calculcated correctly for vertical lines.
#[test]
fn rectangle_above_viewport() {
    let mut ctx = get_ctx(10, 10, false);
    let rect = Rect::new(2.0, -5.0, 8.0, 8.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "rectangle_above_viewport");
}

/// Requires winding of the first row of tiles to be calculcated correctly for sloped lines.
#[test]
fn triangle_above_and_wider_than_viewport() {
    let mut ctx = get_ctx(10, 10, false);

    let path = {
        let mut path = BezPath::new();
        path.move_to((5.0, -5.0));
        path.line_to((14., 6.));
        path.line_to((-8., 6.));
        path.close_path();

        path
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_path(&path);

    check_ref(&ctx, "triangle_above_and_wider_than_viewport");
}

/// Requires winding and pixel coverage to be calculcated correctly for tiles preceding the
/// viewport in scan direction.
#[test]
fn rectangle_left_of_viewport() {
    let mut ctx = get_ctx(10, 10, false);
    let rect = Rect::new(-4.0, 3.0, 1.0, 8.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "rectangle_left_of_viewport");
}

#[test]
fn filling_nonzero_rule() {
    let mut ctx = get_ctx(100, 100, false);
    let star = crossed_line_star();

    ctx.set_paint(MAROON.into());
    ctx.fill_path(&star);

    check_ref(&ctx, "filling_nonzero_rule");
}

#[test]
fn filling_evenodd_rule() {
    let mut ctx = get_ctx(100, 100, false);
    let star = crossed_line_star();

    ctx.set_paint(MAROON.into());
    ctx.set_fill_rule(Fill::EvenOdd);
    ctx.fill_path(&star);

    check_ref(&ctx, "filling_evenodd_rule");
}

#[test]
fn filled_aligned_rect() {
    let mut ctx = get_ctx(30, 20, false);
    let rect = Rect::new(1.0, 1.0, 29.0, 19.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_aligned_rect");
}

#[test]
fn stroked_unaligned_rect() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke {
        width: 1.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_unaligned_rect");
}

#[test]
fn stroked_unaligned_rect_as_path() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0).to_path(0.1);
    let stroke = Stroke {
        width: 1.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(&rect);

    check_ref(&ctx, "stroked_unaligned_rect_as_path");
}

#[test]
fn stroked_aligned_rect() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = miter_stroke_2();

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_aligned_rect");
}

#[test]
fn overflowing_stroked_rect() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(12.5, 12.5, 17.5, 17.5);
    let stroke = Stroke {
        width: 5.0,
        join: Join::Miter,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "overflowing_stroked_rect");
}

#[test]
fn round_stroked_rect() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke::new(3.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "round_stroked_rect");
}

#[test]
fn bevel_stroked_rect() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 25.0, 25.0);
    let stroke = Stroke {
        width: 3.0,
        join: Join::Bevel,
        ..Default::default()
    };

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "bevel_stroked_rect");
}

#[test]
fn filled_unaligned_rect() {
    let mut ctx = get_ctx(30, 20, false);
    let rect = Rect::new(1.5, 1.5, 28.5, 18.5);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_unaligned_rect");
}

#[test]
fn filled_transformed_rect_1() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);

    ctx.set_transform(Affine::translate((10.0, 10.0)));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_transformed_rect_1");
}

#[test]
fn filled_transformed_rect_2() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 10.0, 10.0);

    ctx.set_transform(Affine::scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_transformed_rect_2");
}

#[test]
fn filled_transformed_rect_3() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);

    ctx.set_transform(Affine::new([2.0, 0.0, 0.0, 2.0, 5.0, 5.0]));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_transformed_rect_3");
}

#[test]
fn filled_transformed_rect_4() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(10.0, 10.0, 20.0, 20.0);

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(15.0, 15.0),
    ));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_transformed_rect_4");
}

#[test]
fn stroked_transformed_rect_1() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::translate((10.0, 10.0)));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_transformed_rect_1");
}

#[test]
fn stroked_transformed_rect_2() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(5.0, 5.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_transformed_rect_2");
}

#[test]
fn stroked_transformed_rect_3() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::new([2.0, 0.0, 0.0, 2.0, 5.0, 5.0]));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_transformed_rect_3");
}

#[test]
fn stroked_transformed_rect_4() {
    let mut ctx = get_ctx(30, 30, false);
    let rect = Rect::new(10.0, 10.0, 20.0, 20.0);
    let stroke = miter_stroke_2();

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(15.0, 15.0),
    ));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&rect);

    check_ref(&ctx, "stroked_transformed_rect_4");
}

#[test]
fn strip_inscribed_rect() {
    let mut ctx = get_ctx(30, 20, false);
    let rect = Rect::new(1.5, 9.5, 28.5, 11.5);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "strip_inscribed_rect");
}

#[test]
fn filled_vertical_hairline_rect() {
    let mut ctx = get_ctx(5, 8, false);
    let rect = Rect::new(2.25, 0.0, 2.75, 8.0);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_vertical_hairline_rect");
}

#[test]
fn filled_vertical_hairline_rect_2() {
    let mut ctx = get_ctx(10, 10, false);
    let rect = Rect::new(4.5, 0.5, 5.5, 9.5);

    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.fill_rect(&rect);

    check_ref(&ctx, "filled_vertical_hairline_rect_2");
}

#[test]
fn stroked_glyphs() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.glyph_run(&font)
        .font_size(font_size)
        .stroke_glyphs(glyphs.into_iter());

    check_ref(&ctx, "stroked_glyphs");
}

#[test]
fn skewed_glyphs() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.glyph_run(&font)
        .font_size(font_size)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.0))
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "skewed_glyphs");
}

#[test]
fn scaled_glyphs() {
    let mut ctx = get_ctx(150, 125, false);
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.glyph_run(&font)
        .font_size(font_size)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "scaled_glyphs");
}

/// ***DO NOT USE THIS OUTSIDE OF THESE TESTS***
///
/// This function is used for _TESTING PURPOSES ONLY_. If you need to layout and shape
/// text for your application, use a proper text shaping library like `Parley`.
///
/// We use this function as a convenience for testing; to get some glyphs shaped and laid
/// out in a small amount of code without having to go through the trouble of setting up a
/// full text layout pipeline, which you absolutely should do in application code.
fn layout_glyphs(text: &str, font_size: f32) -> (Font, Vec<Glyph>) {
    const ROBOTO_FONT: &[u8] = include_bytes!("../../../examples/assets/roboto/Roboto-Regular.ttf");
    let font = Font::new(Blob::new(Arc::new(ROBOTO_FONT)), 0);

    let font_ref = {
        let file_ref = FileRef::new(font.data.as_ref()).unwrap();
        match file_ref {
            FileRef::Font(f) => f,
            FileRef::Collection(collection) => collection.get(font.index).unwrap(),
        }
    };
    let font_size = skrifa::instance::Size::new(font_size);
    let axes = font_ref.axes();
    let variations: Vec<(&str, f32)> = vec![];
    let var_loc = axes.location(variations.as_slice());
    let charmap = font_ref.charmap();
    let metrics = font_ref.metrics(font_size, &var_loc);
    let line_height = metrics.ascent - metrics.descent + metrics.leading;
    let glyph_metrics = font_ref.glyph_metrics(font_size, &var_loc);

    let mut pen_x = 0_f32;
    let mut pen_y = 0_f32;

    let glyphs = text
        .chars()
        .filter_map(|ch| {
            if ch == '\n' {
                pen_y += line_height;
                pen_x = 0.0;
                return None;
            }
            let gid = charmap.map(ch).unwrap_or_default();
            let advance = glyph_metrics.advance_width(gid).unwrap_or_default();
            let x = pen_x;
            pen_x += advance;
            Some(Glyph {
                id: gid.to_u32(),
                x,
                y: pen_y,
            })
        })
        .collect::<Vec<_>>();

    (font, glyphs)
}

#[test]
fn filled_glyphs() {
    let mut ctx = get_ctx(300, 70, false);
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5).into());
    ctx.glyph_run(&font)
        .font_size(font_size)
        .fill_glyphs(glyphs.into_iter());

    check_ref(&ctx, "filled_glyphs");
}

#[test]
fn oversized_star() {
    let mut ctx = get_ctx(100, 100, true);

    // Create a star path that extends beyond the render context boundaries
    // Center it in the middle of the viewport
    let star_path = circular_star(Point::new(50., 50.), 10, 30., 90.);

    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_path(&star_path);

    let stroke = Stroke::new(2.0);
    ctx.set_paint(DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(&star_path);

    check_ref(&ctx, "oversized_star");
}

#[test]
fn clip_triangle_with_star() {
    let mut ctx: RenderContext = get_ctx(100, 100, true);

    let mut triangle_path = BezPath::new();
    triangle_path.move_to((10.0, 10.0));
    triangle_path.line_to((90.0, 20.0));
    triangle_path.line_to((20.0, 90.0));
    triangle_path.close_path();

    let stroke = Stroke::new(1.0);
    ctx.set_paint(DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(&triangle_path);

    let star_path = circular_star(Point::new(50., 50.), 13, 25., 45.);

    ctx.clip(&star_path);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_path(&triangle_path);
    ctx.finish();

    check_ref(&ctx, "clip_triangle_with_star");
}

#[test]
fn clip_rectangle_with_star_nonzero() {
    let mut ctx = get_ctx(100, 100, true);
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    // Create a self-intersecting star shape that will show the difference between fill rules
    let star_path = crossed_line_star();

    // Set the fill rule to NonZero before applying the clip
    ctx.set_fill_rule(Fill::NonZero);
    // Apply the star as a clip
    ctx.clip(&star_path);
    // Draw a rectangle that should be clipped by the star
    // The NonZero fill rule will treat self-intersecting regions as filled
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();
    check_ref(&ctx, "clip_rectangle_with_star_nonzero");
}

#[test]
fn clip_rectangle_with_star_evenodd() {
    let mut ctx = get_ctx(100, 100, true);
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    // Create a self-intersecting star shape that will show the difference between fill rules
    let star_path = crossed_line_star();

    // Set the fill rule to EvenOdd before applying the clip
    ctx.set_fill_rule(Fill::EvenOdd);
    // Apply the star as a clip
    ctx.clip(&star_path);
    // Draw a rectangle that should be clipped by the star
    // The EvenOdd rule should create a "hole" in the middle where the paths overlap
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();

    check_ref(&ctx, "clip_rectangle_with_star_evenodd");
}

#[test]
fn clip_rectangle_and_circle() {
    let mut ctx = get_ctx(100, 100, true);

    // Create first clipping region - a rectangle on the left side
    let clip_rect = Rect::new(10.0, 30.0, 50.0, 70.0);

    // Create second clipping region - a circle on the right side
    let circle_center = Point::new(65.0, 50.0);
    let circle_radius = 30.0;
    let clip_circle = Circle::new(circle_center, circle_radius).to_path(0.1);

    // Draw outlines of our clipping regions to visualize them
    let stroke = Stroke::new(1.0);
    ctx.set_paint(DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_rect(&clip_rect);
    ctx.stroke_path(&clip_circle);

    // Apply both clips
    ctx.clip(&clip_rect.to_path(0.1));
    ctx.clip(&clip_circle);

    // Then a filled rectangle that covers most of the canvas
    let large_rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&large_rect);
    ctx.finish();
    check_ref(&ctx, "clip_rectangle_and_circle");
}

#[test]
fn clip_with_translation() {
    let mut ctx = get_ctx(100, 100, true);

    // Apply a translation transform
    ctx.set_transform(Affine::translate((30.0, 30.0)));

    // Create and apply a clipping rectangle
    let clip_rect = Rect::new(0.0, 0.0, 40.0, 40.0);
    draw_clipping_outline(&mut ctx, &clip_rect.to_path(0.1));
    ctx.clip(&clip_rect.to_path(0.1));

    // Draw a rectangle that should be clipped
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();
    check_ref(&ctx, "clip_with_translation");
}

#[test]
fn clip_with_scale() {
    let mut ctx = get_ctx(100, 100, true);

    ctx.set_transform(Affine::scale(2.0));

    // Create and apply a clipping rectangle
    let clip_rect = Rect::new(10.0, 10.0, 40.0, 40.0);
    draw_clipping_outline(&mut ctx, &clip_rect.to_path(0.1));
    ctx.clip(&clip_rect.to_path(0.1));

    // Draw a rectangle that should be clipped
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();

    check_ref(&ctx, "clip_with_scale");
}

#[test]
fn clip_with_rotate() {
    let mut ctx = get_ctx(100, 100, true);

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(50.0, 50.0),
    ));

    // Create and apply a clipping rectangle
    let clip_rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    draw_clipping_outline(&mut ctx, &clip_rect.to_path(0.1));
    ctx.clip(&clip_rect.to_path(0.1));

    // Draw a rectangle that should be clipped
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();

    check_ref(&ctx, "clip_with_rotate");
}

#[test]
fn clip_transformed_rect() {
    let mut ctx = get_ctx(100, 100, true);

    let clip_rect = Rect::new(20.0, 20.0, 80.0, 80.0);

    draw_clipping_outline(&mut ctx, &clip_rect.to_path(0.1));

    ctx.clip(&clip_rect.to_path(0.1));

    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(50.0, 50.0),
    ));

    // Draw a smaller rectangle that should be clipped
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();

    check_ref(&ctx, "clip_transformed_rect");
}

#[test]
fn clip_with_multiple_transforms() {
    let mut ctx = get_ctx(100, 100, true);

    // Apply initial transform
    ctx.set_transform(Affine::rotate_about(
        45.0 * PI / 180.0,
        Point::new(50.0, 50.0),
    ));

    // Create and apply first clip
    let clip_rect1 = Rect::new(20.0, 20.0, 80.0, 80.0);
    draw_clipping_outline(&mut ctx, &clip_rect1.to_path(0.1));
    ctx.clip(&clip_rect1.to_path(0.1));

    // Apply another transform
    ctx.set_transform(Affine::scale(1.5));

    // Create and apply second clip
    let clip_rect2 = Rect::new(30.0, 30.0, 70.0, 70.0);
    draw_clipping_outline(&mut ctx, &clip_rect2.to_path(0.1));
    ctx.clip(&clip_rect2.to_path(0.1));

    // Draw a rectangle that should be clipped by both regions
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);
    ctx.finish();

    check_ref(&ctx, "clip_with_multiple_transforms");
}

#[test]
fn clip_with_save_restore() {
    let mut ctx = get_ctx(100, 100, true);

    // Create first clipping region - a rectangle on the left side
    let clip_rect1 = Rect::new(10.0, 30.0, 50.0, 70.0);
    draw_clipping_outline(&mut ctx, &clip_rect1.to_path(0.1));
    ctx.clip(&clip_rect1.to_path(0.1));

    // Save the state after first clip
    ctx.save();

    // Add second clipping region - a circle on the right side
    let circle_center = Point::new(65.0, 50.0);
    let circle_radius = 30.0;
    let clip_circle = Circle::new(circle_center, circle_radius).to_path(0.1);
    draw_clipping_outline(&mut ctx, &clip_circle);
    ctx.clip(&clip_circle);

    // Draw a rectangle that should be clipped by both regions
    let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(REBECCA_PURPLE.into());
    ctx.fill_rect(&rect);

    // Restore to state before second clip
    ctx.restore();

    // Draw another rectangle that should only be clipped by the first region
    let rect2 = Rect::new(0.0, 0.0, 100.0, 100.0);
    ctx.set_paint(DARK_GREEN.with_alpha(0.5).into());
    ctx.fill_rect(&rect2);
    ctx.finish();
    check_ref(&ctx, "clip_with_save_restore");
}

fn draw_clipping_outline(ctx: &mut RenderContext, path: &BezPath) {
    let stroke = Stroke::new(1.0);
    ctx.set_paint(DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(path);
}

fn miter_stroke_2() -> Stroke {
    Stroke {
        width: 2.0,
        join: Join::Miter,
        ..Default::default()
    }
}

fn bevel_stroke_2() -> Stroke {
    Stroke {
        width: 2.0,
        join: Join::Bevel,
        ..Default::default()
    }
}

fn compose_destination() -> RenderContext {
    let mut ctx = get_ctx(50, 50, true);
    let rect = Rect::new(4.5, 4.5, 35.5, 35.5);
    ctx.set_paint(YELLOW.with_alpha(0.35).into());
    ctx.set_stroke(bevel_stroke_2());
    ctx.fill_rect(&rect);

    ctx
}

fn compose_source(ctx: &mut RenderContext) {
    let rect = Rect::new(14.5, 14.5, 45.5, 45.5);
    ctx.set_paint(DARK_GREEN.with_alpha(0.8).into());
    ctx.fill_rect(&rect);
}

macro_rules! compose_impl {
    ($mode:path, $name:expr) => {
        let mut ctx = compose_destination();
        ctx.set_blend_mode(peniko::BlendMode::new(peniko::Mix::Normal, $mode));
        compose_source(&mut ctx);

        check_ref(&ctx, $name);
    };
}

#[test]
fn compose_solid_src_over() {
    compose_impl!(Compose::SrcOver, "compose_solid_src_over");
}

fn crossed_line_star() -> BezPath {
    let mut path = BezPath::new();
    path.move_to((50.0, 10.0));
    path.line_to((75.0, 90.0));
    path.line_to((10.0, 40.0));
    path.line_to((90.0, 40.0));
    path.line_to((25.0, 90.0));
    path.line_to((50.0, 10.0));

    path
}

fn circular_star(center: Point, n: usize, inner: f64, outer: f64) -> BezPath {
    let mut path = BezPath::new();
    let start_angle = -std::f64::consts::FRAC_PI_2;
    path.move_to(center + outer * Vec2::from_angle(start_angle));
    for i in 1..n * 2 {
        let th = start_angle + i as f64 * std::f64::consts::PI / n as f64;
        let r = if i % 2 == 0 { outer } else { inner };
        path.line_to(center + r * Vec2::from_angle(th));
    }
    path.close_path();
    path
}
