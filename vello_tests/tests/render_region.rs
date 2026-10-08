// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for confining a render to a region of the target.

use crate::load_image;
use crate::renderer::{GpuRenderer, Renderer};
use crate::util::{get_ctx_with_depth_buffer, stops_blue_green_red_yellow};
use vello_common::color::palette::css::{BLUE, LIME, ORANGE, REBECCA_PURPLE};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, Circle, Point, Rect, Shape, Stroke};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::{
    BlendMode, Compose, Extend, Gradient, ImageQuality, ImageSampler, LinearGradientPosition, Mix,
};
use vello_common::pixmap::Pixmap;
use vello_cpu::RenderMode;
use vello_dev_macros::vello_test;
use vello_gpu::{ClearSettings, RectU16, RenderRegion, TargetInit};

/// Two disjoint rectangles with unaligned edges.
const REGION: &[RectU16] = &[RectU16::new(9, 13, 47, 51), RectU16::new(58, 54, 94, 91)];
const OVERLAPPING_REGION: &[RectU16] =
    &[RectU16::new(10, 10, 60, 60), RectU16::new(40, 40, 90, 90)];
/// Rectangles that are empty once clamped to the target.
const OUTSIDE_REGION: &[RectU16] = &[RectU16::new(100, 20, 140, 60), RectU16::new(30, 30, 30, 70)];
const CLEAR_RECTS: &[RectU16] = &[RectU16::new(0, 0, 30, 30), RectU16::new(36, 44, 100, 70)];
const CLEAR_COLOR: AlphaColor<Srgb> = AlphaColor::from_rgb8(18, 52, 86);
const TRANSLUCENT_CLEAR_COLOR: AlphaColor<Srgb> = AlphaColor::from_rgba8(18, 52, 86, 129);

/// Render the frame whose pixels a confined render must preserve outside of its region.
fn render_previous_frame(ctx: &mut impl Renderer) {
    ctx.set_render_region(RenderRegion::Viewport);
    ctx.set_target_init(TargetInit::Clear(ClearSettings::default()));
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(20.0, 20.0, 80.0, 80.0));
    ctx.flush();
    ctx.render();
    ctx.reset();
}

fn draw_frame(ctx: &mut impl Renderer) {
    ctx.set_paint(LIME.with_alpha(0.6));
    ctx.fill_path(&Circle::new((50.0, 50.0), 42.5).to_path(0.1));
    ctx.set_paint(ORANGE);
    ctx.fill_rect(&Rect::new(4.5, 30.25, 90.75, 40.5));
}

fn draw_shapes(ctx: &mut impl Renderer, image: &ImageSource) {
    draw_frame(ctx);
    // Large enough to be split into an opaque interior and anti-aliased edges.
    ctx.fill_rect(&Rect::new(11.25, 58.5, 49.75, 95.5));

    ctx.set_paint(Gradient {
        kind: LinearGradientPosition {
            start: Point::new(52.0, 0.0),
            end: Point::new(97.0, 0.0),
        }
        .into(),
        stops: stops_blue_green_red_yellow(),
        ..Default::default()
    });
    ctx.fill_rect(&Rect::new(52.5, 8.25, 97.0, 70.75));

    ctx.set_paint_transform(Affine::scale(3.3));
    ctx.set_paint(Image {
        image: image.clone(),
        sampler: ImageSampler {
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Medium,
            alpha: 1.0,
        },
    });
    ctx.fill_rect(&Rect::new(20.5, 66.0, 70.0, 98.25));
    ctx.set_paint_transform(Affine::IDENTITY);

    ctx.set_stroke(Stroke::new(2.5));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.8));
    ctx.stroke_path(&Circle::new((50.0, 50.0), 30.0).to_path(0.1));
}

fn draw_layers(ctx: &mut impl Renderer) {
    ctx.push_opacity_layer(0.7);
    draw_frame(ctx);
    ctx.pop_layer();

    ctx.push_clip_layer(&Circle::new((70.0, 70.0), 20.0).to_path(0.1));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(40.0, 40.0, 100.0, 100.0));
    ctx.pop_layer();

    // The content of this layer lies outside of `REGION` and is only moved into it by the filter.
    ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
        dx: -40.0,
        dy: 0.0,
    }));
    ctx.set_paint(ORANGE);
    ctx.fill_rect(&Rect::new(60.0, 16.0, 90.0, 48.0));
    ctx.pop_layer();
}

fn draw_root_blend(ctx: &mut impl Renderer) {
    draw_frame(ctx);
    ctx.push_blend_layer(BlendMode::new(Mix::Multiply, Compose::SrcOver));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(30.0, 30.0, 80.0, 80.0));
    ctx.pop_layer();
}

#[vello_test(skip_webgl, gpu_no_depth)]
fn render_region_confines_viewport_clear(ctx: &mut impl Renderer) {
    render_previous_frame(ctx);
    ctx.set_render_region(RenderRegion::Rects(REGION));
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Viewport {
        color: CLEAR_COLOR,
    }));
    draw_frame(ctx);
}

/// Overlapping rectangles must still composite translucent content only once.
#[vello_test(skip_webgl, transparent)]
fn render_region_overlapping_rects(ctx: &mut impl Renderer) {
    render_previous_frame(ctx);
    ctx.set_render_region(RenderRegion::Rects(OVERLAPPING_REGION));
    ctx.set_target_init(TargetInit::SrcOver);
    ctx.set_paint(LIME.with_alpha(0.5));
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
}

const EXACT_REGIONS: &[&[RectU16]] = &[
    &[RectU16::new(9, 13, 47, 51)],
    REGION,
    OVERLAPPING_REGION,
    // The two edges exposed by a diagonal scroll.
    &[RectU16::new(0, 0, 100, 7), RectU16::new(93, 7, 100, 100)],
    OUTSIDE_REGION,
];

const EXACT_TARGET_INITS: &[TargetInit<'static>] = &[
    TargetInit::SrcOver,
    TargetInit::Clear(ClearSettings::Rects {
        color: TRANSLUCENT_CLEAR_COLOR,
        rects: CLEAR_RECTS,
    }),
    // A load-op clear when unconfined, drawn rectangles when confined.
    TargetInit::Clear(ClearSettings::Viewport {
        color: TRANSLUCENT_CLEAR_COLOR,
    }),
];

/// Assert that confined renders of the scene drawn by `draw` match an unconfined render exactly
/// inside the region, and keep the previous frame outside of it.
fn assert_confined_render_is_exact(draw: impl Fn(&mut GpuRenderer, &ImageSource)) {
    // Covers the whole target, so that every region has pixels to redraw.
    let draw = |ctx: &mut GpuRenderer, image: &ImageSource| {
        ctx.set_paint(REBECCA_PURPLE.with_alpha(0.25));
        ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
        draw(ctx, image);
    };

    for use_depth_buffer in [true, false] {
        let mut ctx = get_ctx_with_depth_buffer::<GpuRenderer>(
            100,
            100,
            true,
            0,
            "baseline",
            RenderMode::OptimizeSpeed,
            use_depth_buffer,
        );
        let image = ctx.get_image_source(load_image!("rgb_image_2x3"));

        render_previous_frame(&mut ctx);
        let previous = ctx.snapshot();

        for &target_init in EXACT_TARGET_INITS {
            render_previous_frame(&mut ctx);
            draw(&mut ctx, &image);
            ctx.set_target_init(target_init);
            ctx.render();
            let unconfined = ctx.snapshot();
            ctx.reset();

            for &region in EXACT_REGIONS {
                render_previous_frame(&mut ctx);
                draw(&mut ctx, &image);
                ctx.set_target_init(target_init);
                ctx.set_render_region(RenderRegion::Rects(region));
                ctx.render();
                let confined = ctx.snapshot();
                ctx.reset();

                assert_confined_pixels(
                    &previous,
                    &unconfined,
                    &confined,
                    region,
                    &format!("{target_init:?}, {region:?}, depth buffer: {use_depth_buffer}"),
                );
            }
        }
    }
}

fn assert_confined_pixels(
    previous: &Pixmap,
    unconfined: &Pixmap,
    confined: &Pixmap,
    region: &[RectU16],
    case: &str,
) {
    let width = usize::from(previous.width());
    let mut changed_inside = false;
    let mut changed_outside = false;
    let mut has_inside = false;

    for (index, ((previous, unconfined), confined)) in previous
        .data()
        .iter()
        .zip(unconfined.data())
        .zip(confined.data())
        .enumerate()
    {
        let x = u16::try_from(index % width).unwrap();
        let y = u16::try_from(index / width).unwrap();
        let inside = region.iter().any(|rect| rect.contains(x, y));
        let expected = if inside { unconfined } else { previous };
        assert_eq!(confined, expected, "pixel ({x}, {y}) of {case}");

        has_inside |= inside;
        if unconfined != previous {
            changed_inside |= inside;
            changed_outside |= !inside;
        }
    }

    // Otherwise the test couldn't tell a confined render apart from an unconfined one.
    assert!(changed_outside, "{case}");
    assert!(changed_inside || !has_inside, "{case}");
}

#[test]
fn render_region_gpu_exact_shapes() {
    assert_confined_render_is_exact(draw_shapes);
}

#[test]
fn render_region_gpu_exact_layers() {
    assert_confined_render_is_exact(|ctx, _| {
        draw_layers(ctx);
        ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation: 3.0,
            edge_mode: EdgeMode::None,
        }));
        ctx.set_paint(REBECCA_PURPLE);
        ctx.fill_rect(&Rect::new(4.0, 70.0, 56.0, 96.0));
        ctx.pop_layer();
    });
}

#[test]
fn render_region_gpu_exact_root_blend() {
    assert_confined_render_is_exact(|ctx, _| draw_root_blend(ctx));
}

#[test]
fn render_region_gpu_exact_root_clip_rect() {
    assert_confined_render_is_exact(|ctx, image| {
        ctx.push_clip_rect(&Rect::new(9.0, 13.0, 94.0, 91.0));
        draw_shapes(ctx, image);
        ctx.pop_clip();
    });
}

/// A confined viewport clear matches a load-op clear on a float target, to within one half-float
/// step: the two conversions from `f32` may round differently.
#[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
#[test]
fn render_region_gpu_exact_float_clear() {
    const BYTES_PER_PIXEL: usize = 8;
    let clear = TargetInit::Clear(ClearSettings::Viewport {
        color: TRANSLUCENT_CLEAR_COLOR,
    });
    let mut ctx = GpuRenderer::new_with_format(100, 100, wgpu::TextureFormat::Rgba16Float);

    ctx.set_target_init(clear);
    ctx.render();
    let unconfined = ctx.read_target();

    ctx.set_target_init(TargetInit::Clear(ClearSettings::default()));
    ctx.render();
    ctx.set_target_init(clear);
    ctx.set_render_region(RenderRegion::Rects(REGION));
    ctx.render();
    let confined = ctx.read_target();

    for (index, (confined, unconfined)) in confined
        .chunks_exact(BYTES_PER_PIXEL)
        .zip(unconfined.chunks_exact(BYTES_PER_PIXEL))
        .enumerate()
    {
        let x = u16::try_from(index % 100).unwrap();
        let y = u16::try_from(index / 100).unwrap();
        let inside = REGION.iter().any(|rect| rect.contains(x, y));
        if !inside {
            assert_eq!(confined, [0; BYTES_PER_PIXEL], "pixel ({x}, {y})");
            continue;
        }
        for (c, u) in confined.chunks_exact(2).zip(unconfined.chunks_exact(2)) {
            let c = u16::from_le_bytes([c[0], c[1]]);
            let u = u16::from_le_bytes([u[0], u[1]]);
            assert!(c.abs_diff(u) <= 1, "pixel ({x}, {y}): {c:#06x} vs {u:#06x}");
        }
    }
}
