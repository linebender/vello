// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use vello_common::blurred_rounded_rect::painted_rect;
use vello_common::color::palette::css::{BLACK, PALE_GOLDENROD, REBECCA_PURPLE};
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, BezPath, Circle, Point, Rect, RoundedRect, Shape, Vec2};
use vello_common::peniko::Fill;
use vello_common::pixmap::Pixmap;
use vello_cpu::{RasterizerSettings, RenderContext, RenderMode, RenderSettings, Resources};
use vello_dev_macros::vello_test;

fn rect_with(ctx: &mut impl Renderer, radius: f32, std_dev: f32, affine: Affine) {
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_transform(affine);
    ctx.fill_blurred_rounded_rect(&rect, radius, std_dev);
}

#[vello_test]
fn blurred_rounded_rect_zero(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 0.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_zero_with_radius(ctx: &mut impl Renderer) {
    rect_with(ctx, 10.0, 0.0, Affine::IDENTITY);
}

/// A negative standard deviation is treated as zero: this draws what
/// `blurred_rounded_rect_zero_with_radius` draws.
#[vello_test]
fn blurred_rounded_rect_negative_std_dev(ctx: &mut impl Renderer) {
    rect_with(ctx, 10.0, -5.0, Affine::IDENTITY);
}

/// A NaN standard deviation is treated as zero: this draws what
/// `blurred_rounded_rect_zero_with_radius` draws.
#[vello_test]
fn blurred_rounded_rect_nan_std_dev(ctx: &mut impl Renderer) {
    rect_with(ctx, 10.0, f32::NAN, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_none(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 0.1, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_small_std_dev(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 5.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_medium_std_dev(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_large_std_dev(ctx: &mut impl Renderer) {
    rect_with(ctx, 0.0, 20.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_with_radius(ctx: &mut impl Renderer) {
    rect_with(ctx, 10.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_with_large_radius(ctx: &mut impl Renderer) {
    rect_with(ctx, 30.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_with_transform(ctx: &mut impl Renderer) {
    rect_with(
        ctx,
        10.0,
        10.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
    );
}

#[vello_test]
fn blurred_rounded_rect_thin(ctx: &mut impl Renderer) {
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_blurred_rounded_rect(&Rect::new(20.0, 38.0, 80.0, 62.0), 8.0, 8.0);
}

fn inverse_rect_with(ctx: &mut impl Renderer, radius: f32, std_dev: f32, affine: Affine) {
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    // Extend the path to the blur's extent, so the fully covered area outside of `rect` is painted.
    let path = painted_rect(rect, std_dev).to_path(0.1);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_transform(affine);
    ctx.fill_blurred_rounded_rect_in(&path, &rect, radius, std_dev, true);
}

#[vello_test]
fn inverse_blurred_rounded_rect_small_std_dev(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 0.0, 5.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_medium_std_dev(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 0.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_large_std_dev(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 0.0, 20.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_with_radius(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 10.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_with_large_radius(ctx: &mut impl Renderer) {
    inverse_rect_with(ctx, 30.0, 10.0, Affine::IDENTITY);
}

#[vello_test]
fn inverse_blurred_rounded_rect_with_transform(ctx: &mut impl Renderer) {
    inverse_rect_with(
        ctx,
        10.0,
        10.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
    );
}

fn rect_in_with(
    ctx: &mut impl Renderer,
    path: &BezPath,
    radius: f32,
    std_dev: f32,
    invert: bool,
    affine: Affine,
) {
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_transform(affine);
    ctx.fill_blurred_rounded_rect_in(path, &rect, radius, std_dev, invert);
}

#[vello_test]
fn blurred_rounded_rect_in_circle(ctx: &mut impl Renderer) {
    let path = Circle::new(Point::new(50.0, 50.0), 45.0).to_path(0.1);
    rect_in_with(ctx, &path, 10.0, 10.0, false, Affine::IDENTITY);
}

#[vello_test]
fn blurred_rounded_rect_in_circle_with_transform(ctx: &mut impl Renderer) {
    let path = Circle::new(Point::new(50.0, 50.0), 45.0).to_path(0.1);
    rect_in_with(
        ctx,
        &path,
        10.0,
        10.0,
        false,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
    );
}

/// The typical inset box shadow use case: the inverse blur is clipped to the shape it is
/// inset into.
#[vello_test]
fn inverse_blurred_rounded_rect_in_rect(ctx: &mut impl Renderer) {
    let path = Rect::new(20.0, 20.0, 80.0, 80.0).to_path(0.1);
    rect_in_with(ctx, &path, 10.0, 10.0, true, Affine::IDENTITY);
}

/// Emulate a CSS inset box shadow: the inverse blurred rounded rectangle is shrunk by the spread
/// and offset relative to the border box it is painted into, and clipped to that border box.
fn inset_box_shadow(
    ctx: &mut impl Renderer,
    offset: Vec2,
    spread: f32,
    radius: f32,
    std_dev: f32,
    affine: Affine,
) {
    let border_box = Rect::new(20.0, 20.0, 80.0, 80.0);
    let path = RoundedRect::from_rect(border_box, f64::from(radius)).to_path(0.1);
    let spread = f64::from(spread);
    let shadow_rect = border_box.inflate(-spread, -spread) + offset;
    let shadow_radius = (radius - spread as f32).max(0.0);

    ctx.set_transform(affine);
    ctx.set_paint(PALE_GOLDENROD);
    ctx.fill_path(&path);

    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_blurred_rounded_rect_in(&path, &shadow_rect, shadow_radius, std_dev, true);
}

#[vello_test]
fn inset_box_shadow_offset_down_right(ctx: &mut impl Renderer) {
    inset_box_shadow(ctx, Vec2::new(8.0, 8.0), 0.0, 10.0, 6.0, Affine::IDENTITY);
}

#[vello_test]
fn inset_box_shadow_offset_up_left(ctx: &mut impl Renderer) {
    inset_box_shadow(ctx, Vec2::new(-8.0, -8.0), 0.0, 10.0, 6.0, Affine::IDENTITY);
}

#[vello_test]
fn inset_box_shadow_offset_horizontal(ctx: &mut impl Renderer) {
    inset_box_shadow(ctx, Vec2::new(12.0, 0.0), 0.0, 10.0, 4.0, Affine::IDENTITY);
}

/// An offset shadow larger than the blur, so the shadow has a hard edge and only covers the
/// top and left of the border box.
#[vello_test]
fn inset_box_shadow_offset_without_blur(ctx: &mut impl Renderer) {
    inset_box_shadow(ctx, Vec2::new(15.0, 15.0), 0.0, 0.0, 0.0, Affine::IDENTITY);
}

/// A spread shrinks the shadow rectangle (and its corner radius), so the shadow also covers the
/// bottom and right of the border box despite the offset.
#[vello_test]
fn inset_box_shadow_with_spread(ctx: &mut impl Renderer) {
    inset_box_shadow(ctx, Vec2::new(4.0, 4.0), 8.0, 10.0, 4.0, Affine::IDENTITY);
}

#[vello_test]
fn inset_box_shadow_offset_with_transform(ctx: &mut impl Renderer) {
    inset_box_shadow(
        ctx,
        Vec2::new(8.0, 8.0),
        0.0,
        10.0,
        6.0,
        Affine::rotate_about(20.0_f64.to_radians(), Point::new(50.0, 50.0)),
    );
}

/// The shape is filled with the current fill rule, so the overlapping region of the two
/// circles is not painted.
#[vello_test]
fn blurred_rounded_rect_in_even_odd(ctx: &mut impl Renderer) {
    let mut path = Circle::new(Point::new(35.0, 50.0), 40.0).to_path(0.1);
    path.extend(Circle::new(Point::new(65.0, 50.0), 40.0).to_path(0.1));

    ctx.set_fill_rule(Fill::EvenOdd);
    rect_in_with(ctx, &path, 10.0, 10.0, false, Affine::IDENTITY);
}

const CPU_SIZE: u16 = 200;

/// The alpha of every pixel `draw` paints in black on a transparent `vello_cpu` canvas, on the
/// calling thread, since filter layers do not run on worker threads.
fn cpu_alpha(render_mode: RenderMode, draw: impl FnOnce(&mut RenderContext)) -> Vec<u8> {
    let settings = RenderSettings {
        num_threads: 0,
        ..Default::default()
    };
    let mut ctx = RenderContext::new_with(CPU_SIZE, CPU_SIZE, settings);
    ctx.set_paint(BLACK);
    draw(&mut ctx);
    ctx.flush();
    let mut pixmap = Pixmap::new(CPU_SIZE, CPU_SIZE);
    ctx.render_with(
        &mut pixmap,
        &mut Resources::new(),
        RasterizerSettings {
            render_mode,
            ..Default::default()
        },
    );
    pixmap.data().iter().map(|pixel| pixel.a).collect()
}

/// See <https://github.com/linebender/vello/issues/2043>.
#[test]
fn blurred_rounded_rect_matches_gaussian_blur_filter() {
    let rect = Rect::new(50.0, 50.0, 150.0, 150.0);
    let std_dev = 8.0;
    let size = usize::from(CPU_SIZE);

    for render_mode in [RenderMode::OptimizeSpeed, RenderMode::OptimizeQuality] {
        let blurred = cpu_alpha(render_mode, |ctx| {
            ctx.fill_blurred_rounded_rect(&rect, 0.0, std_dev);
        });
        let filtered = cpu_alpha(render_mode, |ctx| {
            ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::gaussian_blur(
                std_dev,
                EdgeMode::None,
            )));
            ctx.fill_rect(&rect);
            ctx.pop_layer();
        });
        let difference = |index: usize| blurred[index].abs_diff(filtered[index]);

        // Away from the corners the approximation follows a Gaussian; towards them it departs by a
        // few levels.
        let centre_lines = (0..size)
            .flat_map(|position| [position * size + size / 2, size / 2 * size + position])
            .map(difference)
            .max();
        assert!(
            centre_lines <= Some(2),
            "{render_mode:?}: along the lines through the centre, the blurred rectangle's alpha differs from the Gaussian blur filter's by {centre_lines:?}",
        );

        let (index, largest) = (0..size * size)
            .map(|index| (index, difference(index)))
            .max_by_key(|(_, difference)| *difference)
            .unwrap();
        assert!(
            largest <= 6,
            "{render_mode:?}: the blurred rectangle's alpha differs from the Gaussian blur filter's by {largest} at ({}, {})",
            index % size,
            index / size,
        );
    }
}
