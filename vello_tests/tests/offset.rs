// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for rendering a scene at an offset into the target.

use crate::renderer::Renderer;
use crate::util::{crossed_line_star, stops_green_blue};
use vello_common::color::palette::css::{BLUE, LIME, REBECCA_PURPLE, RED, ROYAL_BLUE, TOMATO};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Circle, Point, Rect, Shape};
use vello_common::peniko::{BlendMode, Compose, Gradient, LinearGradientPosition, Mix};
use vello_dev_macros::vello_test;
use vello_gpu::{ClearSettings, RectU16, TargetInit};

const OFFSET: (i32, i32) = (-11, -5);
const CLEAR_RECTS: &[RectU16] = &[
    RectU16::new(6, 6, 32, 32),
    RectU16::new(56, 10, 90, 44),
    RectU16::new(22, 58, 66, 92),
];

fn fill_circle(ctx: &mut impl Renderer) {
    ctx.set_paint(LIME);
    ctx.fill_path(&Circle::new((50.0, 50.0), 45.0).to_path(0.1));
}

fn fill_target(ctx: &mut impl Renderer, color: AlphaColor<Srgb>) {
    ctx.set_paint(color);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.flush();
    ctx.render();
    ctx.reset();
}

#[vello_test]
fn offset_overflowing_shape_stays_clipped(ctx: &mut impl Renderer) {
    ctx.set_offset((50, 50));
    ctx.set_paint(LIME);
    ctx.fill_path(&Circle::new((0.0, 0.0), 45.0).to_path(0.1));
}

#[vello_test]
fn offset_gradient(ctx: &mut impl Renderer) {
    ctx.set_offset((-49, -51));
    ctx.set_paint(Gradient {
        kind: LinearGradientPosition {
            start: Point::new(20.0, 20.0),
            end: Point::new(80.0, 20.0),
        }
        .into(),
        stops: stops_green_blue(),
        ..Default::default()
    });
    ctx.fill_rect(&Rect::new(20.0, 20.0, 80.0, 80.0));
}

fn fill_clipped_star(ctx: &mut impl Renderer, offset: (i32, i32)) {
    ctx.set_offset(offset);
    ctx.push_clip_layer(&crossed_line_star());
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.pop_layer();
}

#[vello_test]
fn offset_up_left(ctx: &mut impl Renderer) {
    fill_clipped_star(ctx, (-49, -51));
}

#[vello_test]
fn offset_up_right(ctx: &mut impl Renderer) {
    fill_clipped_star(ctx, (49, -51));
}

#[vello_test]
fn offset_down_left(ctx: &mut impl Renderer) {
    fill_clipped_star(ctx, (-49, 51));
}

#[vello_test]
fn offset_down_right(ctx: &mut impl Renderer) {
    fill_clipped_star(ctx, (49, 51));
}

#[vello_test(skip_multithreaded, gpu_tolerance = 2)]
fn offset_filter_drop_shadow(ctx: &mut impl Renderer) {
    ctx.set_offset(OFFSET);
    ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: -20.0,
        dy: -20.0,
        std_deviation: 3.0,
        color: TOMATO,
        edge_mode: EdgeMode::None,
    }));
    ctx.set_paint(ROYAL_BLUE);
    ctx.fill_rect(&Rect::new(20.0, 20.0, 60.0, 60.0));
    ctx.pop_layer();
}

#[vello_test(transparent, gpu_tolerance = 2)]
fn offset_clear_viewport(ctx: &mut impl Renderer) {
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Viewport {
        color: AlphaColor::from_rgba8(18, 52, 86, 120),
    }));
    ctx.set_offset(OFFSET);
    fill_circle(ctx);
}

#[vello_test(transparent)]
fn offset_outside_target(ctx: &mut impl Renderer) {
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Viewport {
        color: AlphaColor::from_rgba8(18, 52, 86, 120),
    }));
    ctx.set_offset((100, 0));
    fill_circle(ctx);
}

#[vello_test(skip_webgl)]
fn offset_clear_rects(ctx: &mut impl Renderer) {
    fill_target(ctx, BLUE);
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Rects {
        color: AlphaColor::TRANSPARENT,
        rects: CLEAR_RECTS,
    }));
    ctx.set_offset(OFFSET);
}

#[vello_test(skip_webgl)]
fn offset_src_over_isolates_root_blend(ctx: &mut impl Renderer) {
    fill_target(ctx, RED);
    ctx.set_target_init(TargetInit::SrcOver);
    ctx.set_offset(OFFSET);

    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    let hole = Circle::new((50.0, 50.0), 25.0).to_path(0.1);
    ctx.push_layer(
        Some(&hole),
        Some(BlendMode::new(Mix::Normal, Compose::Clear)),
        None,
        None,
        None,
    );
    ctx.fill_path(&hole);
    ctx.pop_layer();
}
