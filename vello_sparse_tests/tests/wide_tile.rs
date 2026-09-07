// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Legacy tests for the former wide-tile-based Vello Hybrid architecture.

use crate::renderer::Renderer;
use crate::util::{stops_blue_green_red_yellow, stops_green_blue};
use vello_common::color::palette::css::{BLUE, GREEN, LIME, RED};
use vello_common::filter_effects::{Filter, FilterPrimitive};
use vello_common::kurbo::{BezPath, Point, Rect, Shape};
use vello_common::peniko::{Color, Extend, Fill, Gradient};
use vello_common::tile::Tile;
use vello_cpu::peniko::LinearGradientPosition;
use vello_dev_macros::vello_test;

#[vello_test(height = 8)]
fn wide_tile_clip_single_wide_tile(ctx: &mut impl Renderer) {
    const WIDTH: f64 = 100.0;
    const HEIGHT: f64 = Tile::HEIGHT as f64;
    const OFFSET: f64 = WIDTH / 3.0;

    let colors = [RED, GREEN, BLUE];

    for (i, color) in colors.iter().enumerate() {
        let clip_rect = Rect::new((i as f64) * OFFSET, 0.0, WIDTH, HEIGHT);
        ctx.push_clip_layer(&clip_rect.to_path(0.1));
        ctx.set_paint(*color);
        ctx.fill_rect(&Rect::new(0.0, 0.0, WIDTH, HEIGHT));
    }
    for _ in colors.iter() {
        ctx.pop_layer();
    }
}

// See <https://github.com/linebender/vello/pull/975#issuecomment-2858372366>
#[vello_test(no_ref)]
fn wide_tile_clip_completely_in_out_of_bounds_wide_tile(ctx: &mut impl Renderer) {
    ctx.push_clip_layer(&Rect::new(300.0, 8.0, 350.0, 48.0).to_path(0.1));
    ctx.pop_layer();
}

// If the bbox of a filter layer doesn't start on the top-left wide tile, we will shift
// the image so the top-left wide tile of the bbox starts at (0, 0). This test
// ensures that complex paints are also appropriately shifted. The correct behavior is
// to see the whole gradient, the wrong behavior would be to only see a blue rectangle.
#[vello_test(skip_multithreaded, width = 512, height = 4)]
fn wide_tile_filter_with_complex_paint_and_wide_tile_shift(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::Offset { dx: 0.0, dy: 0.0 });

    let gradient = Gradient {
        kind: LinearGradientPosition {
            start: Point::new(256.0, 0.0),
            end: Point::new(512.0, 0.0),
        }
        .into(),
        stops: stops_blue_green_red_yellow(),
        extend: Extend::Pad,
        ..Default::default()
    };

    ctx.push_filter_layer(filter);
    ctx.set_paint(gradient);
    ctx.fill_rect(&Rect::new(256.0, 0.0, 612.0, 4.0));
    ctx.pop_layer();
}

#[vello_test(width = 600, height = 32)]
fn wide_tile_gradient_on_3_wide_tiles(ctx: &mut impl Renderer) {
    let rect = Rect::new(4.0, 4.0, 596.0, 28.0);

    let gradient = Gradient {
        kind: LinearGradientPosition {
            start: Point::new(0.0, 0.0),
            end: Point::new(600.0, 0.0),
        }
        .into(),
        stops: stops_green_blue(),
        ..Default::default()
    };

    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);
}

#[vello_test(width = 256, height = 4, no_ref)]
// https://github.com/LaurenzV/cpu-sparse-experiments/issues/30
fn wide_tile_shape_at_wide_tile_boundary(ctx: &mut impl Renderer) {
    let mut path = BezPath::new();
    path.move_to((248.0, 0.0));
    path.line_to((257.0, 0.0));
    path.line_to((257.0, 2.0));
    path.line_to((248.0, 2.0));
    path.close_path();

    ctx.set_fill_rule(Fill::EvenOdd);
    ctx.set_paint(LIME);
    ctx.fill_path(&path);
}

#[vello_test]
fn wide_tile_issue_fast_path_strips_in_later_round(ctx: &mut impl Renderer) {
    ctx.push_layer(None, None, None, None, None);
    ctx.push_layer(None, None, None, None, None);
    ctx.push_layer(None, None, None, None, None);
    ctx.set_paint(Color::from_rgba8(0, 0, 255, 255));
    ctx.fill_rect(&Rect::new(10.0, 10.0, 70.0, 70.0));
    ctx.pop_layer();
    ctx.pop_layer();
    ctx.pop_layer();

    ctx.set_paint(Color::from_rgba8(255, 0, 0, 255));
    ctx.fill_rect(&Rect::new(30.0, 30.0, 90.0, 90.0));
}

#[vello_test]
fn wide_tile_issue_coarse_batch_in_later_round(ctx: &mut impl Renderer) {
    ctx.push_layer(None, None, None, None, None);
    ctx.push_layer(None, None, None, None, None);
    ctx.push_layer(None, None, None, None, None);
    ctx.set_paint(Color::from_rgba8(0, 0, 255, 255));
    ctx.fill_rect(&Rect::new(10.0, 10.0, 70.0, 70.0));
    ctx.pop_layer();
    ctx.pop_layer();
    ctx.pop_layer();

    ctx.push_layer(None, None, None, None, None);
    ctx.set_paint(Color::from_rgba8(255, 0, 0, 255));
    ctx.fill_rect(&Rect::new(30.0, 30.0, 90.0, 90.0));
    ctx.pop_layer();
}

#[vello_test]
fn wide_tile_issue_fast_path_strips_and_coarse_batch_in_later_round(ctx: &mut impl Renderer) {
    ctx.push_layer(None, None, None, None, None);
    ctx.push_layer(None, None, None, None, None);
    ctx.push_layer(None, None, None, None, None);
    ctx.set_paint(Color::from_rgba8(0, 0, 255, 255));
    ctx.fill_rect(&Rect::new(25.0, 10.0, 75.0, 60.0));
    ctx.pop_layer();
    ctx.pop_layer();
    ctx.pop_layer();

    ctx.set_paint(Color::from_rgba8(0, 255, 0, 255));
    ctx.fill_rect(&Rect::new(10.0, 40.0, 60.0, 90.0));

    ctx.push_layer(None, None, None, None, None);
    ctx.set_paint(Color::from_rgba8(255, 0, 0, 255));
    ctx.fill_rect(&Rect::new(40.0, 40.0, 90.0, 90.0));
    ctx.pop_layer();
}
