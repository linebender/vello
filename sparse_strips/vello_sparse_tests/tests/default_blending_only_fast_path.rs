// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for the interleaved fast-path / scheduled-strip rendering when
//! `SceneConstraints::default_blending_only()` is active.

use crate::renderer::Renderer;
use vello_common::color::palette::css::{BLUE, GREEN, REBECCA_PURPLE, RED};
use vello_common::kurbo::{Circle, Rect, Shape};
use vello_dev_macros::vello_test;

#[vello_test]
fn default_blending_only_fast_before_clip(ctx: &mut impl Renderer) {
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));

    let clip = Circle::new((50.0, 50.0), 30.0).to_path(0.1);
    ctx.push_clip_layer(&clip);
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.pop_layer();
}

#[vello_test]
fn default_blending_only_fast_before_and_after_clip(ctx: &mut impl Renderer) {
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 40.0));

    let clip = Circle::new((50.0, 50.0), 25.0).to_path(0.1);
    ctx.push_clip_layer(&clip);
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.pop_layer();

    ctx.set_paint(GREEN);
    ctx.fill_rect(&Rect::new(10.0, 60.0, 90.0, 90.0));
}

#[vello_test]
fn default_blending_only_fast_around_opacity(ctx: &mut impl Renderer) {
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 50.0));

    ctx.push_opacity_layer(0.5);
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(20.0, 20.0, 80.0, 80.0));
    ctx.pop_layer();

    ctx.set_paint(GREEN);
    ctx.fill_rect(&Rect::new(10.0, 50.0, 90.0, 90.0));
}

#[vello_test]
fn default_blending_only_nested_layers(ctx: &mut impl Renderer) {
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(5.0, 5.0, 95.0, 95.0));

    let clip = Rect::new(20.0, 20.0, 80.0, 80.0).to_path(0.1);
    ctx.push_clip_layer(&clip);
    ctx.push_opacity_layer(0.5);
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.pop_layer();
    ctx.pop_layer();

    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(30.0, 30.0, 70.0, 70.0));
}

#[vello_test(transparent)]
fn default_blending_only_empty_fast_batch(ctx: &mut impl Renderer) {
    let clip = Circle::new((50.0, 50.0), 40.0).to_path(0.1);
    ctx.push_clip_layer(&clip);
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.pop_layer();
}

#[vello_test]
fn default_blending_only_multiple_segments(ctx: &mut impl Renderer) {
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 30.0));

    let clip = Circle::new((50.0, 50.0), 20.0).to_path(0.1);
    ctx.push_clip_layer(&clip);
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.pop_layer();

    ctx.set_paint(GREEN);
    ctx.fill_rect(&Rect::new(0.0, 35.0, 100.0, 65.0));

    ctx.push_opacity_layer(0.5);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(10.0, 10.0, 90.0, 90.0));
    ctx.pop_layer();

    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(0.0, 70.0, 100.0, 100.0));
}
