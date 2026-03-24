// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! All tests here should work but currently don't. The module is not included in
//! `mod.rs`, so the tests will never be run.

use vello_api::peniko::color::palette::css::{REBECCA_PURPLE, TOMATO};
use vello_api::peniko::kurbo::Point;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_dev_macros::vello_test;
use crate::renderer::Renderer;
use crate::util::circular_star;

#[vello_test(width = 256, height = 100)]
fn unresolved_issues_filter_gaussian_blur_edge_mode_none(ctx: &mut impl Renderer) {
    // TODO: The bottom part looks wrong compared to the other edges.
    crate::filter::blur_with_edge_mode(ctx, EdgeMode::None);
}

// This one seems to look wrong in both, vello_cpu and vello_hybrid.
#[vello_test(skip_multithreaded)]
fn filter_with_inner_clip(ctx: &mut impl Renderer) {
    let filter = Filter::from_primitive(FilterPrimitive::Offset { dx: 10.0, dy: 10.0 });
    let clip = Rect::new(20.0, 20.0, 70.0, 70.0).to_path(0.1);

    ctx.push_filter_layer(filter);
    ctx.push_clip_layer(&clip);
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
    ctx.pop_layer();
    ctx.pop_layer();
}
