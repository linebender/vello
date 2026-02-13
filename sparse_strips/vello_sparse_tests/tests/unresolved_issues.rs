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

// TODO: See https://github.com/linebender/vello/issues/1421, currently causes a crash.
#[vello_test]
fn unresolved_issues_filter_issue_1421(ctx: &mut impl Renderer) {
    let filter_flood = Filter::from_primitive(FilterPrimitive::Flood { color: TOMATO });
    let star_path = circular_star(Point::new(50.0, 50.0), 5, 20.0, 40.0);

    ctx.push_layer(Some(&star_path), None, None, None, Some(filter_flood));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_path(&star_path);
    ctx.pop_layer();
}

#[vello_test(width = 256, height = 100)]
fn unresolved_issues_filter_gaussian_blur_edge_mode_none(ctx: &mut impl Renderer) {
    // TODO: The bottom part looks wrong compared to the other edges.
    crate::filter::blur_with_edge_mode(ctx, EdgeMode::None);
}