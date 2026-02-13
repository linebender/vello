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