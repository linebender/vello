// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integration tests for layer-based filter rendering.

use vello_common::filter_effects::{Filter, FilterPrimitive};
use vello_cpu::color::palette::css::{BLUE, RED};
use vello_cpu::kurbo::Rect;
use vello_cpu::{Pixmap, RenderContext};

#[test]
fn test_layer_with_identity_filter() {
    // Create a simple scene with a filter
    // This validates the layer allocation and filter infrastructure

    let width = 100;
    let height = 100;
    let mut ctx = RenderContext::new(width, height);

    // Draw a red rectangle
    ctx.set_paint(RED);
    ctx.fill_rect(&Rect::from_points((10., 10.), (50., 50.)));

    // Push a layer with an identity filter (no-op)
    let filter = Filter::from_primitive(FilterPrimitive::Offset { dx: 10.0, dy: 10.0 });
    ctx.push_layer(None, None, None, None, Some(filter));

    // Draw a blue rectangle on the filtered layer
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::from_points((30., 30.), (70., 70.)));

    // Pop the layer
    ctx.pop_layer();

    // Render
    let mut target = Pixmap::new(width, height);
    ctx.flush();
    ctx.render_to_pixmap(&mut target);

    // Basic validation: Check that we have non-transparent pixels
    let has_red = target
        .data()
        .iter()
        .any(|p| p.r > 0 && p.g == 0 && p.b == 0);
    let has_blue = target
        .data()
        .iter()
        .any(|p| p.r == 0 && p.g == 0 && p.b > 0);

    assert!(has_red, "Expected to find red pixels");
    assert!(has_blue, "Expected to find blue pixels");
}
