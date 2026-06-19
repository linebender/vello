// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Visual regression tests specifically designed to exercise hybrid scheduler paths.
//!
//! Note that these tests were designed to work with the initial version of the scheduler.
//! It's possible that, if the scheduling algorithm changes, those paths aren't exercised anymore
//! in the way they initially were intended to be. Nevertheless, regardless of the exact
//! implementation, they will still always be useful to test the functionality of deeper
//! layer stacks.

use crate::renderer::Renderer;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Circle, Rect, Shape};
use vello_common::peniko::{BlendMode, Color, Compose, Mix};
use vello_dev_macros::vello_test;

// Some of the comments below refer to a so-called "primer child". This is an
// otherwise-unneeded child layer whose sole purpose is to introduce an earlier node, forcing the
// scheduler to allocate atlas space for the parent before processing the following child. See the
// scheduler documentation for more information.

/// Test the behavior when a spill to a page 1 is forced and that page is later reused.
#[vello_test(cpu_u8_tolerance = 2, hybrid_tolerance = 1)]
fn hybrid_schedule_atlas_page_one_reuse(ctx: &mut impl Renderer) {
    const BOUNDS: Rect = Rect::new(6.0, 6.0, 94.0, 94.0);
    let clip = Rect::new(6.5, 6.5, 93.5, 93.5).to_path(0.1);

    // Depth one shifts the pressure layer below to even parity.
    ctx.push_layer(None, None, None, None, None);
    // This depth-two allocation remains live on even page zero while all three branches are
    // scheduled, forcing their depth-four allocations onto even page one.
    ctx.push_layer(None, None, None, None, None);
    ctx.set_paint(Color::from_rgb8(24, 28, 58));
    ctx.fill_rect(&BOUNDS);

    // The primer child causes the depth-two target to be allocated before any of the depth-four
    // branches request storage.
    ctx.push_layer(None, None, None, None, None); // Depth three primer.
    ctx.set_paint(Color::from_rgb8(55, 38, 86).with_alpha(0.35));
    ctx.fill_rect(&BOUNDS);
    ctx.pop_layer();

    // The first depth-four branch occupies even page one while the depth-two allocation remains
    // live on even page zero.
    ctx.push_layer(None, None, None, None, None);
    ctx.set_paint(Color::from_rgb8(22, 124, 132).with_alpha(0.55));
    ctx.fill_rect(&BOUNDS);
    ctx.push_layer(None, None, None, None, None);
    ctx.set_paint(Color::from_rgb8(255, 103, 122).with_alpha(0.82));
    ctx.fill_path(&Circle::new((50.0, 50.0), 44.0).to_path(0.1));
    ctx.pop_layer();
    ctx.pop_layer();

    // Once the first branch has been released, the second depth-four branch reuses even page one.
    ctx.push_layer(None, None, None, None, None);
    ctx.set_paint(Color::from_rgb8(96, 72, 168).with_alpha(0.45));
    ctx.fill_rect(&BOUNDS);
    ctx.push_layer(Some(&clip), None, None, None, None);
    ctx.set_paint(Color::from_rgb8(112, 235, 184).with_alpha(0.8));
    ctx.fill_path(&Circle::new((50.0, 50.0), 44.0).to_path(0.1));
    ctx.pop_layer();
    ctx.pop_layer();

    // Once the second branch has been released, the third depth-four branch reuses the same
    // cleared region on even page one again. This time with a blend mode, for good measure.
    ctx.push_layer(None, None, None, None, None);
    ctx.set_paint(Color::from_rgb8(238, 156, 72).with_alpha(0.38));
    ctx.fill_rect(&BOUNDS);
    ctx.push_layer(
        Some(&clip),
        Some(BlendMode::new(Mix::Multiply, Compose::SrcOver)),
        None,
        None,
        None,
    );
    ctx.set_paint(Color::from_rgb8(255, 226, 94).with_alpha(0.86));
    ctx.fill_path(&Circle::new((50.0, 50.0), 44.0).to_path(0.1));
    ctx.pop_layer();
    ctx.pop_layer();

    ctx.pop_layer();
    ctx.pop_layer();
}

/// Test the behavior when forcing even deeper spills (in this case to page index 2).
#[vello_test(cpu_u8_tolerance = 1, hybrid_tolerance = 1)]
fn hybrid_schedule_atlas_page_index_two(ctx: &mut impl Renderer) {
    const BOUNDS: Rect = Rect::new(6.0, 6.0, 94.0, 94.0);

    ctx.push_layer(None, None, None, None, None); // Depth one.
    ctx.push_layer(None, None, None, None, None); // Depth two: even page zero.
    ctx.set_paint(Color::from_rgb8(64, 48, 132).with_alpha(0.75));
    ctx.fill_rect(&BOUNDS);

    // The primer child causes the depth-two target to be allocated on even page zero before deeper
    // layers request storage.
    ctx.push_layer(None, None, None, None, None); // Depth three primer.
    ctx.set_paint(Color::from_rgb8(213, 70, 126).with_alpha(0.3));
    ctx.fill_rect(&BOUNDS);
    ctx.pop_layer();

    ctx.push_layer(None, None, None, None, None); // Depth three.
    ctx.push_layer(None, None, None, None, None); // Depth four: even page one.
    ctx.set_paint(Color::from_rgb8(36, 190, 166).with_alpha(0.62));
    ctx.fill_rect(&BOUNDS);

    // The primer child causes the depth-four target to be allocated on even page one before the
    // depth-six child requests storage.
    ctx.push_layer(None, None, None, None, None); // Depth five primer.
    ctx.set_paint(Color::from_rgb8(255, 194, 86).with_alpha(0.32));
    ctx.fill_rect(&BOUNDS);
    ctx.pop_layer();

    ctx.push_layer(None, None, None, None, None); // Depth five.
    ctx.push_layer(None, None, None, None, None); // Depth six: even page two.
    ctx.set_paint(Color::from_rgb8(255, 103, 122).with_alpha(0.86));
    ctx.fill_path(&Circle::new((50.0, 50.0), 44.0).to_path(0.1));
    ctx.pop_layer();
    ctx.pop_layer();

    ctx.pop_layer();
    ctx.pop_layer();

    ctx.pop_layer();
    ctx.pop_layer();
}

/// Test the behavior when a whole filter layer needs to be processed on page 1.
#[vello_test(skip_multithreaded, cpu_u8_tolerance = 3, hybrid_tolerance = 1)]
fn hybrid_schedule_filter_atlas_page_one(ctx: &mut impl Renderer) {
    const BOUNDS: Rect = Rect::new(6.0, 6.0, 94.0, 94.0);
    let filter = Filter::from_primitive(FilterPrimitive::Offset { dx: 0.0, dy: 0.0 });

    ctx.push_layer(None, None, None, None, None); // Depth one.
    ctx.push_layer(None, None, None, None, None); // Depth two: even page zero.
    ctx.set_paint(Color::from_rgb8(54, 72, 154).with_alpha(0.62));
    ctx.fill_rect(&BOUNDS);

    // The primer child causes the depth-two target to be allocated on even page zero before the
    // next child requests storage.
    ctx.push_layer(None, None, None, None, None); // Depth three primer.
    ctx.set_paint(Color::from_rgb8(172, 104, 214).with_alpha(0.24));
    ctx.fill_rect(&BOUNDS);
    ctx.pop_layer();

    ctx.push_layer(None, None, None, None, None); // Depth three: odd page zero.
    ctx.set_paint(Color::from_rgb8(32, 168, 148).with_alpha(0.58));
    ctx.fill_rect(&BOUNDS);

    // The primer child causes the depth-three target to be allocated on odd page zero before the
    // next child requests storage.
    ctx.push_layer(None, None, None, None, None); // Depth four primer.
    ctx.set_paint(Color::from_rgb8(255, 180, 92).with_alpha(0.26));
    ctx.fill_rect(&BOUNDS);
    ctx.pop_layer();

    // The filter source cannot fit beside the live even allocation, and its padded temporary
    // cannot fit beside the live odd allocation.
    ctx.push_filter_layer(filter); // Depth four: even page one, odd temporary page one.
    ctx.set_paint(Color::from_rgb8(112, 126, 232).with_alpha(0.9));
    ctx.fill_path(&Circle::new((50.0, 50.0), 44.0).to_path(0.1));
    ctx.pop_layer();

    ctx.pop_layer();
    ctx.pop_layer();
    ctx.pop_layer();
}

/// Test the behavior when multiple filter layers are batched together in a single layer texture.
#[vello_test(skip_multithreaded, hybrid_tolerance = 2)]
fn hybrid_schedule_filter_batch_page_zero(ctx: &mut impl Renderer) {
    let blur_two = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 2.0,
        edge_mode: EdgeMode::None,
    });
    let blur_four = Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 4.0,
        edge_mode: EdgeMode::None,
    });
    let drop_shadow = Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 4.0,
        dy: 4.0,
        std_deviation: 4.0,
        color: Color::from_rgb8(40, 28, 72).with_alpha(0.8),
        edge_mode: EdgeMode::None,
    });
    let offset_zero = Filter::from_primitive(FilterPrimitive::Offset { dx: 0.0, dy: 0.0 });
    let texture_sizing_filter = Filter::from_primitive(FilterPrimitive::Offset {
        dx: 180.0,
        dy: 180.0,
    });

    // Make sure that the minimum texture size is expanded enough by drawing a large, nearly
    // invisible layer.
    ctx.push_filter_layer(texture_sizing_filter);
    ctx.set_paint(Color::WHITE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 20.0, 20.0));
    ctx.pop_layer();

    for (filter, bounds, color) in [
        (
            blur_two,
            Rect::new(12.5, 12.5, 42.5, 42.5),
            Color::from_rgb8(255, 105, 120),
        ),
        (
            blur_four,
            Rect::new(57.5, 12.5, 87.5, 42.5),
            Color::from_rgb8(55, 190, 178),
        ),
        (
            drop_shadow,
            Rect::new(12.5, 57.5, 42.5, 87.5),
            Color::from_rgb8(255, 199, 82),
        ),
        (
            offset_zero,
            Rect::new(57.5, 57.5, 87.5, 87.5),
            Color::from_rgb8(118, 112, 235),
        ),
    ] {
        ctx.push_filter_layer(filter);
        ctx.set_paint(color);
        ctx.fill_rect(&bounds);
        ctx.pop_layer();
    }
}
