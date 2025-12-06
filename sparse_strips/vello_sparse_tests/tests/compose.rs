// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::renderer::Renderer;
use vello_common::color::palette::css::{BLUE, GREEN, LIME, PURPLE, RED, YELLOW};
use vello_common::kurbo::{Affine, Circle, Point, Rect, Shape};
use vello_common::peniko::{BlendMode, Color, Compose, Mix};
use vello_dev_macros::vello_test;

fn compose(ctx: &mut impl Renderer, compose: Compose) {
    ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcOver));

    // Draw the destination layer.
    ctx.set_paint(YELLOW.with_alpha(1.0));
    ctx.fill_rect(&Rect::new(10.0, 10.0, 70.0, 70.0));
    // Draw the source layer.
    ctx.push_blend_layer(BlendMode::new(Mix::Normal, compose));
    ctx.set_paint(BLUE.with_alpha(1.0));
    ctx.fill_rect(&Rect::new(30.0, 30.0, 90.0, 90.0));
    // Compose.
    ctx.pop_layer();
    ctx.pop_layer();
}

#[vello_test]
fn compose_src_over(ctx: &mut impl Renderer) {
    compose(ctx, Compose::SrcOver);
}

#[vello_test]
fn compose_xor(ctx: &mut impl Renderer) {
    compose(ctx, Compose::Xor);
}

#[vello_test]
fn compose_clear(ctx: &mut impl Renderer) {
    compose(ctx, Compose::Clear);
}

#[vello_test]
fn compose_copy(ctx: &mut impl Renderer) {
    compose(ctx, Compose::Copy);
}

#[vello_test]
fn compose_dest(ctx: &mut impl Renderer) {
    compose(ctx, Compose::Dest);
}

#[vello_test]
fn compose_dest_over(ctx: &mut impl Renderer) {
    compose(ctx, Compose::DestOver);
}

#[vello_test]
fn compose_src_in(ctx: &mut impl Renderer) {
    compose(ctx, Compose::SrcIn);
}

#[vello_test]
fn compose_src_out(ctx: &mut impl Renderer) {
    compose(ctx, Compose::SrcOut);
}

#[vello_test]
fn compose_dest_in(ctx: &mut impl Renderer) {
    compose(ctx, Compose::DestIn);
}

#[vello_test]
fn compose_dest_out(ctx: &mut impl Renderer) {
    compose(ctx, Compose::DestOut);
}

#[vello_test]
fn compose_src_atop(ctx: &mut impl Renderer) {
    compose(ctx, Compose::SrcAtop);
}

#[vello_test]
fn compose_dest_atop(ctx: &mut impl Renderer) {
    compose(ctx, Compose::DestAtop);
}

#[vello_test]
fn compose_plus(ctx: &mut impl Renderer) {
    compose(ctx, Compose::Plus);
}

#[vello_test(height = 8, width = 100)]
fn composed_layers_nesting(ctx: &mut impl Renderer) {
    ctx.push_blend_layer(BlendMode {
        mix: Mix::Normal,
        compose: Compose::SrcOver,
    });
    {
        ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcOver));
        {
            ctx.set_paint(BLUE);
            ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 4.0));
        }
        ctx.pop_layer();

        ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::DestOut));
        {
            ctx.set_paint(RED);
            ctx.fill_rect(&Rect::new(33.0, 0.0, 66.0, 4.0));
        }
        ctx.pop_layer();
    }
    ctx.pop_layer();
}

#[vello_test(height = 8, width = 100)]
fn repeatedly_compose_to_bottom_layer(ctx: &mut impl Renderer) {
    ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcOver));
    {
        ctx.set_paint(BLUE);
        ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 4.0));
    }
    ctx.pop_layer();

    ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::DestOut));
    {
        ctx.set_paint(RED);
        ctx.fill_rect(&Rect::new(33.0, 0.0, 66.0, 4.0));
    }
    ctx.pop_layer();
}

#[vello_test(width = 100, height = 100, transparent)]
fn complex_composed_layers(ctx: &mut impl Renderer) {
    ctx.push_blend_layer(BlendMode {
        mix: Mix::Normal,
        compose: Compose::SrcOver,
    });

    ctx.set_paint(Color::from_rgb8(240, 240, 240));
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));

    // Base layer circles to shine through
    ctx.set_paint(YELLOW);
    ctx.fill_path(&Circle::new(Point::new(30.0, 30.0), 25.0).to_path(0.1));
    ctx.set_paint(BLUE);
    ctx.fill_path(&Circle::new(Point::new(70.0, 30.0), 25.0).to_path(0.1));
    ctx.set_paint(RED);
    ctx.fill_path(&Circle::new(Point::new(30.0, 70.0), 25.0).to_path(0.1));
    ctx.set_paint(GREEN);
    ctx.fill_path(&Circle::new(Point::new(70.0, 70.0), 25.0).to_path(0.1));

    // Layer 1: Simulate clip to center area using compositing
    ctx.push_layer(None, None, None, None, None);
    {
        // Draw the "clip" shape first
        ctx.set_paint(Color::WHITE);
        ctx.fill_path(&Circle::new(Point::new(50.0, 50.0), 40.0).to_path(0.1));

        // Now use SrcIn to clip content to that shape
        ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcIn));
        {
            // Layer 2: Semi-transparent purple overlay
            ctx.push_layer(None, None, Some(0.7), None, None);
            ctx.set_paint(PURPLE);
            ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
            ctx.pop_layer();

            // Layer 3: XOR blend mode with smaller clip simulation
            ctx.push_layer(None, None, None, None, None);
            {
                // Draw the "clip" rectangle
                ctx.set_paint(Color::WHITE);
                ctx.fill_rect(&Rect::new(25.0, 25.0, 75.0, 75.0));

                // Use SrcIn to clip the XOR content
                ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcIn));
                {
                    ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::Xor));

                    // Draw overlapping circles with XOR
                    ctx.set_paint(RED);
                    ctx.fill_path(&Circle::new(Point::new(40.0, 40.0), 20.0).to_path(0.1));
                    ctx.set_paint(BLUE);
                    ctx.fill_path(&Circle::new(Point::new(60.0, 40.0), 20.0).to_path(0.1));

                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();

            // Layer 4: Nested opacity with Plus blend
            ctx.push_layer(None, None, Some(0.5), None, None);
            ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::Plus));
            {
                ctx.set_paint(YELLOW);
                ctx.fill_path(&Circle::new(Point::new(50.0, 60.0), 15.0).to_path(0.1));

                // Layer 5: "Clip" simulation with SrcIn
                ctx.push_layer(None, None, None, None, None);
                {
                    ctx.set_paint(Color::WHITE);
                    ctx.fill_path(&Circle::new(Point::new(50.0, 50.0), 30.0).to_path(0.1));

                    // Use SrcIn twice (once for the clip, once for the original SrcIn)
                    ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcIn));
                    {
                        ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcIn));
                        {
                            ctx.set_paint(GREEN);
                            ctx.fill_rect(&Rect::new(35.0, 35.0, 65.0, 65.0));
                        }
                        ctx.pop_layer();
                    }
                    ctx.pop_layer();
                }
                ctx.pop_layer();
            }
            ctx.pop_layer();
            ctx.pop_layer();

            // Layer 6: Final overlay with DestOut to create a hole
            ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::DestOut));
            ctx.set_paint(Color::BLACK);
            ctx.fill_path(&Circle::new(Point::new(50.0, 50.0), 10.0).to_path(0.1));
            ctx.pop_layer();
        }
        ctx.pop_layer();
    }
    ctx.pop_layer();

    ctx.pop_layer();
}

#[vello_test(
    width = 100,
    height = 100,
    transparent,
    cpu_u8_tolerance = 2,
    hybrid_tolerance = 2
)]
fn deep_compose(ctx: &mut impl Renderer) {
    const INITIAL_RADIUS: f64 = 48.0;
    const RADIUS_DECREMENT: f64 = 4.5;
    const LAYER_COUNT: usize = 10;
    const CENTER: Point = Point::new(50.0, 50.0);

    const COLORS: [Color; LAYER_COUNT] = [
        Color::from_rgba8(120, 0, 0, 50),
        Color::from_rgba8(120, 60, 0, 50),
        Color::from_rgba8(120, 120, 0, 50),
        Color::from_rgba8(60, 120, 0, 50),
        Color::from_rgba8(0, 120, 0, 50),
        Color::from_rgba8(0, 120, 60, 50),
        Color::from_rgba8(0, 120, 120, 50),
        Color::from_rgba8(0, 60, 120, 50),
        Color::from_rgba8(0, 0, 120, 50),
        Color::from_rgba8(60, 0, 120, 50),
    ];

    // Composition modes are intentionally "additive".
    const COMPOSE_MODES: [Compose; LAYER_COUNT] = [
        Compose::SrcOver,
        Compose::Plus,
        Compose::SrcOver,
        Compose::DestOver,
        Compose::SrcOver,
        Compose::Plus,
        Compose::SrcAtop,
        Compose::Plus,
        Compose::SrcOver,
        Compose::SrcOver,
    ];

    ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcOver));
    ctx.set_paint(Color::BLACK);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));

    let mut radius = INITIAL_RADIUS;

    for i in 0..LAYER_COUNT {
        ctx.push_blend_layer(BlendMode::new(Mix::Normal, COMPOSE_MODES[i]));

        ctx.set_paint(COLORS[i]);
        ctx.fill_path(&Circle::new(CENTER, radius).to_path(0.1));

        radius -= RADIUS_DECREMENT;
    }

    for _ in 0..=LAYER_COUNT {
        ctx.pop_layer();
    }
}

// Ensure that compose and mix work together in the same blend layer.
#[vello_test(width = 160, height = 160)]
fn mix_compose_combined_test_matrix(ctx: &mut impl Renderer) {
    let mix_modes = [
        Mix::Normal,
        Mix::Multiply,
        Mix::Screen,
        Mix::Overlay,
        Mix::Darken,
        Mix::Lighten,
        Mix::ColorDodge,
        Mix::ColorBurn,
        Mix::HardLight,
        Mix::SoftLight,
        Mix::Difference,
        Mix::Exclusion,
        Mix::Hue,
        Mix::Saturation,
        Mix::Color,
        Mix::Luminosity,
    ];

    let compose_modes = [
        Compose::Clear,
        Compose::Copy,
        Compose::Dest,
        Compose::SrcOver,
        Compose::DestOver,
        Compose::SrcIn,
        Compose::DestIn,
        Compose::SrcOut,
        Compose::DestOut,
        Compose::SrcAtop,
        Compose::DestAtop,
        Compose::Xor,
        Compose::Plus,
        Compose::PlusLighter,
    ];

    let cell_size = 10.0;

    ctx.set_paint(Color::from_rgb8(30, 30, 30));
    ctx.fill_rect(&Rect::new(0.0, 0.0, 160.0, 160.0));

    for (row, mix_mode) in mix_modes.iter().enumerate() {
        for (col, compose_mode) in compose_modes.iter().enumerate() {
            let x = (col as f64) * cell_size;
            let y = (row as f64) * cell_size;

            ctx.set_transform(Affine::translate((x, y)));

            ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcOver));

            // Draw magenta rectangle for destination.
            ctx.set_paint(Color::from_rgb8(200, 0, 200).with_alpha(0.7));
            ctx.fill_rect(&Rect::new(0.0, 0.0, cell_size * 0.7, cell_size * 0.7));

            // Push mix + compose blend layer with fill
            ctx.push_blend_layer(BlendMode::new(*mix_mode, *compose_mode));
            ctx.set_paint(Color::from_rgb8(10, 200, 200).with_alpha(0.7)); // Cyan
            ctx.fill_rect(&Rect::new(
                cell_size * 0.3,
                cell_size * 0.3,
                cell_size,
                cell_size,
            ));
            ctx.pop_layer();

            ctx.pop_layer();
        }
    }
}

fn compose_non_isolated(ctx: &mut impl Renderer, compose: Compose) {
    // Just to isolate from the white background.
    ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcOver));

    let rect1 = Rect::new(10.5, 10.5, 70.5, 70.5);
    ctx.set_paint(BLUE.with_alpha(0.5));
    ctx.fill_rect(&rect1);
    ctx.set_blend_mode(BlendMode::new(Mix::Normal, compose));
    let rect2 = Rect::new(30.5, 30.5, 90.5, 90.5);
    ctx.set_paint(LIME.with_alpha(0.5));
    ctx.fill_rect(&rect2);

    ctx.pop_layer();
}

#[vello_test]
fn compose_non_isolated_src_over(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::SrcOver);
}

#[vello_test]
fn compose_non_isolated_xor(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::Xor);
}

#[vello_test]
fn compose_non_isolated_clear(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::Clear);
}

#[vello_test]
fn compose_non_isolated_copy(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::Copy);
}

#[vello_test]
fn compose_non_isolated_dest(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::Dest);
}

#[vello_test]
fn compose_non_isolated_dest_over(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::DestOver);
}

#[vello_test]
fn compose_non_isolated_src_in(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::SrcIn);
}

#[vello_test]
fn compose_non_isolated_src_out(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::SrcOut);
}

#[vello_test]
fn compose_non_isolated_dest_in(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::DestIn);
}

#[vello_test]
fn compose_non_isolated_dest_out(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::DestOut);
}

#[vello_test]
fn compose_non_isolated_src_atop(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::SrcAtop);
}

#[vello_test]
fn compose_non_isolated_dest_atop(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::DestAtop);
}

#[vello_test]
fn compose_non_isolated_plus(ctx: &mut impl Renderer) {
    compose_non_isolated(ctx, Compose::Plus);
}
