// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scene demonstrating blurred rounded rectangle rendering.

use vello_common::kurbo::{Affine, Rect};
use vello_common::peniko::color::palette;
use vello_common::peniko::color::{AlphaColor, Srgb};

use crate::{ExampleScene, RenderingContext};

/// Scene state for blurred rounded rectangle examples.
#[derive(Debug)]
pub struct BlurredRoundedRectScene;

impl BlurredRoundedRectScene {
    /// Create a new `BlurredRoundedRectScene`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BlurredRoundedRectScene {
    fn default() -> Self {
        Self::new()
    }
}

impl ExampleScene for BlurredRoundedRectScene {
    fn render<T: RenderingContext>(
        &mut self,
        target: &mut T,
        _resources: &mut T::Resources,
        root_transform: Affine,
    ) {
        render(target, root_transform);
    }
}

/// Draw blurred rounded rectangles adapted from Vello Classic's test scene.
pub fn render(ctx: &mut impl RenderingContext, root_transform: Affine) {
    let bounds = Rect::new(0.0, 0.0, f64::from(ctx.width()), f64::from(ctx.height()));
    ctx.set_transform(Affine::IDENTITY);
    ctx.set_paint(palette::css::WHITE);
    ctx.fill_rect(&bounds);

    let scene_transform = root_transform
        * Affine::scale_non_uniform(ctx.width() as f64 / 1200.0, ctx.height() as f64 / 1200.0);
    let rect = Rect::from_center_size((0.0, 0.0), (300.0, 240.0));

    draw_blurred_rect(
        ctx,
        scene_transform * Affine::translate((300.0, 300.0)),
        rect,
        palette::css::BLUE,
        50.0,
        45.0,
    );

    draw_blurred_rect(
        ctx,
        scene_transform
            * Affine::translate((900.0, 300.0))
            * Affine::skew(20_f64.to_radians().tan(), 0.0),
        rect,
        palette::css::BLACK,
        50.0,
        45.0,
    );

    draw_blurred_rect(
        ctx,
        scene_transform,
        Rect::new(100.0, 800.0, 400.0, 1100.0),
        palette::css::BLACK,
        150.0,
        45.0,
    );

    draw_blurred_rect(
        ctx,
        scene_transform,
        Rect::new(600.0, 800.0, 900.0, 900.0),
        palette::css::BLACK,
        150.0,
        45.0,
    );

    draw_blurred_rect(
        ctx,
        scene_transform * Affine::translate((600.0, 600.0)) * Affine::scale_non_uniform(2.2, 0.9),
        rect,
        palette::css::BLACK,
        50.0,
        30.0,
    );
}

fn draw_blurred_rect(
    ctx: &mut impl RenderingContext,
    transform: Affine,
    rect: Rect,
    color: AlphaColor<Srgb>,
    radius: f32,
    std_dev: f32,
) {
    ctx.set_transform(transform);
    ctx.set_paint(color);
    ctx.fill_blurred_rounded_rect(&rect, radius, std_dev);
}
