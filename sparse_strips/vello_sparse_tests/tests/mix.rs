// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::load_image;
use crate::renderer::Renderer;
use smallvec::smallvec;
use vello_common::color::palette::css::{BLUE, LIME, MAGENTA, ORANGE, RED, YELLOW};
use vello_common::color::{AlphaColor, DynamicColor, Srgb};
use vello_common::kurbo::{Affine, Point, Rect};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::{
    BlendMode, Color, ColorStop, ColorStops, Compose, Extend, Gradient, ImageQuality, Mix,
};
use vello_cpu::peniko::{ImageSampler, LinearGradientPosition};
use vello_dev_macros::vello_test;

fn cowboy_img(ctx: &mut impl Renderer) -> ImageSource {
    ctx.get_image_source(load_image!("cowboy"))
}

// The outputs have been compared visually with tiny-skia, and except for two cases (where tiny-skia
// is wrong), the overall visual effect looks the same.
fn mix(ctx: &mut impl Renderer, blend_mode: BlendMode) {
    let rect = Rect::new(0.0, 0.0, 80.0, 80.0);

    let gradient = Gradient {
        kind: LinearGradientPosition {
            start: Point::new(0.0, 0.0),
            end: Point::new(80.0, 0.0),
        }
        .into(),
        stops: ColorStops(smallvec![
            ColorStop {
                offset: 0.0,
                color: DynamicColor::from_alpha_color(BLUE.with_alpha(0.86)),
            },
            ColorStop {
                offset: 0.25,
                color: DynamicColor::from_alpha_color(MAGENTA.with_alpha(0.86)),
            },
            ColorStop {
                offset: 0.5,
                color: DynamicColor::from_alpha_color(RED.with_alpha(0.86)),
            },
            ColorStop {
                offset: 0.75,
                color: DynamicColor::from_alpha_color(YELLOW.with_alpha(0.86)),
            },
            ColorStop {
                offset: 1.0,
                color: DynamicColor::from_alpha_color(LIME.with_alpha(0.86)),
            },
        ]),
        ..Default::default()
    };

    let image = Image {
        image: cowboy_img(ctx),
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality: ImageQuality::Low,
            alpha: 1.0,
        },
    };

    ctx.set_transform(Affine::translate((10.0, 10.0)));
    ctx.set_paint(image);
    ctx.fill_rect(&rect);
    ctx.push_blend_layer(blend_mode);
    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

#[vello_test(hybrid_tolerance = 1)]
fn mix_normal(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Normal, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1, hybrid_tolerance = 1)]
fn mix_multiply(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Multiply, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
fn mix_screen(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Screen, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1, hybrid_tolerance = 1)]
fn mix_darken(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Darken, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1, hybrid_tolerance = 1)]
fn mix_lighten(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Lighten, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 4, hybrid_tolerance = 5)]
fn mix_color_dodge(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::ColorDodge, Compose::SrcOver));
}

// Unfortunately this one just needs such a high tolerance, but it was manually verified
// that it's due to impreciseness and not a bug.
// At some point, we have the following constellation:
//  f32: source: [1.0, 0.125, 0.0, 0.86], background: [0.99215686, 0.8784314, 0.1882353, 1.0]
//  u8: source: [255, 31, 0, 219], background: [253, 224, 48, 255]
// After plugging into the formula, we get:
//  f32:  1.0 - ((1.0 - 0.8784314) / 0.125) = 0.027451038 (RGB value of around 7)
//  u8/u16:  255 - (((255 - 224) * 255) / 31) = 0
// And therefore a very large difference for that one component.
#[vello_test(cpu_u8_tolerance = 5, hybrid_tolerance = 1)]
fn mix_color_burn(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::ColorBurn, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1, hybrid_tolerance = 1)]
fn mix_hard_light(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::HardLight, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
fn mix_soft_light(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::SoftLight, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
fn mix_difference(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Difference, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
fn mix_exclusion(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Exclusion, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1, hybrid_tolerance = 1)]
fn mix_overlay(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Overlay, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
fn mix_hue(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Hue, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
fn mix_saturation(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Saturation, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 2, hybrid_tolerance = 1)]
fn mix_color(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Color, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1, hybrid_tolerance = 1)]
fn mix_luminosity(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Luminosity, Compose::SrcOver));
}

#[vello_test(transparent)]
fn mix_with_transparent_bg(ctx: &mut impl Renderer) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(0, 0, 128, 128));
    ctx.fill_rect(&rect);
    ctx.push_blend_layer(BlendMode::new(Mix::ColorDodge, Compose::SrcOver));
    ctx.set_paint(YELLOW);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

fn mix_solid(ctx: &mut impl Renderer, mix: Mix) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(122, 85, 73, 255));
    ctx.fill_rect(&rect);

    ctx.push_blend_layer(BlendMode::new(mix, Compose::SrcOver));
    ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(255, 255, 0, 255));
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

// The below 2 test cases yield different results than in tiny-skia, but I checked by converting
// the test case into SVG and viewing it in Chrome, and our version should be right.
// Color of rectangle should be (106, 106, 0).
#[vello_test]
fn mix_color_with_solid(ctx: &mut impl Renderer) {
    mix_solid(ctx, Mix::Color);
}

// Color of rectangle should be (213, 52, 0).
#[vello_test]
fn mix_saturation_with_solid(ctx: &mut impl Renderer) {
    mix_solid(ctx, Mix::Saturation);
}

// Currently `vello_hybrid` does not support gradients. This test ensure mix is tested in
// `vello_hybrid` by using solid colors.
#[vello_test(width = 80, height = 160)]
fn mix_modes_non_gradient_test_matrix(ctx: &mut impl Renderer) {
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

    // Base colors for destination.
    let base_colors = [
        RED,
        Color::from_rgb8(10, 230, 10),
        BLUE,
        YELLOW,
        MAGENTA,
        Color::from_rgb8(10, 230, 230),
        Color::from_rgb8(128, 128, 128),
        Color::from_rgb8(64, 64, 64),
    ];

    let cell_size = 10.0;

    ctx.set_paint(Color::from_rgb8(30, 30, 30));
    ctx.fill_rect(&Rect::new(0.0, 0.0, 80.0, 160.0));

    for (row, mix_mode) in mix_modes.iter().enumerate() {
        for (col, base_color) in base_colors.iter().enumerate() {
            let x = (col as f64) * (cell_size);
            let y = (row as f64) * (cell_size);

            ctx.set_transform(Affine::translate((x, y)));

            // Draw base rectangle (destination)
            ctx.set_paint(*base_color);
            ctx.fill_rect(&Rect::new(0.0, 0.0, cell_size, cell_size));

            // Apply blend mode and draw overlay (source)
            ctx.push_blend_layer(BlendMode::new(*mix_mode, Compose::SrcOver));

            ctx.set_paint(ORANGE.with_alpha(0.7));
            ctx.fill_rect(&Rect::new(0.0, 0.0, cell_size * 0.7, cell_size * 0.7));

            ctx.set_paint(Color::WHITE.with_alpha(0.5));
            ctx.fill_rect(&Rect::new(
                cell_size * 0.3,
                cell_size * 0.3,
                cell_size,
                cell_size,
            ));

            ctx.pop_layer();
        }
    }
}

fn mix_non_isolated(ctx: &mut impl Renderer, mix: Mix) {
    // Just to isolate from the white background.
    ctx.push_blend_layer(BlendMode::new(Mix::Normal, Compose::SrcOver));

    let rect1 = Rect::new(10.5, 10.5, 70.5, 70.5);
    ctx.set_paint(BLUE.with_alpha(0.5));
    ctx.fill_rect(&rect1);
    ctx.set_blend_mode(BlendMode::new(mix, Compose::SrcOver));
    let rect2 = Rect::new(30.5, 30.5, 90.5, 90.5);
    ctx.set_paint(LIME.with_alpha(0.5));
    ctx.fill_rect(&rect2);

    ctx.pop_layer();
}

#[vello_test]
fn mix_non_isolated_difference(ctx: &mut impl Renderer) {
    mix_non_isolated(ctx, Mix::Difference);
}

#[vello_test]
fn mix_non_isolated_soft_light(ctx: &mut impl Renderer) {
    mix_non_isolated(ctx, Mix::SoftLight);
}

#[vello_test]
fn mix_non_isolated_color_dodge(ctx: &mut impl Renderer) {
    mix_non_isolated(ctx, Mix::ColorDodge);
}
