// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::load_image;
use crate::renderer::Renderer;
use smallvec::smallvec;
use vello_common::color::palette::css::{BLUE, LIME, MAGENTA, RED, YELLOW};
use vello_common::color::{AlphaColor, DynamicColor, Srgb};
use vello_common::kurbo::{Affine, Point, Rect};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::{
    BlendMode, ColorStop, ColorStops, Compose, Extend, Gradient, GradientKind, ImageQuality, Mix,
};
use vello_dev_macros::vello_test;

// The outputs have been compared visually with tiny-skia, and except for two cases (where tiny-skia
// is wrong), the overall visual effect looks the same.
fn mix(ctx: &mut impl Renderer, blend_mode: BlendMode) {
    let rect = Rect::new(0.0, 0.0, 80.0, 80.0);

    let gradient = Gradient {
        kind: GradientKind::Linear {
            start: Point::new(0.0, 0.0),
            end: Point::new(80.0, 0.0),
        },
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
        source: ImageSource::Pixmap(load_image!("cowboy")),
        x_extend: Extend::Pad,
        y_extend: Extend::Pad,
        quality: ImageQuality::Low,
    };

    ctx.set_transform(Affine::translate((10.0, 10.0)));
    ctx.set_paint(image);
    ctx.fill_rect(&rect);
    ctx.push_blend_layer(blend_mode);
    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

#[vello_test]
fn mix_normal(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Normal, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
fn mix_multiply(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Multiply, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
fn mix_screen(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Screen, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
fn mix_darken(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Darken, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
fn mix_lighten(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Lighten, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 4)]
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
#[vello_test(cpu_u8_tolerance = 5)]
fn mix_color_burn(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::ColorBurn, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
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

#[vello_test(cpu_u8_tolerance = 1)]
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

#[vello_test(cpu_u8_tolerance = 2)]
fn mix_color(ctx: &mut impl Renderer) {
    mix(ctx, BlendMode::new(Mix::Color, Compose::SrcOver));
}

#[vello_test(cpu_u8_tolerance = 1)]
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
