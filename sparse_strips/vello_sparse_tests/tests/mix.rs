// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::image::load_image;
use smallvec::smallvec;
use vello_common::color::palette::css::{BLUE, LIME, MAGENTA, RED, YELLOW};
use vello_common::color::{AlphaColor, DynamicColor, Srgb};
use vello_common::kurbo::{Affine, Point, Rect};
use vello_common::paint::{Gradient, Image};
use vello_common::peniko::{BlendMode, Compose, Extend, Mix};
use vello_common::peniko::{ColorStop, ColorStops, GradientKind, ImageQuality};
use vello_cpu::RenderContext;
use vello_macros::v_test;

// The outputs have been compared visually with tiny-skia, and except for two cases (where tiny-skia
// is wrong), the overall visual effect looks the same.
fn mix(ctx: &mut RenderContext, name: &str, blend_mode: BlendMode) {
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
        pixmap: load_image("cowboy"),
        x_extend: Extend::Pad,
        y_extend: Extend::Pad,
        quality: ImageQuality::Low,
        transform: Affine::IDENTITY,
    };

    ctx.set_transform(Affine::translate((10.0, 10.0)));
    ctx.set_paint(image);
    ctx.fill_rect(&rect);
    ctx.push_blend_layer(blend_mode);
    ctx.set_paint(gradient);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

#[v_test]
fn mix_normal(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_normal",
        BlendMode::new(Mix::Normal, Compose::SrcOver),
    );
}

#[v_test]
fn mix_multiply(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_multiply",
        BlendMode::new(Mix::Multiply, Compose::SrcOver),
    );
}

#[v_test]
fn mix_screen(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_screen",
        BlendMode::new(Mix::Screen, Compose::SrcOver),
    );
}

#[v_test]
fn mix_darken(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_darken",
        BlendMode::new(Mix::Darken, Compose::SrcOver),
    );
}

#[v_test]
fn mix_lighten(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_lighten",
        BlendMode::new(Mix::Lighten, Compose::SrcOver),
    );
}

#[v_test]
fn mix_color_dodge(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_color_dodge",
        BlendMode::new(Mix::ColorDodge, Compose::SrcOver),
    );
}

#[v_test]
fn mix_color_burn(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_color_burn",
        BlendMode::new(Mix::ColorBurn, Compose::SrcOver),
    );
}

#[v_test]
fn mix_hard_light(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_hard_light",
        BlendMode::new(Mix::HardLight, Compose::SrcOver),
    );
}

#[v_test]
fn mix_soft_light(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_soft_light",
        BlendMode::new(Mix::SoftLight, Compose::SrcOver),
    );
}

#[v_test]
fn mix_difference(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_difference",
        BlendMode::new(Mix::Difference, Compose::SrcOver),
    );
}

#[v_test]
fn mix_exclusion(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_exclusion",
        BlendMode::new(Mix::Exclusion, Compose::SrcOver),
    );
}

#[v_test]
fn mix_overlay(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_overlay",
        BlendMode::new(Mix::Overlay, Compose::SrcOver),
    );
}

#[v_test]
fn mix_hue(ctx: &mut RenderContext) {
    mix(ctx, "mix_hue", BlendMode::new(Mix::Hue, Compose::SrcOver));
}

#[v_test]
fn mix_saturation(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_saturation",
        BlendMode::new(Mix::Saturation, Compose::SrcOver),
    );
}

#[v_test]
fn mix_color(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_color",
        BlendMode::new(Mix::Color, Compose::SrcOver),
    );
}

#[v_test]
fn mix_luminosity(ctx: &mut RenderContext) {
    mix(
        ctx,
        "mix_luminosity",
        BlendMode::new(Mix::Luminosity, Compose::SrcOver),
    );
}

#[v_test(transparent)]
fn mix_with_transparent_bg(ctx: &mut RenderContext) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(0, 0, 128, 128));
    ctx.fill_rect(&rect);
    ctx.push_blend_layer(BlendMode::new(Mix::ColorDodge, Compose::SrcOver));
    ctx.set_paint(YELLOW);
    ctx.fill_rect(&rect);
    ctx.pop_layer();
}

fn mix_solid(ctx: &mut RenderContext, mix: Mix) {
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
#[v_test]
fn mix_color_with_solid(ctx: &mut RenderContext) {
    mix_solid(ctx, Mix::Color);
}

// Color of rectangle should be (213, 52, 0).
#[v_test]
fn mix_saturation_with_solid(ctx: &mut RenderContext) {
    mix_solid(ctx, Mix::Saturation);
}
