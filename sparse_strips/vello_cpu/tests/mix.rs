// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use smallvec::smallvec;
use vello_common::peniko::{ColorStop, ColorStops, GradientKind, ImageQuality};
use vello_common::paint::{Gradient, Image};
use vello_common::color::{AlphaColor, DynamicColor, Srgb};
use vello_common::color::palette::css::{BLUE, LIME, MAGENTA, RED, YELLOW};
use vello_common::kurbo::{Affine, Point, Rect};
use vello_common::peniko::{BlendMode, Compose, Mix, Extend};
use crate::image::load_image;
use crate::util::{check_ref, get_ctx};

// The outputs have been compared visually with tiny-skia, and except for two cases (where tiny-skia
// is wrong), the overall visual effect looks the same.
fn mix(name: &str, blend_mode: BlendMode) {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(0.0, 0.0, 80.0, 80.0);

    let gradient = Gradient {
        kind: GradientKind::Linear {
            start: Point::new(0.0, 0.0),
            end: Point::new(80.0, 0.0)
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

    check_ref(&ctx, name);
}

#[test]
fn mix_normal() {

    mix("mix_normal", BlendMode::new(Mix::Normal, Compose::SrcOver));
}

#[test]
fn mix_multiply() {
    mix("mix_multiply", BlendMode::new(Mix::Multiply, Compose::SrcOver));
}

#[test]
fn mix_screen() {
    mix("mix_screen", BlendMode::new(Mix::Screen, Compose::SrcOver));
}

#[test]
fn mix_darken() {
    mix("mix_darken", BlendMode::new(Mix::Darken, Compose::SrcOver));
}

#[test]
fn mix_lighten() {
    mix("mix_lighten", BlendMode::new(Mix::Lighten, Compose::SrcOver));
}

#[test]
fn mix_color_dodge() {
    mix("mix_color_dodge", BlendMode::new(Mix::ColorDodge, Compose::SrcOver));
}

#[test]
fn mix_color_burn() {
    mix("mix_color_burn", BlendMode::new(Mix::ColorBurn, Compose::SrcOver));
}

#[test]
fn mix_hard_light() {
    mix("mix_hard_light", BlendMode::new(Mix::HardLight, Compose::SrcOver));
}

#[test]
fn mix_soft_light() {
    mix("mix_soft_light", BlendMode::new(Mix::SoftLight, Compose::SrcOver));
}

#[test]
fn mix_difference() {
    mix("mix_difference", BlendMode::new(Mix::Difference, Compose::SrcOver));
}

#[test]
fn mix_exclusion() {
    mix("mix_exclusion", BlendMode::new(Mix::Exclusion, Compose::SrcOver));
}

#[test]
fn mix_overlay() {
    mix("mix_overlay", BlendMode::new(Mix::Overlay, Compose::SrcOver));
}

#[test]
fn mix_hue() {
    mix("mix_hue", BlendMode::new(Mix::Hue, Compose::SrcOver));
}

#[test]
fn mix_saturation() {
    mix("mix_saturation", BlendMode::new(Mix::Saturation, Compose::SrcOver));
}

#[test]
fn mix_color() {
    mix("mix_color", BlendMode::new(Mix::Color, Compose::SrcOver));
}


#[test]
fn mix_luminosity() {
    mix("mix_luminosity", BlendMode::new(Mix::Luminosity, Compose::SrcOver));
}

macro_rules! mix_solid {
    ($name:expr, $mix_mode:expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
        ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(122, 85, 73, 255));
        ctx.fill_rect(&rect);
        
        ctx.push_blend_layer($mix_mode);
        ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(255, 255, 0, 255));
        ctx.fill_rect(&rect);
        ctx.pop_layer();
    
        check_ref(&ctx, $name);
    }
}

#[test]
fn mix_with_transparent_bg() {
    let mut ctx = get_ctx(100, 100, true);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    ctx.set_paint(AlphaColor::<Srgb>::from_rgba8(0, 0, 128, 128));
    ctx.fill_rect(&rect);
    ctx.push_blend_layer(BlendMode::new(Mix::ColorDodge, Compose::SrcOver));
    ctx.set_paint(YELLOW);
    ctx.fill_rect(&rect);
    ctx.pop_layer();

    check_ref(&ctx, "mix_with_transparent_bg");
}

// The below 2 test cases yield different results than in tiny-skia, but I checked by converting
// the test case into SVG and viewing it in Chrome, and our version should be right.
// Color of rectangle should be (106, 106, 0).
#[test]
fn mix_color_with_solid() {
    mix_solid!("mix_color_with_solid", BlendMode::new(Mix::Color, Compose::SrcOver));
}

// Color of rectangle should be (213, 52, 0).
#[test]
fn mix_color_with_saturation() {
    mix_solid!("mix_saturation_with_solid", BlendMode::new(Mix::Saturation, Compose::SrcOver));
}
