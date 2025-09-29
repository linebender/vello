// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::gradient::tan_45;
use crate::load_image;
use crate::renderer::Renderer;
use crate::util::crossed_line_star;
use std::f64::consts::PI;
use std::sync::Arc;
use vello_common::color::palette::css::REBECCA_PURPLE;
use vello_common::kurbo::{Affine, Point, Rect};
use vello_common::kurbo::{Shape, Triangle};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::ImageSampler;
use vello_common::peniko::{Extend, ImageQuality};
use vello_dev_macros::vello_test;

fn rgb_img_10x10(ctx: &mut impl Renderer) -> ImageSource {
    ctx.get_image_source(load_image!("rgb_image_10x10"))
}

fn rgb_img_10x10_alpha_multiplied(ctx: &mut impl Renderer, alpha: u8) -> ImageSource {
    let mut pix = load_image!("rgb_image_10x10");
    Arc::make_mut(&mut pix).multiply_alpha(alpha);
    ctx.get_image_source(pix)
}

fn rgb_img_2x2(ctx: &mut impl Renderer) -> ImageSource {
    ctx.get_image_source(load_image!("rgb_image_2x2"))
}

fn rgb_img_2x3(ctx: &mut impl Renderer) -> ImageSource {
    ctx.get_image_source(load_image!("rgb_image_2x3"))
}

fn rgba_img_10x10(ctx: &mut impl Renderer) -> ImageSource {
    ctx.get_image_source(load_image!("rgba_image_10x10"))
}

fn luma_img_10x10(ctx: &mut impl Renderer) -> ImageSource {
    ctx.get_image_source(load_image!("luma_image_10x10"))
}

fn lumaa_img_10x10(ctx: &mut impl Renderer) -> ImageSource {
    ctx.get_image_source(load_image!("lumaa_image_10x10"))
}

fn repeat(ctx: &mut impl Renderer, x_extend: Extend, y_extend: Extend) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    let image_source = rgb_img_10x10(ctx);

    ctx.set_paint_transform(Affine::translate((45.0, 45.0)));
    ctx.set_paint(Image {
        image: image_source,
        sampler: ImageSampler {
            x_extend,
            y_extend,
            quality: ImageQuality::Low,
            alpha: 1.0,
        },
    });
    ctx.fill_rect(&rect);
}

#[vello_test]
fn image_reflect_x_pad_y(ctx: &mut impl Renderer) {
    repeat(ctx, Extend::Reflect, Extend::Pad);
}

#[vello_test]
fn image_pad_x_repeat_y(ctx: &mut impl Renderer) {
    repeat(ctx, Extend::Pad, Extend::Repeat);
}

#[vello_test]
fn image_reflect_x_reflect_y(ctx: &mut impl Renderer) {
    repeat(ctx, Extend::Reflect, Extend::Reflect);
}

#[vello_test]
fn image_repeat_x_repeat_y(ctx: &mut impl Renderer) {
    repeat(ctx, Extend::Repeat, Extend::Repeat);
}

#[vello_test]
fn image_pad_x_pad_y(ctx: &mut impl Renderer) {
    repeat(ctx, Extend::Pad, Extend::Pad);
}

fn transform(ctx: &mut impl Renderer, transform: Affine, l: f64, t: f64, r: f64, b: f64) {
    let rect = Rect::new(l, t, r, b);
    let image_source = rgb_img_10x10(ctx);

    let image = Image {
        image: image_source,
        sampler: ImageSampler {
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            alpha: 1.0,
        },
    };

    ctx.set_transform(transform);
    ctx.set_paint(image);
    ctx.fill_rect(&rect);
}

#[vello_test]
fn image_with_transform_identity(ctx: &mut impl Renderer) {
    transform(ctx, Affine::IDENTITY, 25.0, 25.0, 75.0, 75.0);
}

#[vello_test]
fn image_with_transform_translate(ctx: &mut impl Renderer) {
    transform(ctx, Affine::translate((25.0, 25.0)), 0.0, 0.0, 50.0, 50.0);
}

#[vello_test]
fn image_with_transform_scale(ctx: &mut impl Renderer) {
    transform(ctx, Affine::scale(2.0), 12.5, 12.5, 37.5, 37.5);
}

#[vello_test]
fn image_with_transform_negative_scale(ctx: &mut impl Renderer) {
    transform(
        ctx,
        Affine::translate((100.0, 100.0)) * Affine::scale(-2.0),
        12.5,
        12.5,
        37.5,
        37.5,
    );
}

#[vello_test]
fn image_with_transform_scale_and_translate(ctx: &mut impl Renderer) {
    transform(
        ctx,
        Affine::new([2.0, 0.0, 0.0, 2.0, 25.0, 25.0]),
        0.0,
        0.0,
        25.0,
        25.0,
    );
}

#[vello_test(ignore = "fails in Windows CI for some reason.")]
fn image_with_transform_rotate_1(ctx: &mut impl Renderer) {
    transform(
        ctx,
        Affine::rotate_about(PI / 4.0, Point::new(50.0, 50.0)),
        25.0,
        25.0,
        75.0,
        75.0,
    );
}

#[vello_test(ignore = "fails in Windows CI for some reason.")]
fn image_with_transform_rotate_2(ctx: &mut impl Renderer) {
    transform(
        ctx,
        Affine::rotate_about(-PI / 4.0, Point::new(50.0, 50.0)),
        25.0,
        25.0,
        75.0,
        75.0,
    );
}

#[vello_test]
fn image_with_transform_scaling_non_uniform(ctx: &mut impl Renderer) {
    transform(
        ctx,
        Affine::scale_non_uniform(1.0, 2.0),
        25.0,
        12.5,
        75.0,
        37.5,
    );
}

#[vello_test]
fn image_with_transform_skew_x_1(ctx: &mut impl Renderer) {
    transform(
        ctx,
        Affine::translate((-50.0, 0.0)) * Affine::skew(tan_45(), 0.0),
        25.0,
        25.0,
        75.0,
        75.0,
    );
}

#[vello_test]
fn image_with_transform_skew_x_2(ctx: &mut impl Renderer) {
    transform(
        ctx,
        Affine::translate((50.0, 0.0)) * Affine::skew(-tan_45(), 0.0),
        25.0,
        25.0,
        75.0,
        75.0,
    );
}

#[vello_test]
fn image_with_transform_skew_y_1(ctx: &mut impl Renderer) {
    transform(
        ctx,
        Affine::translate((0.0, 50.0)) * Affine::skew(0.0, -tan_45()),
        25.0,
        25.0,
        75.0,
        75.0,
    );
}

#[vello_test]
fn image_with_transform_skew_y_2(ctx: &mut impl Renderer) {
    transform(
        ctx,
        Affine::translate((0.0, -50.0)) * Affine::skew(0.0, tan_45()),
        25.0,
        25.0,
        75.0,
        75.0,
    );
}

#[vello_test]
fn image_complex_shape(ctx: &mut impl Renderer) {
    let path = crossed_line_star();
    let image_source = rgb_img_10x10(ctx);

    let image = Image {
        image: image_source,
        sampler: ImageSampler {
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            alpha: 1.0,
        },
    };

    ctx.set_paint(image);
    ctx.fill_path(&path);
}

#[vello_test]
fn image_global_alpha(ctx: &mut impl Renderer) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let image = Image {
        image: rgb_img_10x10_alpha_multiplied(ctx, 75),
        sampler: ImageSampler {
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            alpha: 1.0,
        },
    };

    ctx.set_paint(image);
    ctx.fill_rect(&rect);
}

#[vello_test]
fn image_with_opacity(ctx: &mut impl Renderer) {
    ctx.set_paint(REBECCA_PURPLE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));

    ctx.push_opacity_layer(0.5);

    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    let image_source = rgb_img_10x10(ctx);

    let image = Image {
        image: image_source,
        sampler: ImageSampler {
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            alpha: 1.0,
        },
    };

    ctx.set_paint(image);
    ctx.fill_rect(&rect);

    ctx.pop_layer();
}

fn image_format(ctx: &mut impl Renderer, image_source: ImageSource) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    let image = Image {
        image: image_source,
        sampler: ImageSampler {
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            alpha: 1.0,
        },
    };

    ctx.set_paint(image);
    ctx.fill_rect(&rect);
}

#[vello_test]
fn image_rgb_image(ctx: &mut impl Renderer) {
    let image_source = rgb_img_10x10(ctx);
    image_format(ctx, image_source);
}

#[vello_test]
fn image_rgba_image(ctx: &mut impl Renderer) {
    let image_source = rgba_img_10x10(ctx);
    image_format(ctx, image_source);
}

#[vello_test]
fn image_luma_image(ctx: &mut impl Renderer) {
    let image_source = luma_img_10x10(ctx);
    image_format(ctx, image_source);
}

#[vello_test]
fn image_lumaa_image(ctx: &mut impl Renderer) {
    let image_source = lumaa_img_10x10(ctx);
    image_format(ctx, image_source);
}

fn quality(
    ctx: &mut impl Renderer,
    transform: Affine,
    image_source: ImageSource,
    quality: ImageQuality,
    extend: Extend,
) {
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);

    ctx.set_paint_transform(transform);
    let image = Image {
        image: image_source,
        sampler: ImageSampler {
            x_extend: extend,
            y_extend: extend,
            quality,
            alpha: 1.0,
        },
    };

    ctx.set_paint(image);
    ctx.fill_rect(&rect);
}

// Outputs of those tests were compared against Blend2D and tiny-skia.

#[vello_test]
fn image_bilinear_identity(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::IDENTITY,
        image_source,
        ImageQuality::Medium,
        Extend::Reflect,
    );
}

#[vello_test]
fn image_bilinear_2x_scale(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::scale(2.0),
        image_source,
        ImageQuality::Medium,
        Extend::Reflect,
    );
}

#[vello_test]
fn image_bilinear_5x_scale(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::scale(5.0),
        image_source,
        ImageQuality::Medium,
        Extend::Reflect,
    );
}

#[vello_test]
fn image_bilinear_10x_scale(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::scale(10.0),
        image_source,
        ImageQuality::Medium,
        Extend::Reflect,
    );
}

#[vello_test]
fn image_bilinear_with_rotation(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::scale(5.0) * Affine::rotate(45.0_f64.to_radians()),
        image_source,
        ImageQuality::Medium,
        Extend::Reflect,
    );
}

#[vello_test]
fn image_bilinear_with_translation(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::scale(5.0) * Affine::translate((10.0, 10.0)),
        image_source,
        ImageQuality::Medium,
        Extend::Reflect,
    );
}

#[vello_test]
fn image_bilinear_10x_scale_2(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x3(ctx);
    quality(
        ctx,
        Affine::scale(10.0),
        image_source,
        ImageQuality::Medium,
        Extend::Reflect,
    );
}

// This one looks slightly different from tiny-skia. In tiny-skia, it looks exactly the same as with
// `Nearest`, while in our case it looks overall a bit darker. I'm not 100% sure who is right here,
// but I think ours should be correct, because AFAIK for bicubic scaling, the output image does
// not necessarily need to look the same as with `Nearest` with identity scaling. Would be nice to
// verify this somehow, though.
//
// We also ported the cubic polynomials directly from current Skia, while tiny-skia (seems?) to use
// either an outdated version or a slightly adapted one.
#[vello_test]
fn image_bicubic_identity(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::IDENTITY,
        image_source,
        ImageQuality::High,
        Extend::Reflect,
    );
}

#[vello_test(hybrid_tolerance = 3)]
fn image_bicubic_2x_scale(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::scale(2.0),
        image_source,
        ImageQuality::High,
        Extend::Reflect,
    );
}

#[vello_test(hybrid_tolerance = 5)]
fn image_bicubic_5x_scale(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::scale(5.0),
        image_source,
        ImageQuality::High,
        Extend::Reflect,
    );
}

#[vello_test(hybrid_tolerance = 7)]
fn image_bicubic_10x_scale(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::scale(10.0),
        image_source,
        ImageQuality::High,
        Extend::Reflect,
    );
}

#[vello_test(hybrid_tolerance = 7)]
fn image_bicubic_with_rotation(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::scale(5.0) * Affine::rotate(45.0_f64.to_radians()),
        image_source,
        ImageQuality::High,
        Extend::Reflect,
    );
}

#[vello_test(hybrid_tolerance = 5)]
fn image_bicubic_with_translation(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    quality(
        ctx,
        Affine::scale(5.0) * Affine::translate((10.0, 10.0)),
        image_source,
        ImageQuality::High,
        Extend::Reflect,
    );
}

#[vello_test(hybrid_tolerance = 7)]
fn image_bicubic_10x_scale_2(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x3(ctx);
    quality(
        ctx,
        Affine::scale(10.0),
        image_source,
        ImageQuality::High,
        Extend::Reflect,
    );
}

#[vello_test]
fn image_with_multiple_clip_layers(ctx: &mut impl Renderer) {
    let image_source = rgb_img_2x2(ctx);
    let image_rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    let clipped_area1 = Rect::new(20.0, 20.0, 80.0, 80.0);
    let clipped_area2 = Triangle::new(
        Point::new(90.0, 10.0),
        Point::new(32.0, 46.0),
        Point::new(54.0, 68.0),
    );

    ctx.push_clip_layer(&clipped_area1.to_path(0.1));
    ctx.push_clip_layer(&clipped_area2.to_path(0.1));
    ctx.set_paint_transform(Affine::IDENTITY);
    ctx.set_paint(Image {
        image: image_source,
        sampler: ImageSampler {
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            alpha: 1.0,
        },
    });
    ctx.fill_rect(&image_rect);
    ctx.pop_layer();
    ctx.pop_layer();
}
