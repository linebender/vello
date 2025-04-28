// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;
use vello_common::kurbo::{Affine, Point, Rect};
use vello_common::paint::Image;
use vello_common::pixmap::Pixmap;
use vello_common::peniko::{Extend, ImageQuality};
use crate::gradient::tan_45;
use crate::util::{check_ref, crossed_line_star, get_ctx};

pub(crate) fn load_image(name: &str) -> Arc<Pixmap> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/assets/{name}.png"));
    Arc::new(Pixmap::from_png(&std::fs::read(path).unwrap()).unwrap())
}

fn rgb_img_10x10() -> Arc<Pixmap> {
    load_image("rgb_image_10x10")
}

fn rgb_img_2x2() -> Arc<Pixmap> {
    load_image("rgb_image_2x2")
}

fn rgb_img_2x3() -> Arc<Pixmap> {
    load_image("rgb_image_2x3")
}

fn rgba_img_10x10() -> Arc<Pixmap> {
    load_image("rgba_image_10x10")
}

fn luma_img_10x10() -> Arc<Pixmap> {
    load_image("luma_image_10x10")
}

fn lumaa_img_10x10() -> Arc<Pixmap> {
    load_image("lumaa_image_10x10")
}

macro_rules! repeat {
    ($name:expr, $x_repeat:expr, $y_repeat:expr) => {
        let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    let im = rgb_img_10x10();
    
    ctx.set_paint(Image {
        pixmap: im,
        x_extend: $x_repeat,
        y_extend: $y_repeat,
        quality: ImageQuality::Low,
        transform: Affine::translate((45.0, 45.0)),
    });
    ctx.fill_rect(&rect);

    check_ref(&ctx, $name);
    };
}

#[test]
fn image_reflect_x_pad_y() {
    repeat!("image_reflect_x", Extend::Reflect, Extend::Pad);
}

#[test]
fn image_pad_x_repeat_y() {
    repeat!("image_repeat_y", Extend::Pad, Extend::Repeat);
}

#[test]
fn image_reflect_x_reflect_y() {
    repeat!("image_reflect_x_reflect_y", Extend::Reflect, Extend::Reflect);
}

#[test]
fn image_repeat_x_repeat_y() {
    repeat!("image_repeat_x_repeat_y", Extend::Repeat, Extend::Repeat);
}

#[test]
fn image_pad_x_pad_y() {
    repeat!("image_pad_x_pad_y", Extend::Pad, Extend::Pad);
}

macro_rules! transform {
    ($name:expr, $transform:expr, $p0:expr, $p1: expr, $p2:expr, $p3: expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new($p0, $p1, $p2, $p3);

        let image = Image {
            pixmap: rgb_img_10x10(),
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            transform: Affine::IDENTITY,
        };

        ctx.set_transform($transform);
        ctx.set_paint(image);
        ctx.fill_rect(&rect);

        check_ref(&ctx, $name);
    };
}

#[test]
fn image_with_transform_identity() {
    transform!(
            "image_with_transform_identity",
            Affine::IDENTITY,
            25.0,
            25.0,
            75.0,
            75.0
        );
}

#[test]
fn image_with_transform_translate() {
    transform!(
            "image_with_transform_translate",
            Affine::translate((25.0, 25.0)),
            0.0,
            0.0,
            50.0,
            50.0
        );
}

#[test]
fn image_with_transform_scale() {
    transform!(
            "image_with_transform_scale",
            Affine::scale(2.0),
            12.5,
            12.5,
            37.5,
            37.5
        );
}

#[test]
fn image_with_transform_negative_scale() {
    transform!(
            "image_with_transform_negative_scale",
            Affine::translate((100.0, 100.0)) * Affine::scale(-2.0),
            12.5,
            12.5,
            37.5,
            37.5
        );
}

#[test]
fn image_with_transform_scale_and_translate() {
    transform!(
            "image_with_transform_scale_and_translate",
            Affine::new([2.0, 0.0, 0.0, 2.0, 25.0, 25.0]),
            0.0,
            0.0,
            25.0,
            25.0
        );
}

// TODO: The below two test cases fail on Windows CI for some reason.
#[test]
#[ignore]
fn image_with_transform_rotate_1() {
    transform!(
            "image_with_transform_rotate_1",
            Affine::rotate_about(PI / 4.0, Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0
        );
}

#[test]
#[ignore]
fn image_with_transform_rotate_2() {
    transform!(
            "image_with_transform_rotate_2",
            Affine::rotate_about(-PI / 4.0, Point::new(50.0, 50.0)),
            25.0,
            25.0,
            75.0,
            75.0
        );
}

#[test]
fn image_with_transform_scaling_non_uniform() {
    transform!(
            "image_with_transform_scaling_non_uniform",
            Affine::scale_non_uniform(1.0, 2.0),
            25.0,
            12.5,
            75.0,
            37.5
        );
}

#[test]
fn image_with_transform_skew_x_1() {
    let transform = Affine::translate((-50.0, 0.0)) * Affine::skew(tan_45(), 0.0);
    transform!(
            "image_with_transform_skew_x_1",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
}

#[test]
fn image_with_transform_skew_x_2() {
    let transform = Affine::translate((50.0, 0.0)) * Affine::skew(-tan_45(), 0.0);
    transform!(
            "image_with_transform_skew_x_2",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
}

#[test]
fn image_with_transform_skew_y_1() {
    let transform = Affine::translate((0.0, 50.0)) * Affine::skew(0.0, -tan_45());
    transform!(
            "image_with_transform_skew_y_1",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
}

#[test]
fn image_with_transform_skew_y_2() {
    let transform = Affine::translate((0.0, -50.0)) * Affine::skew(0.0, tan_45());
    transform!(
            "image_with_transform_skew_y_2",
            transform,
            25.0,
            25.0,
            75.0,
            75.0
        );
}

#[test]
fn image_complex_shape() {
    let mut ctx = get_ctx(100, 100, false);
    let path = crossed_line_star();

    let image = Image {
        pixmap: rgb_img_10x10(),
        x_extend: Extend::Repeat,
        y_extend: Extend::Repeat,
        quality: ImageQuality::Low,
        transform: Affine::IDENTITY,
    };

    ctx.set_paint(image);
    ctx.fill_path(&path);

    check_ref(&ctx, "image_complex_shape");
}

#[test]
fn image_global_alpha() {
    let mut ctx = get_ctx(100, 100, false);
    let rect = Rect::new(10.0, 10.0, 90.0 ,90.0);
    
    let mut pix = rgb_img_10x10();
    Arc::make_mut(&mut pix).multiply_alpha(75);

    let image = Image {
        pixmap: pix,
        x_extend: Extend::Repeat,
        y_extend: Extend::Repeat,
        quality: ImageQuality::Low,
        transform: Affine::IDENTITY,
    };

    ctx.set_paint(image);
    ctx.fill_rect(&rect);

    check_ref(&ctx, "image_global_alpha");
}

macro_rules! image_format {
    ($name:expr, $image:expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0 ,90.0);
    
        let image = Image {
            pixmap: $image,
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            transform: Affine::IDENTITY,
        };
    
        ctx.set_paint(image);
        ctx.fill_rect(&rect);
    
        check_ref(&ctx, $name);
    };
}

#[test]
fn image_rgb_image() {
    image_format!("image_rgb_image", rgb_img_10x10());
}

#[test]
fn image_rgba_image() {
    image_format!("image_rgba_image", rgba_img_10x10());
}

#[test]
fn image_luma_image() {
    image_format!("image_luma_image", luma_img_10x10());
}

#[test]
fn image_lumaa_image() {
    image_format!("image_lumaa_image", lumaa_img_10x10());
}

macro_rules! quality {
    ($name:expr, $transform:expr, $image:expr, $quality:expr, $extend:expr) => {
        let mut ctx = get_ctx(100, 100, false);
        let rect = Rect::new(10.0, 10.0, 90.0, 90.0);
    
        let image = Image {
            pixmap: $image,
            x_extend: $extend,
            y_extend: $extend,
            quality: $quality,
            transform: $transform,
        };
    
        ctx.set_paint(image);
        ctx.fill_rect(&rect);
    
        check_ref(&ctx, $name);
    };
}

// Outputs of those tests were compared against Blend2D and tiny-skia.

#[test]
fn image_bilinear_identity() {
    quality!("image_bilinear_identity", Affine::IDENTITY, rgb_img_2x2(), ImageQuality::Medium, Extend::Reflect);
}

#[test]
fn image_bilinear_2x_scale() {
    quality!("image_bilinear_2x_scale", Affine::scale(2.0), rgb_img_2x2(), ImageQuality::Medium, Extend::Reflect);
}

#[test]
fn image_bilinear_5x_scale() {
    quality!("image_bilinear_5x_scale", Affine::scale(5.0), rgb_img_2x2(), ImageQuality::Medium, Extend::Reflect);
}

#[test]
fn image_bilinear_10x_scale() {
    quality!("image_bilinear_10x_scale", Affine::scale(10.0), rgb_img_2x2(), ImageQuality::Medium, Extend::Reflect);
}

#[test]
fn image_bilinear_with_rotation() {
    quality!("image_bilinear_with_rotation", Affine::scale(5.0) * Affine::rotate(45.0_f64.to_radians()), rgb_img_2x2(), ImageQuality::Medium, Extend::Reflect);
}

#[test]
fn image_bilinear_with_translation() {
    quality!("image_bilinear_with_translation", Affine::scale(5.0) * Affine::translate((10.0, 10.0)), rgb_img_2x2(), ImageQuality::Medium, Extend::Reflect);
}

#[test]
fn image_bilinear_10x_scale_2() {
    quality!("image_bilinear_10x_scale_2", Affine::scale(10.0), rgb_img_2x3(), ImageQuality::Medium, Extend::Reflect);
}

// This one looks slightly different from tiny-skia. In tiny-skia, it looks exactly the same as with
// `Nearest`, while in our case it looks overall a bit darker. I'm not 100% sure who is right here,
// but I think ours should be correct, because AFAIK for bicubic scaling, the output image does
// not necessarily need to look the same as with `Nearest` with identity scaling. Would be nice to
// verify this somehow, though.
//
// We also ported the cubic polynomials directly from current Skia, while tiny-skia (seems?) to use
// either an outdated version or a slightly adapted one.
#[test]
fn image_bicubic_identity() {
    quality!("image_bicubic_identity", Affine::IDENTITY, rgb_img_2x2(), ImageQuality::High, Extend::Reflect);
}

#[test]
fn image_bicubic_2x_scale() {
    quality!("image_bicubic_2x_scale", Affine::scale(2.0), rgb_img_2x2(), ImageQuality::High, Extend::Reflect);
}

#[test]
fn image_bicubic_5x_scale() {
    quality!("image_bicubic_5x_scale", Affine::scale(5.0), rgb_img_2x2(), ImageQuality::High, Extend::Reflect);
}

#[test]
fn image_bicubic_10x_scale() {
    quality!("image_bicubic_10x_scale", Affine::scale(10.0), rgb_img_2x2(), ImageQuality::High, Extend::Reflect);
}

#[test]
fn image_bicubic_with_rotation() {
    quality!("image_bicubic_with_rotation", Affine::scale(5.0) * Affine::rotate(45.0_f64.to_radians()), rgb_img_2x2(), ImageQuality::High, Extend::Reflect);
}

#[test]
fn image_bicubic_with_translation() {
    quality!("image_bicubic_with_translation", Affine::scale(5.0) * Affine::translate((10.0, 10.0)), rgb_img_2x2(), ImageQuality::High, Extend::Reflect);
}

#[test]
fn image_bicubic_10x_scale_2() {
    quality!("image_bicubic_10x_scale_2", Affine::scale(10.0), rgb_img_2x3(), ImageQuality::High, Extend::Reflect);
}