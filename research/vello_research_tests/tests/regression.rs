// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests to ensure that certain issues which don't deserve a test scene don't regress

use scenes::ImageCache;
use scenes::SimpleText;
use vello::{
    AaConfig, Scene,
    kurbo::{Affine, Rect, RoundedRect, Stroke},
    peniko::{
        Color, ColorStop, Extend, Gradient, ImageFormat, ImageQuality, InterpolationAlphaSpace,
        color::palette,
    },
};
use vello_tests::{TestParams, smoke_snapshot_test_sync, snapshot_test_sync};

/// Test created from <https://github.com/linebender/vello/issues/616>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn rounded_rectangle_watertight() {
    let mut scene = Scene::new();
    let rect = RoundedRect::new(60.0, 10.0, 80.0, 30.0, 10.0);
    let stroke = Stroke::new(2.0);
    scene.stroke(&stroke, Affine::IDENTITY, palette::css::WHITE, None, &rect);
    let mut params = TestParams::new("rounded_rectangle_watertight", 70, 30);
    params.anti_aliasing = AaConfig::Msaa16;
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

const DATA_IMAGE_PNG: &[u8] = include_bytes!("../snapshots/smoke/data_image_roundtrip.png");

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_data_image_roundtrip_extend_pad() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Pad);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new(
        "data_image_roundtrip",
        image.image.width,
        image.image.height,
    );
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_data_image_roundtrip_extend_reflect() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Reflect);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new(
        "data_image_roundtrip",
        image.image.width,
        image.image.height,
    );
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_data_image_roundtrip_extend_repeat() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Repeat);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new(
        "data_image_roundtrip",
        image.image.width,
        image.image.height,
    );
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Test created from <https://github.com/linebender/vello/issues/662>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stroke_width_zero() {
    let mut scene = Scene::new();
    let stroke = Stroke::new(0.0);
    let rect = Rect::new(10.0, 10.0, 40.0, 40.0);
    let rect_stroke_color = palette::css::PEACH_PUFF;
    scene.stroke(&stroke, Affine::IDENTITY, rect_stroke_color, None, &rect);
    let mut params = TestParams::new("stroke_width_zero", 50, 50);
    params.anti_aliasing = AaConfig::Msaa16;
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
#[expect(clippy::cast_possible_truncation, reason = "Test code")]
fn text_stroke_width_zero() {
    let font_size = 12.;
    let mut scene = Scene::new();
    let mut simple_text = SimpleText::new();
    simple_text.add_run(
        &mut scene,
        None,
        font_size,
        palette::css::WHITE,
        Affine::translate((0., f64::from(font_size))),
        None,
        None,
        &Stroke::new(0.),
        "Testing text",
    );
    let params = TestParams::new(
        "text_stroke_width_zero",
        (font_size * 6.) as _,
        (font_size * 1.25).ceil() as _,
    );
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// <https://github.com/web-platform-tests/wpt/blob/18c64a74b1/html/canvas/element/fill-and-stroke-styles/2d.gradient.interpolate.coloralpha.html>
/// See <https://github.com/linebender/vello/issues/1056>.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_gradient_color_alpha_premultiplied() {
    let mut scene = Scene::new();
    let viewport = Rect::new(0., 0., 100., 50.);
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Gradient::new_linear((0., 0.), (100., 0.))
            .with_stops([
                ColorStop {
                    offset: 0.,
                    color: Color::from_rgba8(255, 255, 0, 0).into(),
                },
                ColorStop {
                    offset: 1.,
                    color: Color::from_rgba8(0, 0, 255, 255).into(),
                },
            ])
            .with_interpolation_alpha_space(InterpolationAlphaSpace::Premultiplied),
        None,
        &viewport,
    );
    let mut params = TestParams::new("gradient_color_alpha_premultiplied", 100, 50);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// <https://github.com/web-platform-tests/wpt/blob/18c64a74b1/html/canvas/element/fill-and-stroke-styles/2d.gradient.interpolate.coloralpha.html>
/// See <https://github.com/linebender/vello/issues/1056>.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_gradient_color_alpha_unpremultiplied() {
    let mut scene = Scene::new();
    let viewport = Rect::new(0., 0., 100., 50.);
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Gradient::new_linear((0., 0.), (100., 0.))
            .with_stops([
                ColorStop {
                    offset: 0.,
                    color: Color::from_rgba8(255, 255, 0, 0).into(),
                },
                ColorStop {
                    offset: 1.,
                    color: Color::from_rgba8(0, 0, 255, 255).into(),
                },
            ])
            .with_interpolation_alpha_space(InterpolationAlphaSpace::Unpremultiplied),
        None,
        &viewport,
    );
    let mut params = TestParams::new("gradient_color_alpha_unpremultiplied", 100, 50);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Test created from <https://github.com/linebender/vello/issues/680>
fn many_bins(use_cpu: bool) {
    let mut scene = Scene::new();
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        palette::css::RED,
        None,
        &Rect::new(-5., -5., 256. * 20., 256. * 20.),
    );
    let params = TestParams {
        use_cpu,
        ..TestParams::new("many_bins", 256 * 17, 256 * 17)
    };
    let image = vello_tests::render_then_debug_sync(&scene, &params).unwrap();
    assert_eq!(image.format, ImageFormat::Rgba8, "image should be Rgba8");
    let mut red_count = 0;
    let mut black_count = 0;
    for pixel in image.data.data().chunks_exact(4) {
        let &[r, g, b, a] = pixel else { unreachable!() };
        let is_red = r == 255 && g == 0 && b == 0 && a == 255;
        let is_black = r == 0 && g == 0 && b == 0 && a == 255;
        if !is_red && !is_black {
            panic!("{pixel:?}");
        }
        match (is_red, is_black) {
            (true, true) => unreachable!(),
            (true, false) => red_count += 1,
            (false, true) => black_count += 1,
            (false, false) => panic!("Got unexpected pixel {pixel:?}"),
        }
    }
    let drawn_bins = 17 /* x bins */ * 17 /* y bins*/;
    let expected_red_count = drawn_bins * 256 /* tiles per bin */ * 256 /* Pixels per tile */;
    assert_eq!(
        red_count, expected_red_count,
        "number of drawn red pixels should match the expected pixel count for 17x17 bins"
    );
    assert_eq!(
        black_count, 0,
        "no black pixels should remain in the render"
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn many_bins_gpu() {
    many_bins(false);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn many_bins_cpu() {
    many_bins(true);
}
