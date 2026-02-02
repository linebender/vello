// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests for stateful compositing (`Scene::set_*`).
//!
//! To generate/update reference snapshots locally:
//!
//! ```sh
//! VELLO_TEST_CREATE=all cargo test -p vello_tests --test stateful_compositing_snapshots
//! VELLO_TEST_UPDATE=all cargo test -p vello_tests --test stateful_compositing_snapshots
//! ```

use vello::Scene;
use vello::kurbo::{Affine, Circle, Rect};
use vello::peniko::{
    BlendMode, Compose, Fill, Gradient, ImageAlphaType, ImageBrush, ImageData, ImageFormat,
    ImageQuality, ImageSampler, Mix, color::palette,
};
use vello_tests::{TestParams, smoke_snapshot_test_sync};

fn rect(x: f64, y: f64, w: f64, h: f64) -> Rect {
    Rect::from_origin_size((x, y), (w, h))
}

fn assert_no_layers(scene: &Scene) {
    let encoding = scene.encoding();
    assert!(
        // Avoid depending directly on `vello_encoding` from this test crate.
        // These are stable draw-tag values within Vello Classic:
        // - BEGIN_CLIP = 0x49
        // - END_CLIP = 0x21
        !encoding
            .draw_tags
            .iter()
            .any(|tag| tag.0 == 0x49 || tag.0 == 0x21),
        "stateful compositing should not be emulated by inserting layers"
    );
}

fn pixel_rgba8(image: &ImageData, x: u32, y: u32) -> [u8; 4] {
    assert!(
        x < image.width && y < image.height,
        "pixel ({x},{y}) out of bounds for {}x{}",
        image.width,
        image.height
    );
    let i = ((y * image.width + x) * 4) as usize;
    let bytes = image.data.data();
    [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]
}

fn draw_checkerboard(scene: &mut Scene, width: u32, height: u32, cell: u32) {
    let light = palette::css::WHITE;
    let dark = palette::css::BLACK;
    let cells_x = width.div_ceil(cell);
    let cells_y = height.div_ceil(cell);
    for y in 0..cells_y {
        for x in 0..cells_x {
            let is_light = ((x + y) & 1) == 0;
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                if is_light { light } else { dark },
                None,
                &rect(
                    x as f64 * f64::from(cell),
                    y as f64 * f64::from(cell),
                    f64::from(cell),
                    f64::from(cell),
                ),
            );
        }
    }
}

fn assert_half_alpha_over_white_square(rgba: [u8; 4]) {
    // With an opaque background, SrcOver yields opaque output; we check that global alpha had an
    // effect by requiring the output not be a pure primary color.
    let [r, g, b, a] = rgba;
    assert!(
        a >= 250,
        "expected opaque output over opaque background, got {rgba:?}"
    );
    assert!(
        r > 30 && b > 30,
        "expected tinted color (alpha applied), got {rgba:?}"
    );
    assert!(
        g >= 200,
        "expected strong green-ish component, got {rgba:?}"
    );
}

fn assert_half_alpha_skyblue_over_white_square(rgba: [u8; 4]) {
    // DeepSkyBlue (0,191,255) over white (255,255,255) with alpha=0.5 gives ~ (128,223,255).
    // This isn't exact (premul/rounding), but should stay comfortably in these ranges.
    let [r, g, b, a] = rgba;
    assert!(
        a >= 250,
        "expected opaque output over opaque background, got {rgba:?}"
    );
    assert!(
        (90..=170).contains(&r),
        "expected ~half-mix red channel, got {rgba:?}"
    );
    assert!(
        (190..=255).contains(&g),
        "expected high green channel, got {rgba:?}"
    );
    assert!(
        (230..=255).contains(&b),
        "expected high blue channel, got {rgba:?}"
    );
}

#[test]
fn stateful_compositing_encodes_without_layers() {
    let mut scene = Scene::new();

    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::GRAY,
        None,
        &rect(0.0, 0.0, 64.0, 64.0),
    );

    scene.set_blend_mode(Mix::Multiply);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::DEEP_SKY_BLUE,
        None,
        &rect(8.0, 8.0, 48.0, 48.0),
    );

    scene.set_composite(
        BlendMode {
            mix: Mix::Normal,
            compose: Compose::Copy,
        },
        0.5,
    );
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::LIME,
        None,
        &rect(16.0, 16.0, 32.0, 32.0),
    );

    assert_no_layers(&scene);

    let encoding = scene.encoding();
    assert!(
        encoding.styles.len() >= 3,
        "expected multiple styles due to composite state changes"
    );
    assert!(
        encoding
            .styles
            .windows(2)
            .any(|w| w[0].composite != w[1].composite),
        "expected composite state to be captured into styles"
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stateful_multiply_and_alpha() {
    let mut scene = Scene::new();

    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::GRAY,
        None,
        &rect(0.0, 0.0, 64.0, 64.0),
    );

    scene.set_blend_mode(Mix::Multiply);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::DEEP_SKY_BLUE,
        None,
        &rect(8.0, 8.0, 48.0, 48.0),
    );

    scene.set_blend_mode(Mix::Normal);
    scene.set_global_alpha(0.5);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::LIME,
        None,
        &rect(16.0, 16.0, 32.0, 32.0),
    );

    assert_no_layers(&scene);

    let mut params = TestParams::new("stateful_multiply_and_alpha", 64, 64);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stateful_compose_copy() {
    let mut scene = Scene::new();

    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::RED,
        None,
        &rect(0.0, 0.0, 64.0, 64.0),
    );

    scene.set_blend_mode(BlendMode {
        mix: Mix::Normal,
        compose: Compose::Copy,
    });
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::GREEN,
        None,
        &rect(8.0, 8.0, 48.0, 48.0),
    );

    assert_no_layers(&scene);

    let mut params = TestParams::new("stateful_compose_copy", 64, 64);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stateful_compose_xor() {
    let mut scene = Scene::new();

    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::BLUE,
        None,
        &rect(0.0, 0.0, 64.0, 64.0),
    );

    scene.set_blend_mode(BlendMode {
        mix: Mix::Normal,
        compose: Compose::Xor,
    });
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::YELLOW,
        None,
        &rect(16.0, 0.0, 48.0, 64.0),
    );

    assert_no_layers(&scene);

    let mut params = TestParams::new("stateful_compose_xor", 64, 64);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Visual proof that "emulate stateful blending by using a single layer" is not equivalent.
///
/// Left half: stateful per-draw `multiply` (new capability).
/// Right half: a single isolated `multiply` layer (old workaround).
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stateful_vs_isolated_layer_multiply() {
    let mut scene = Scene::new();

    // Background (same for both halves).
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::WHITE,
        None,
        &rect(0.0, 0.0, 64.0, 64.0),
    );
    // Two-tone backdrop so differences are easier to see.
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::LIGHT_GRAY,
        None,
        &rect(0.0, 0.0, 64.0, 32.0),
    );

    // Left half: stateful multiply.
    scene.set_blend_mode(Mix::Multiply);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::DEEP_SKY_BLUE,
        None,
        &rect(4.0, 8.0, 24.0, 48.0),
    );
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::YELLOW,
        None,
        &rect(12.0, 16.0, 24.0, 24.0),
    );
    scene.set_blend_mode(Mix::Normal);

    // Right half: isolated layer multiply (single group blend).
    let clip = rect(32.0, 0.0, 32.0, 64.0);
    scene.push_layer(Fill::NonZero, Mix::Multiply, 1.0, Affine::IDENTITY, &clip);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::DEEP_SKY_BLUE,
        None,
        &rect(36.0, 8.0, 24.0, 48.0),
    );
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        palette::css::YELLOW,
        None,
        &rect(44.0, 16.0, 24.0, 24.0),
    );
    scene.pop_layer();

    let mut params = TestParams::new("stateful_vs_isolated_layer_multiply", 64, 64);
    params.base_color = Some(palette::css::BLACK);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stateful_global_alpha_applies_to_linear_gradient() {
    let mut scene = Scene::new();

    draw_checkerboard(&mut scene, 32, 32, 8);

    scene.set_composite(
        BlendMode {
            mix: Mix::Normal,
            compose: Compose::SrcOver,
        },
        0.5,
    );

    let grad = Gradient::new_linear((0.0, 0.0), (32.0, 0.0)).with_stops([palette::css::LIME; 2]);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &grad,
        None,
        &rect(0.0, 0.0, 32.0, 32.0),
    );

    assert_no_layers(&scene);

    let params = TestParams::new("stateful_alpha_lin_grad", 32, 32);
    let mut snapshot = smoke_snapshot_test_sync(scene, &params).unwrap();
    snapshot.assert_mean_less_than(0.001);
    assert_half_alpha_over_white_square(pixel_rgba8(&snapshot.raw_rendered, 4, 4));
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stateful_global_alpha_applies_to_radial_gradient() {
    let mut scene = Scene::new();

    draw_checkerboard(&mut scene, 32, 32, 8);

    scene.set_composite(
        BlendMode {
            mix: Mix::Normal,
            compose: Compose::SrcOver,
        },
        0.5,
    );

    let grad = Gradient::new_two_point_radial((16.0, 16.0), 0.0_f32, (16.0, 16.0), 16.0_f32)
        .with_stops([palette::css::LIME; 2]);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &grad,
        None,
        &rect(0.0, 0.0, 32.0, 32.0),
    );

    assert_no_layers(&scene);

    let params = TestParams::new("stateful_alpha_rad_grad", 32, 32);
    let mut snapshot = smoke_snapshot_test_sync(scene, &params).unwrap();
    snapshot.assert_mean_less_than(0.001);
    assert_half_alpha_over_white_square(pixel_rgba8(&snapshot.raw_rendered, 4, 4));
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stateful_global_alpha_applies_to_sweep_gradient() {
    let mut scene = Scene::new();

    draw_checkerboard(&mut scene, 32, 32, 8);

    scene.set_composite(
        BlendMode {
            mix: Mix::Normal,
            compose: Compose::SrcOver,
        },
        0.5,
    );

    let grad = Gradient::new_sweep((16.0, 16.0), 0.0_f32, std::f32::consts::TAU)
        .with_stops([palette::css::LIME; 2]);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &grad,
        None,
        &rect(0.0, 0.0, 32.0, 32.0),
    );

    assert_no_layers(&scene);

    let params = TestParams::new("stateful_alpha_sweep_grad", 32, 32);
    let mut snapshot = smoke_snapshot_test_sync(scene, &params).unwrap();
    snapshot.assert_mean_less_than(0.001);
    assert_half_alpha_over_white_square(pixel_rgba8(&snapshot.raw_rendered, 4, 4));
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stateful_global_alpha_applies_to_images() {
    let mut scene = Scene::new();

    draw_checkerboard(&mut scene, 32, 32, 8);

    scene.set_composite(
        BlendMode {
            mix: Mix::Normal,
            compose: Compose::SrcOver,
        },
        0.5,
    );

    let image = ImageBrush {
        image: ImageData {
            // 2x2 RGBA8: red, green / blue, white
            data: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ]
            .into(),
            format: ImageFormat::Rgba8,
            width: 2,
            height: 2,
            alpha_type: ImageAlphaType::Alpha,
        },
        sampler: ImageSampler {
            quality: ImageQuality::Low,
            ..Default::default()
        },
    };
    scene.draw_image(&image, Affine::scale(16.0));

    assert_no_layers(&scene);

    let params = TestParams::new("stateful_alpha_image", 32, 32);
    let mut snapshot = smoke_snapshot_test_sync(scene, &params).unwrap();
    snapshot.assert_mean_less_than(0.001);
    // (4,4) is inside the top-left source pixel (red) and inside a white checker square.
    let rgba = pixel_rgba8(&snapshot.raw_rendered, 4, 4);
    assert!(rgba[3] >= 250, "expected opaque output, got {rgba:?}");
    assert!(
        rgba[1] > 30 && rgba[2] > 30,
        "expected tint from alpha, got {rgba:?}"
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stateful_global_alpha_applies_to_blur_rect() {
    let mut scene = Scene::new();

    draw_checkerboard(&mut scene, 32, 32, 8);

    scene.set_composite(
        BlendMode {
            mix: Mix::Normal,
            compose: Compose::SrcOver,
        },
        0.5,
    );
    scene.draw_blurred_rounded_rect(
        Affine::IDENTITY,
        rect(0.0, 0.0, 32.0, 32.0),
        palette::css::DEEP_SKY_BLUE,
        0.0,
        2.0,
    );

    assert_no_layers(&scene);

    let params = TestParams::new("stateful_alpha_blur_rect", 32, 32);
    let mut snapshot = smoke_snapshot_test_sync(scene, &params).unwrap();
    snapshot.assert_mean_less_than(0.001);
    // On an opaque checkerboard, SrcOver yields opaque output; verify global alpha by checking
    // the expected tint over a white checker cell.
    assert_half_alpha_skyblue_over_white_square(pixel_rgba8(&snapshot.raw_rendered, 16, 16));
}

/// Visual proof that stateful blend changes between draws are not equivalent to "put each draw in
/// its own isolated layer".
///
/// Left half: stateful `set_blend_mode` changes (new capability).
/// Right half: per-draw isolated layers (old workaround).
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stateful_vs_layer_emulation_blend_switch_image_grad_blur() {
    let mut scene = Scene::new();

    draw_checkerboard(&mut scene, 64, 64, 8);

    let image = ImageBrush {
        image: ImageData {
            // 2x2 RGBA8: red, green / blue, white
            data: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ]
            .into(),
            format: ImageFormat::Rgba8,
            width: 2,
            height: 2,
            alpha_type: ImageAlphaType::Alpha,
        },
        sampler: ImageSampler {
            quality: ImageQuality::Low,
            ..Default::default()
        },
    };

    let grad = Gradient::new_linear((0.0, 0.0), (28.0, 0.0))
        .with_stops([palette::css::ORANGE_RED, palette::css::DEEP_SKY_BLUE]);

    // Left half: stateful blend changes.
    scene.set_global_alpha(1.0);
    scene.set_blend_mode(Mix::Multiply);
    scene.draw_image(&image, Affine::translate((2.0, 2.0)) * Affine::scale(14.0));
    scene.set_blend_mode(Mix::Screen);
    scene.fill(
        Fill::NonZero,
        Affine::translate((2.0, 34.0)),
        &grad,
        None,
        &rect(0.0, 0.0, 28.0, 26.0),
    );
    scene.set_blend_mode(Mix::Difference);
    scene.set_global_alpha(0.8);
    scene.draw_blurred_rounded_rect(
        Affine::translate((2.0, 2.0)),
        rect(0.0, 0.0, 28.0, 58.0),
        palette::css::LIME,
        4.0,
        3.0,
    );
    scene.set_global_alpha(1.0);
    scene.set_blend_mode(Mix::Normal);

    // Right half: isolated layers per draw.
    let clip = rect(32.0, 0.0, 32.0, 64.0);

    scene.push_layer(Fill::NonZero, Mix::Multiply, 1.0, Affine::IDENTITY, &clip);
    scene.draw_image(&image, Affine::translate((34.0, 2.0)) * Affine::scale(14.0));
    scene.pop_layer();

    scene.push_layer(Fill::NonZero, Mix::Screen, 1.0, Affine::IDENTITY, &clip);
    scene.fill(
        Fill::NonZero,
        Affine::translate((34.0, 34.0)),
        &grad,
        None,
        &rect(0.0, 0.0, 28.0, 26.0),
    );
    scene.pop_layer();

    scene.push_layer(Fill::NonZero, Mix::Difference, 0.8, Affine::IDENTITY, &clip);
    scene.draw_blurred_rounded_rect(
        Affine::translate((34.0, 2.0)),
        rect(0.0, 0.0, 28.0, 58.0),
        palette::css::LIME,
        4.0,
        3.0,
    );
    scene.pop_layer();

    let params = TestParams::new("stateful_vs_layer_blend_switch", 64, 64);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Visual proof that `Compose::Copy` with anti-aliased edges preserves the destination outside the
/// covered area (coverage is applied after compositing).
///
/// This hits `draw_image`, gradients, and `draw_blurred_rounded_rect` under `copy`.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stateful_copy_aa_edges_preserve_background_image_grad_blur() {
    let mut scene = Scene::new();

    draw_checkerboard(&mut scene, 64, 64, 8);

    let image = ImageBrush {
        image: ImageData {
            // 2x2 RGBA8: red, green / blue, white
            data: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ]
            .into(),
            format: ImageFormat::Rgba8,
            width: 2,
            height: 2,
            alpha_type: ImageAlphaType::Alpha,
        },
        sampler: ImageSampler {
            quality: ImageQuality::Low,
            ..Default::default()
        },
    };

    let sweep = Gradient::new_sweep((32.0, 32.0), 0.0_f32, std::f32::consts::TAU).with_stops([
        palette::css::RED,
        palette::css::LIME,
        palette::css::BLUE,
        palette::css::RED,
    ]);

    scene.set_composite(
        BlendMode {
            mix: Mix::Normal,
            compose: Compose::Copy,
        },
        1.0,
    );

    // Rotated image rect (diagonal AA edges).
    let img_t = Affine::translate((32.0, 32.0))
        * Affine::rotate(0.45)
        * Affine::translate((-1.0, -1.0))
        * Affine::scale(12.0);
    scene.draw_image(&image, img_t);

    // Gradient circle (AA boundary).
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        &sweep,
        None,
        &Circle::new((32.0, 32.0), 18.0),
    );

    // Blurred rect covering much of the frame.
    //
    // Use the `_in` variant so the coverage/affected area stays well inside the frame; otherwise
    // `draw_blurred_rounded_rect` inflates the computed shape by ~2.5*std_dev, which can cover the
    // entire image and makes this a poor visual proof for `copy`.
    let blur_clip = Circle::new((32.0, 32.0), 22.0);
    scene.draw_blurred_rounded_rect_in(
        &blur_clip,
        Affine::IDENTITY,
        rect(20.0, 20.0, 24.0, 24.0),
        palette::css::DEEP_SKY_BLUE,
        6.0,
        2.5,
    );

    assert_no_layers(&scene);

    let params = TestParams::new("stateful_copy_aa_edges_preserve_background", 64, 64);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}
