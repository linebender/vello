// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(
    clippy::missing_assert_message,
    clippy::allow_attributes_without_reason
)]

use vello::kurbo::{Affine, Rect};
use vello::peniko::{color::palette, Brush, Color, Format};
use vello::Scene;
use vello_tests::TestParams;

fn simple_square(use_cpu: bool) {
    let mut scene = Scene::new();
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::RED),
        None,
        &Rect::from_center_size((100., 100.), (50., 50.)),
    );
    let params = TestParams {
        use_cpu,
        ..TestParams::new("simple_square", 150, 150)
    };
    let image = vello_tests::render_then_debug_sync(&scene, &params).unwrap();
    assert_eq!(image.format, Format::Rgba8);
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
    assert_eq!(red_count, 50 * 50);
    assert_eq!(black_count, 150 * 150 - 50 * 50);
}

fn empty_scene(use_cpu: bool) {
    let scene = Scene::new();

    // Adding an alpha factor here changes the resulting color *slightly*,
    // presumably due to pre-multiplied alpha.
    // We just assume that alpha scenarios work fine
    let color = palette::css::PLUM;
    let params = TestParams {
        use_cpu,
        base_color: Some(color),
        ..TestParams::new("simple_square", 150, 150)
    };
    let image = vello_tests::render_then_debug_sync(&scene, &params).unwrap();
    assert_eq!(image.format, Format::Rgba8);
    for pixel in image.data.data().chunks_exact(4) {
        let &[r, g, b, a] = pixel else { unreachable!() };
        let image_color = Color::from_rgba8(r, g, b, a);
        if image_color.premultiply().difference(color.premultiply()) > 1e-4 {
            panic!("Got {image_color:?}, expected clear color {color:?}");
        }
    }
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn simple_square_gpu() {
    simple_square(false);
}

#[test]
// The fine shader still requires a GPU, and so we still get a wgpu device
// skip this for now
#[cfg_attr(skip_gpu_tests, ignore)]
fn simple_square_cpu() {
    simple_square(true);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn empty_scene_gpu() {
    empty_scene(false);
}

#[test]
// The fine shader still requires a GPU, and so we still get a wgpu device
// skip this for now
#[cfg_attr(skip_gpu_tests, ignore)]
fn empty_scene_cpu() {
    empty_scene(true);
}
