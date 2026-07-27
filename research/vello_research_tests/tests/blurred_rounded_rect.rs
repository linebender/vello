// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests for the inverse (`invert`) blurred rounded rectangle paint.
//!
//! When `invert` is `true`, the complement (`1 - alpha`) of the blur coverage is painted: the
//! brush is fully opaque outside the blurred rounded rectangle and fades to transparent inside it.
//! This is used to implement inset box shadows.
//!
//! These cases mirror the blurred rounded rectangle tests in the `vello_sparse_tests` crate.

use vello::Scene;
use vello::kurbo::{Affine, Point, Rect, RoundedRect, Vec2};
use vello::peniko::Fill;
use vello::peniko::color::palette;
use vello_tests::{TestParams, snapshot_test_sync};

fn snapshot_inverse_blurred_rounded_rect(
    name: &str,
    radius: f64,
    std_dev: f64,
    affine: Affine,
    use_cpu: bool,
) {
    let mut scene = Scene::new();
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    scene.draw_blurred_rounded_rect_in(
        &rect,
        affine,
        rect,
        palette::css::REBECCA_PURPLE,
        radius,
        std_dev,
        true,
    );
    let params = TestParams {
        use_cpu,
        base_color: Some(palette::css::WHITE),
        ..TestParams::new(name, 100, 100)
    };
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_small_std_dev_gpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_small_std_dev",
        0.0,
        5.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_small_std_dev_cpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_small_std_dev",
        0.0,
        5.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_medium_std_dev_gpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_medium_std_dev",
        0.0,
        10.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_medium_std_dev_cpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_medium_std_dev",
        0.0,
        10.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_large_std_dev_gpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_large_std_dev",
        0.0,
        20.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_large_std_dev_cpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_large_std_dev",
        0.0,
        20.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_with_radius_gpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_with_radius",
        10.0,
        10.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_with_radius_cpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_with_radius",
        10.0,
        10.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_with_large_radius_gpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_with_large_radius",
        30.0,
        10.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_with_large_radius_cpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_with_large_radius",
        30.0,
        10.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_with_transform_gpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_with_transform",
        10.0,
        10.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_with_transform_cpu() {
    snapshot_inverse_blurred_rounded_rect(
        "inverse_blurred_rounded_rect_with_transform",
        10.0,
        10.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
        true,
    );
}

/// Emulate a CSS inset box shadow: the inverse blurred rounded rectangle is offset relative to
/// the border box it is painted into, and clipped to that border box.
fn snapshot_inset_box_shadow(
    name: &str,
    offset: Vec2,
    radius: f64,
    std_dev: f64,
    affine: Affine,
    use_cpu: bool,
) {
    let mut scene = Scene::new();
    let border_box = Rect::new(20.0, 20.0, 80.0, 80.0);
    let shape = RoundedRect::from_rect(border_box, radius);

    scene.fill(
        Fill::NonZero,
        affine,
        palette::css::PALE_GOLDENROD,
        None,
        &shape,
    );
    scene.draw_blurred_rounded_rect_in(
        &shape,
        affine,
        border_box + offset,
        palette::css::REBECCA_PURPLE,
        radius,
        std_dev,
        true,
    );

    let params = TestParams {
        use_cpu,
        base_color: Some(palette::css::WHITE),
        ..TestParams::new(name, 100, 100)
    };
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.01);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_offset_down_right_gpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_offset_down_right",
        Vec2::new(8.0, 8.0),
        10.0,
        6.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_offset_down_right_cpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_offset_down_right",
        Vec2::new(8.0, 8.0),
        10.0,
        6.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_offset_up_left_gpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_offset_up_left",
        Vec2::new(-8.0, -8.0),
        10.0,
        6.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_offset_up_left_cpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_offset_up_left",
        Vec2::new(-8.0, -8.0),
        10.0,
        6.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_offset_horizontal_gpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_offset_horizontal",
        Vec2::new(12.0, 0.0),
        10.0,
        4.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_offset_horizontal_cpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_offset_horizontal",
        Vec2::new(12.0, 0.0),
        10.0,
        4.0,
        Affine::IDENTITY,
        true,
    );
}

/// An offset shadow larger than the blur, so the shadow has a hard edge and only covers the
/// top and left of the border box.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_offset_without_blur_gpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_offset_without_blur",
        Vec2::new(15.0, 15.0),
        0.0,
        0.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_offset_without_blur_cpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_offset_without_blur",
        Vec2::new(15.0, 15.0),
        0.0,
        0.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_offset_with_transform_gpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_offset_with_transform",
        Vec2::new(8.0, 8.0),
        10.0,
        6.0,
        Affine::rotate_about(20.0_f64.to_radians(), Point::new(50.0, 50.0)),
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_offset_with_transform_cpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_offset_with_transform",
        Vec2::new(8.0, 8.0),
        10.0,
        6.0,
        Affine::rotate_about(20.0_f64.to_radians(), Point::new(50.0, 50.0)),
        true,
    );
}
