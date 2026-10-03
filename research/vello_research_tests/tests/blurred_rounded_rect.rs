// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests for the blurred rounded rectangle paint.
//!
//! When `invert` is `true`, the complement (`1 - alpha`) of the blur coverage is painted: the
//! brush is fully opaque outside the blurred rounded rectangle and fades to transparent inside it.
//! This is used to implement inset box shadows.
//!
//! These cases mirror the blurred rounded rectangle tests in the `vello_sparse_tests` crate.

use vello::Scene;
use vello::kurbo::{Affine, BezPath, Circle, Point, Rect, RoundedRect, Shape, Vec2};
use vello::peniko::Fill;
use vello::peniko::color::palette;
use vello_research_tests::{TestParams, snapshot_test_sync};

fn snapshot_blurred_rounded_rect(
    name: &str,
    radius: f64,
    std_dev: f64,
    affine: Affine,
    use_cpu: bool,
) {
    let mut scene = Scene::new();
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    scene.draw_blurred_rounded_rect(affine, rect, palette::css::REBECCA_PURPLE, radius, std_dev);
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
fn blurred_rounded_rect_zero_gpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_zero",
        0.0,
        0.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_zero_cpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_zero",
        0.0,
        0.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_zero_with_radius_gpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_zero_with_radius",
        10.0,
        0.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_zero_with_radius_cpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_zero_with_radius",
        10.0,
        0.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_none_gpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_none",
        0.0,
        0.1,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_none_cpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_none",
        0.0,
        0.1,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_small_std_dev_gpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_small_std_dev",
        0.0,
        5.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_small_std_dev_cpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_small_std_dev",
        0.0,
        5.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_medium_std_dev_gpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_medium_std_dev",
        0.0,
        10.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_medium_std_dev_cpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_medium_std_dev",
        0.0,
        10.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_large_std_dev_gpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_large_std_dev",
        0.0,
        20.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_large_std_dev_cpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_large_std_dev",
        0.0,
        20.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_with_radius_gpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_with_radius",
        10.0,
        10.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_with_radius_cpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_with_radius",
        10.0,
        10.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_with_large_radius_gpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_with_large_radius",
        30.0,
        10.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_with_large_radius_cpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_with_large_radius",
        30.0,
        10.0,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_with_transform_gpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_with_transform",
        10.0,
        10.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_with_transform_cpu() {
    snapshot_blurred_rounded_rect(
        "blurred_rounded_rect_with_transform",
        10.0,
        10.0,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
        true,
    );
}

fn snapshot_inverse_blurred_rounded_rect(
    name: &str,
    radius: f64,
    std_dev: f64,
    affine: Affine,
    use_cpu: bool,
) {
    let mut scene = Scene::new();
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    // Extend the shape to the blur kernel, so the fully covered area outside of `rect` is painted.
    let kernel_size = 2.5 * std_dev;
    scene.draw_blurred_rounded_rect_in(
        &rect.inflate(kernel_size, kernel_size),
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

fn snapshot_blurred_rounded_rect_in(
    name: &str,
    path: &impl Shape,
    radius: f64,
    std_dev: f64,
    invert: bool,
    affine: Affine,
    use_cpu: bool,
) {
    let mut scene = Scene::new();
    let rect = Rect::new(20.0, 20.0, 80.0, 80.0);
    scene.draw_blurred_rounded_rect_in(
        path,
        affine,
        rect,
        palette::css::REBECCA_PURPLE,
        radius,
        std_dev,
        invert,
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
fn blurred_rounded_rect_in_circle_gpu() {
    let path = Circle::new(Point::new(50.0, 50.0), 45.0);
    snapshot_blurred_rounded_rect_in(
        "blurred_rounded_rect_in_circle",
        &path,
        10.0,
        10.0,
        false,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_in_circle_cpu() {
    let path = Circle::new(Point::new(50.0, 50.0), 45.0);
    snapshot_blurred_rounded_rect_in(
        "blurred_rounded_rect_in_circle",
        &path,
        10.0,
        10.0,
        false,
        Affine::IDENTITY,
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_in_circle_with_transform_gpu() {
    let path = Circle::new(Point::new(50.0, 50.0), 45.0);
    snapshot_blurred_rounded_rect_in(
        "blurred_rounded_rect_in_circle_with_transform",
        &path,
        10.0,
        10.0,
        false,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_in_circle_with_transform_cpu() {
    let path = Circle::new(Point::new(50.0, 50.0), 45.0);
    snapshot_blurred_rounded_rect_in(
        "blurred_rounded_rect_in_circle_with_transform",
        &path,
        10.0,
        10.0,
        false,
        Affine::rotate_about(45.0_f64.to_radians(), Point::new(50.0, 50.0)),
        true,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_in_rect_gpu() {
    let path = Rect::new(20.0, 20.0, 80.0, 80.0);
    snapshot_blurred_rounded_rect_in(
        "inverse_blurred_rounded_rect_in_rect",
        &path,
        10.0,
        10.0,
        true,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inverse_blurred_rounded_rect_in_rect_cpu() {
    let path = Rect::new(20.0, 20.0, 80.0, 80.0);
    snapshot_blurred_rounded_rect_in(
        "inverse_blurred_rounded_rect_in_rect",
        &path,
        10.0,
        10.0,
        true,
        Affine::IDENTITY,
        true,
    );
}

/// Two overlapping circles with opposite winding directions.
///
/// `draw_blurred_rounded_rect_in` always fills `shape` with the non-zero fill rule, so the
/// overlapping region of the two circles (which has a winding number of zero) is not painted.
/// This gives the same coverage as the even-odd case in the `vello_tests` crate.
fn opposing_circles() -> BezPath {
    let mut path = Circle::new(Point::new(35.0, 50.0), 40.0).to_path(0.1);
    path.extend(
        Circle::new(Point::new(65.0, 50.0), 40.0)
            .to_path(0.1)
            .reverse_subpaths(),
    );
    path
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_in_even_odd_gpu() {
    snapshot_blurred_rounded_rect_in(
        "blurred_rounded_rect_in_even_odd",
        &opposing_circles(),
        10.0,
        10.0,
        false,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn blurred_rounded_rect_in_even_odd_cpu() {
    snapshot_blurred_rounded_rect_in(
        "blurred_rounded_rect_in_even_odd",
        &opposing_circles(),
        10.0,
        10.0,
        false,
        Affine::IDENTITY,
        true,
    );
}

/// Emulate a CSS inset box shadow: the inverse blurred rounded rectangle is shrunk by the spread
/// and offset relative to the border box it is painted into, and clipped to that border box.
fn snapshot_inset_box_shadow(
    name: &str,
    offset: Vec2,
    spread: f64,
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
        border_box.inflate(-spread, -spread) + offset,
        palette::css::REBECCA_PURPLE,
        (radius - spread).max(0.0),
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
        0.0,
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
        0.0,
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
        0.0,
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
        0.0,
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
        0.0,
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
        0.0,
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
        0.0,
        Affine::IDENTITY,
        true,
    );
}

/// A spread shrinks the shadow rectangle (and its corner radius), so the shadow also covers the
/// bottom and right of the border box despite the offset.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_with_spread_gpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_with_spread",
        Vec2::new(4.0, 4.0),
        8.0,
        10.0,
        4.0,
        Affine::IDENTITY,
        false,
    );
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn inset_box_shadow_with_spread_cpu() {
    snapshot_inset_box_shadow(
        "inset_box_shadow_with_spread",
        Vec2::new(4.0, 4.0),
        8.0,
        10.0,
        4.0,
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
        0.0,
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
        0.0,
        10.0,
        6.0,
        Affine::rotate_about(20.0_f64.to_radians(), Point::new(50.0, 50.0)),
        true,
    );
}
