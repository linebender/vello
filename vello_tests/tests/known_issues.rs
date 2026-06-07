// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reproductions for known bugs, to allow test driven development

// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(
    clippy::missing_assert_message,
    clippy::should_panic_without_expect,
    clippy::allow_attributes_without_reason
)]

use vello::Scene;
use vello::kurbo::{Affine, Rect, Triangle};
use vello::peniko::{Mix, color::palette};
use vello_tests::{TestParams, smoke_snapshot_test_sync, snapshot_test_sync};

/// Test for <https://github.com/linebender/vello/issues/1061>
#[test]
#[should_panic]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_layer_size() {
    let mut scene = Scene::new();
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        vello::peniko::color::AlphaColor::from_rgb8(0, 255, 0),
        None,
        &Rect::from_origin_size((0.0, 0.0), (60., 60.)),
    );
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        vello::peniko::color::AlphaColor::from_rgb8(255, 0, 0),
        None,
        &Rect::from_origin_size((20.0, 20.0), (20., 20.)),
    );
    scene.push_layer(
        vello::peniko::Fill::NonZero,
        vello::peniko::Compose::Clear,
        1.0,
        Affine::IDENTITY,
        &Rect::from_origin_size((20.0, 20.0), (20., 20.)),
    );
    scene.pop_layer();
    let params = TestParams::new("layer_size", 60, 60);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// See <https://github.com/linebender/vello/issues/1198>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn clip_blends() {
    let mut scene = Scene::new();

    scene.fill(
        vello::peniko::Fill::EvenOdd,
        Affine::IDENTITY,
        palette::css::BLUE,
        None,
        &Rect::from_origin_size((0., 0.), (100., 100.)),
    );
    let layer_shape = Triangle::from_coords((50., 0.), (0., 100.), (100., 100.));
    scene.push_clip_layer(vello::peniko::Fill::NonZero, Affine::IDENTITY, &layer_shape);
    scene.push_layer(
        vello::peniko::Fill::NonZero,
        Mix::Multiply,
        1.0,
        Affine::IDENTITY,
        &layer_shape,
    );
    scene.fill(
        vello::peniko::Fill::EvenOdd,
        Affine::IDENTITY,
        palette::css::AQUAMARINE,
        None,
        &Rect::from_origin_size((0., 0.), (100., 100.)),
    );
    scene.pop_layer();
    scene.pop_layer();

    let params = TestParams::new("clip_blends", 100, 100);
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}
