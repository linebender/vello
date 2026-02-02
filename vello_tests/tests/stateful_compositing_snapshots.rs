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
use vello::kurbo::{Affine, Rect};
use vello::peniko::{BlendMode, Compose, Fill, Mix, color::palette};
use vello_tests::{TestParams, smoke_snapshot_test_sync};

fn rect(x: f64, y: f64, w: f64, h: f64) -> Rect {
    Rect::from_origin_size((x, y), (w, h))
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

    let mut params = TestParams::new("stateful_compose_xor", 64, 64);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}
