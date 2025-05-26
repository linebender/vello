// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Simple testing of vello scenes.
// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(
    clippy::missing_assert_message,
    clippy::allow_attributes_without_reason
)]
use vello::Scene;
use vello::kurbo::{Affine, Ellipse, Rect};
use vello::peniko::{Brush, color::palette};

#[test]
fn simple_square() {
    let mut scene = Scene::new();
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::RED),
        None,
        &Rect::from_center_size((-25., 0.), (50., 50.)),
    );
    let actual = scene.compute_bb();
    let expected = Rect::new(-50.0, -25.0, 0.0, 25.0);
    assert_eq!(expected, actual);
}

#[test]
fn simple_ellipse() {
    let mut scene = Scene::new();
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::RED),
        None,
        &Ellipse::new((0.0, 0.0), (50.0, 20.0), 0.0),
    );
    let actual = scene.compute_bb();
    let expected = Rect::new(-50.0, 50.0, -20.0, 20.0);
    assert_eq!(expected, actual);
}

#[test]
fn rotated_ellipse() {
    let mut scene = Scene::new();
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(palette::css::RED),
        None,
        &Ellipse::new((0.0, 0.0), (50.0, 20.0), 90.0),
    );
    let actual = scene.compute_bb();
    let expected = Rect::new(-20.0, -50.0, 20.0, 50.0);
    assert_eq!(expected, actual);
}

#[test]
fn empty_scene() {
    let scene = Scene::new();
    let actual = scene.compute_bb();
    let expected = Rect::ZERO;
    assert_eq!(expected, actual);
}
