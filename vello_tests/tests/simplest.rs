// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello::{
    kurbo::{Affine, Rect},
    peniko::{Brush, Color},
    Scene,
};
use vello_tests::{snapshot_test_sync, TestParams};

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn simple_square_gpu() {
    filled_square(false);
}

#[test]
// The fine shader still requires a GPU, and so we still get a wgpu device
// skip this for now
#[cfg_attr(skip_gpu_tests, ignore)]
fn simple_square_cpu() {
    filled_square(true);
}

fn filled_square(use_cpu: bool) {
    let mut scene = Scene::new();
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Brush::Solid(Color::BLUE),
        None,
        &Rect::from_center_size((10., 10.), (6., 6.)),
    );
    let params = TestParams {
        use_cpu,
        ..TestParams::new("filled_square", 20, 20)
    };
    match snapshot_test_sync(scene, &params)
        .and_then(|mut snapshot| snapshot.assert_mean_less_than(0.01))
    {
        Ok(()) => (),
        Err(e) => panic!("{:#}", e),
    }
}
