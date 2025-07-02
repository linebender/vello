// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple example scene with an image.

use std::sync::Arc;
use vello_common::kurbo::{Affine, Rect};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::{Extend, ImageQuality};
use vello_common::pixmap::Pixmap;
use vello_cpu::RenderContext;

use crate::scenes::ExampleScene;

/// Image scene state
#[derive(Debug)]
pub(crate) struct ImageScene {
    cowboy_image: Image,
}

impl ExampleScene for ImageScene {
    fn render(&mut self, ctx: &mut RenderContext, root_transform: Affine) {
        render(ctx, root_transform, &self.cowboy_image);
    }
}

impl ImageScene {
    /// Create a new `ImageScene`
    pub(crate) fn new() -> Self {
        let data = include_bytes!("../../../../../vello_sparse_tests/tests/assets/cowboy.png");
        let pixmap = Pixmap::from_png(&data[..]).unwrap();

        Self {
            cowboy_image: Image {
                source: ImageSource::Pixmap(Arc::new(pixmap)),
                quality: ImageQuality::Medium,
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
            },
        }
    }
}

impl Default for ImageScene {
    fn default() -> Self {
        Self::new()
    }
}

/// Draws a simple scene with shapes
pub(crate) fn render(ctx: &mut RenderContext, root_transform: Affine, image: &Image) {
    // Use a combined transform that includes the root transform
    let scene_transform = Affine::scale(5.0);
    ctx.set_transform(root_transform * scene_transform);

    ctx.set_paint(image.clone());
    ctx.fill_rect(&Rect::new(0., 0., 80., 80.));
}
