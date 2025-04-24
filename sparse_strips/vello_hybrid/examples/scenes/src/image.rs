// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image rendering example scene.

use std::f64::consts::PI;
use std::sync::Arc;

use vello_common::kurbo::BezPath;
use vello_common::peniko::ImageFormat;
use vello_common::pixmap::Pixmap;
use vello_common::{
    kurbo::{Affine, Rect},
    paint::Image,
    peniko::{Extend, ImageQuality},
};
use vello_hybrid::Scene;

use crate::ExampleScene;

/// Image scene state
#[derive(Debug)]
pub struct ImageScene {
    image_data: Arc<Pixmap>,
}

impl ExampleScene for ImageScene {
    fn render(&mut self, scene: &mut Scene, root_transform: Affine) {
        scene.set_transform(
            root_transform
                * Affine::translate((200.0, 50.0))
                * Affine::rotate(PI / 4.0)
                * Affine::scale(0.5),
        );
        scene.set_paint(Image {
            pixmap: self.image_data.clone(),
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality: ImageQuality::Low,
            transform: Affine::translate((0.0, 0.0)),
        });
        scene.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));

        scene.set_transform(
            root_transform
                * Affine::translate((500.0, 50.0))
                * Affine::skew(0.5, 0.5)
                * Affine::scale(0.5),
        );
        scene.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));

        scene.set_transform(root_transform * Affine::translate((0.0, 600.0)));
        let path = heart_shape();
        scene.fill_path(&path);

        scene.set_transform(
            root_transform
                * Affine::translate((700.0, 600.0))
                * Affine::rotate(PI / 4.0)
                * Affine::scale(0.5),
        );
        scene.set_paint(Image {
            pixmap: self.image_data.clone(),
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            transform: Affine::scale(0.25),
        });
        scene.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));

        scene.set_transform(root_transform * Affine::translate((1000.0, 50.0)));
        scene.set_paint(Image {
            pixmap: self.image_data.clone(),
            x_extend: Extend::Repeat,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            transform: Affine::scale(0.25),
        });
        scene.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));

        scene.set_transform(root_transform * Affine::translate((1000.0, 600.0)));
        scene.set_paint(Image {
            pixmap: self.image_data.clone(),
            x_extend: Extend::Reflect,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            transform: Affine::scale(0.25),
        });
        scene.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));

        scene.set_transform(root_transform * Affine::translate((1000.0, 1200.0)));
        scene.set_paint(Image {
            pixmap: self.image_data.clone(),
            x_extend: Extend::Pad,
            y_extend: Extend::Repeat,
            quality: ImageQuality::Low,
            transform: Affine::scale(0.25),
        });
        scene.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));
    }
}

impl ImageScene {
    /// Create a new `ImageScene`
    pub fn new() -> Self {
        let image_data = ImageScene::read_flower_image();
        let mut pixmap = Pixmap::new(image_data.width as u16, image_data.height as u16);
        pixmap.buf = image_data.data;
        Self {
            image_data: Arc::new(pixmap),
        }
    }

    /// Read the flower image
    pub fn read_flower_image() -> ImageData {
        let image_path = include_bytes!("../../../../../examples/assets/splash-flower.jpg");
        decode_image(image_path)
    }
}

/// Decode an image from a byte slice
pub fn decode_image(data: &[u8]) -> ImageData {
    let image = image::ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();
    let width = image.width();
    let height = image.height();
    let data = image.into_rgba8().into_vec();
    ImageData {
        data,
        format: ImageFormat::Rgba8,
        width,
        height,
    }
}

/// Image data
#[derive(Clone, PartialEq, Debug)]
pub struct ImageData {
    /// Blob containing the image data.
    pub data: Vec<u8>,
    /// Pixel format of the image.
    pub format: ImageFormat,
    /// Width of the image.
    pub width: u32,
    /// Height of the image.
    pub height: u32,
}

fn heart_shape() -> BezPath {
    let mut path = BezPath::new();
    path.move_to((320.0, 100.0));
    // Left top curve of the heart
    path.curve_to((160.0, 20.0), (80.0, 120.0), (320.0, 380.0));
    // Right top curve of the heart
    path.curve_to((560.0, 120.0), (480.0, 20.0), (320.0, 100.0));
    path.close_path();
    path
}
