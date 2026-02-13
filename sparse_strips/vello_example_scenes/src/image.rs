// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image rendering example scene.

use vello_common::color::PremulRgba8;
use vello_common::peniko::ImageFormat;
use vello_common::peniko::ImageSampler;
use vello_common::pixmap::Pixmap;
use vello_common::{
    kurbo::{Affine, Rect},
    paint::{Image, ImageSource},
    peniko::{Extend, ImageQuality},
};

use crate::{ExampleScene, RenderingContext};

/// Image scene state
#[derive(Debug, Default)]
pub struct ImageScene {
    img_sources: Vec<ImageSource>,
}

impl ImageScene {
    /// Create a new image scene
    pub fn new(img_sources: Vec<ImageSource>) -> Self {
        Self { img_sources }
    }
}

impl ExampleScene for ImageScene {
    fn render(&mut self, ctx: &mut impl RenderingContext, root_transform: Affine) {
        let flower_id = self.img_sources[0].clone();
        let cowboy_id = self.img_sources[1].clone();

        let images = [&flower_id, &cowboy_id];
        let tile_w = 320.0;
        let tile_h = 240.0;
        let cols = 100;
        let rows = 100;

        for row in 0..rows {
            for col in 0..cols {
                let img = images[(row + col) % images.len()];
                let x = col as f64 * tile_w;
                let y = row as f64 * tile_h;

                ctx.set_transform(root_transform * Affine::translate((x, y)));
                ctx.set_paint_transform(Affine::IDENTITY);
                ctx.set_paint(Image {
                    image: img.clone(),
                    sampler: ImageSampler {
                        x_extend: Extend::Pad,
                        y_extend: Extend::Pad,
                        quality: ImageQuality::Low,
                        alpha: 1.0,
                    },
                });
                ctx.fill_rect(&Rect::new(0.0, 0.0, tile_w, tile_h));
            }
        }
    }
}

impl ImageScene {
    /// Read the flower image
    pub fn read_flower_image() -> Pixmap {
        let image_data = include_bytes!("../../../examples/assets/splash-flower.jpg");
        Self::read_image(image_data)
    }

    /// Read the flower image
    #[expect(clippy::cast_possible_truncation, reason = "deliberate quantization")]
    pub fn read_image(data: &[u8]) -> Pixmap {
        let image_data = decode_image(data);
        let premul_data: Vec<PremulRgba8> = image_data
            .data
            .chunks_exact(4)
            .map(|rgba| {
                let alpha = u16::from(rgba[3]);
                let premultiply = |component| (alpha * (u16::from(component)) / 255) as u8;
                PremulRgba8 {
                    r: premultiply(rgba[0]),
                    g: premultiply(rgba[1]),
                    b: premultiply(rgba[2]),
                    a: alpha as u8,
                }
            })
            .collect();
        Pixmap::from_parts(
            premul_data,
            image_data.width as u16,
            image_data.height as u16,
        )
    }

    /// Read the cowboy image
    pub fn read_cowboy_image() -> Pixmap {
        let data = include_bytes!("../../vello_sparse_tests/tests/assets/cowboy.png");
        Pixmap::from_png(&data[..]).unwrap()
    }
}

/// Decode an image from a byte slice
pub fn decode_image(data: &[u8]) -> ImageData {
    let image = image::load_from_memory(data).expect("Failed to decode image");
    let width = image.width();
    let height = image.height();
    ImageData {
        data: image.into_rgba8().into_vec(),
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
