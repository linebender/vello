// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image rendering example scene.
use std::f64::consts::PI;

use vello_common::color::PremulRgba8;
use vello_common::kurbo::{BezPath, Point, Shape, Vec2};
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
        let splash_flower_id = self.img_sources[0].clone();
        let cowboy_id = self.img_sources[1].clone();

        ctx.set_transform(
            root_transform
                * Affine::translate((200.0, 50.0))
                * Affine::rotate(PI / 4.0)
                * Affine::scale(0.5),
        );
        ctx.set_paint_transform(Affine::translate((0.0, 0.0)));
        ctx.set_paint(Image {
            image: splash_flower_id.clone(),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        });
        ctx.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));

        ctx.set_transform(
            root_transform
                * Affine::translate((500.0, 50.0))
                * Affine::skew(0.5, 0.5)
                * Affine::scale(0.5),
        );
        ctx.set_paint(Image {
            image: splash_flower_id.clone(),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        });
        ctx.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));

        ctx.set_transform(root_transform * Affine::translate((0.0, 500.0)));
        ctx.set_paint(Image {
            image: splash_flower_id.clone(),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        });
        ctx.fill_path(&heart_shape());

        ctx.set_transform(root_transform * Affine::translate((400.0, 480.0)));
        ctx.push_clip_layer(&circular_star(Point::new(300.0, 220.0), 5, 100.0, 150.0).to_path(0.1));
        ctx.set_paint_transform(Affine::IDENTITY);
        ctx.set_paint(Image {
            image: splash_flower_id.clone(),
            sampler: ImageSampler {
                x_extend: Extend::Repeat,
                y_extend: Extend::Repeat,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        });
        ctx.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));
        ctx.pop_layer();

        ctx.set_transform(root_transform * Affine::translate((1000.0, 50.0)));
        ctx.set_paint_transform(Affine::scale(0.25));
        ctx.set_paint(Image {
            image: splash_flower_id.clone(),
            sampler: ImageSampler {
                x_extend: Extend::Repeat,
                y_extend: Extend::Repeat,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        });
        ctx.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));

        ctx.set_transform(root_transform * Affine::translate((1000.0, 600.0)));
        ctx.set_paint_transform(Affine::scale(0.25));
        ctx.set_paint(Image {
            image: splash_flower_id.clone(),
            sampler: ImageSampler {
                x_extend: Extend::Reflect,
                y_extend: Extend::Repeat,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        });
        ctx.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));

        ctx.set_transform(root_transform * Affine::translate((1000.0, 1200.0)));
        ctx.set_paint_transform(Affine::scale(0.25));
        ctx.set_paint(Image {
            image: splash_flower_id.clone(),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Repeat,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        });
        ctx.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));

        ctx.set_transform(root_transform * Affine::translate((100.0, 1000.0)));
        ctx.set_paint_transform(Affine::IDENTITY);
        ctx.set_paint(Image {
            image: cowboy_id,
            sampler: ImageSampler {
                x_extend: Extend::Repeat,
                y_extend: Extend::Repeat,
                quality: ImageQuality::High,
                alpha: 1.0,
            },
        });
        ctx.fill_rect(&Rect::new(0.0, 0.0, 800.0, 160.0));

        ctx.set_transform(
            root_transform
                * Affine::translate((200.0, 1200.0))
                * Affine::rotate(PI / 4.0)
                * Affine::scale(0.5),
        );
        ctx.set_paint_transform(Affine::scale(0.25));
        ctx.set_paint(Image {
            image: splash_flower_id.clone(),
            sampler: ImageSampler {
                x_extend: Extend::Repeat,
                y_extend: Extend::Repeat,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        });
        ctx.fill_rect(&Rect::new(0.0, 0.0, 640.0, 480.0));
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

fn circular_star(center: Point, n: usize, inner: f64, outer: f64) -> BezPath {
    let mut path = BezPath::new();
    let start_angle = -std::f64::consts::FRAC_PI_2;
    path.move_to(center + outer * Vec2::from_angle(start_angle));
    for i in 1..n * 2 {
        let th = start_angle + i as f64 * PI / n as f64;
        let r = if i % 2 == 0 { outer } else { inner };
        path.line_to(center + r * Vec2::from_angle(th));
    }
    path.close_path();
    path
}
