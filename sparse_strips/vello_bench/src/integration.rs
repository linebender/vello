// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integration benchmarks for full rendering pipelines.

use std::sync::Arc;

use criterion::Criterion;
use vello_common::kurbo::{Affine, Rect};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::ImageSampler;
use vello_common::peniko::{Extend, ImageAlphaType, ImageQuality};
use vello_common::pixmap::{PixelMetadata, Pixmap};
use vello_cpu::{RenderContext, Resources};

/// Image scene rendering benchmark.
pub fn images(c: &mut Criterion) {
    if !crate::EXTENDED {
        return;
    }

    let mut g = c.benchmark_group("images");

    let flower_image = load_flower_image();

    const VIEWPORT_WIDTH: u16 = 1280;
    const VIEWPORT_HEIGHT: u16 = 960;

    let ImageSource::Pixmap(ref image_pixmap) = flower_image else {
        panic!("Expected Pixmap");
    };
    let original_width = f64::from(image_pixmap.width());
    let original_height = f64::from(image_pixmap.height());
    let image_count = VIEWPORT_WIDTH / 256;

    g.bench_function("overlapping", |b| {
        let mut renderer = RenderContext::new(VIEWPORT_WIDTH, VIEWPORT_HEIGHT);
        let mut resources = Resources::default();
        let mut pixmap = Pixmap::new(VIEWPORT_WIDTH, VIEWPORT_HEIGHT);

        b.iter(|| {
            renderer.reset();

            for i in (1..=image_count).rev() {
                let width = 256.0 * i as f64;
                let scale = width / original_width;
                let height = original_height * scale;

                renderer.set_paint_transform(Affine::scale(scale));
                renderer.set_paint(Image {
                    image: flower_image.clone(),
                    sampler: ImageSampler {
                        x_extend: Extend::Pad,
                        y_extend: Extend::Pad,
                        quality: ImageQuality::Medium,
                        alpha: 1.0,
                    },
                });
                renderer.fill_rect(&Rect::new(0.0, 0.0, width, height));
            }

            renderer.flush();
            renderer.render(&mut pixmap, &mut resources);
            std::hint::black_box(&pixmap);
        });
    });

    g.finish();
}

fn load_flower_image() -> ImageSource {
    let image_data = include_bytes!("../../../examples/assets/splash-flower.jpg");
    let image = image::load_from_memory(image_data).expect("Failed to decode image");
    let width = image.width();
    let height = image.height();
    let rgba_data = image.into_rgba8().into_vec();

    #[expect(
        clippy::cast_possible_truncation,
        reason = "Image dimensions fit in u16"
    )]
    let pixmap = Pixmap::from_parts(
        rgba_data,
        width as u16,
        height as u16,
        PixelMetadata::new(ImageAlphaType::Alpha, true),
    );

    ImageSource::Pixmap(Arc::new(pixmap))
}
