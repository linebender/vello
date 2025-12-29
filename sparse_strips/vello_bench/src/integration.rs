// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Integration benchmarks for full rendering pipelines.

use std::sync::Arc;

use criterion::Criterion;
use vello_common::kurbo::{Affine, Rect};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::ImageSampler;
use vello_common::peniko::{Extend, ImageQuality};
use vello_common::pixmap::Pixmap;
use vello_cpu::RenderContext;
use vello_cpu::color::AlphaColor;

/// Image scene rendering benchmark.
pub fn images(c: &mut Criterion) {
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
            renderer.render_to_pixmap(&mut pixmap);
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

    let mut may_have_opacities = false;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Image dimensions fit in u16"
    )]
    let pixmap = Pixmap::from_parts_with_opacity(
        rgba_data
            .chunks_exact(4)
            .map(|rgba| {
                let alpha = rgba[3];
                if alpha != 255 {
                    may_have_opacities = true;
                }
                AlphaColor::from_rgba8(rgba[0], rgba[1], rgba[2], alpha)
                    .premultiply()
                    .to_rgba8()
            })
            .collect(),
        width as u16,
        height as u16,
        may_have_opacities,
    );

    ImageSource::Pixmap(Arc::new(pixmap))
}
