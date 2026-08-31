// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use criterion::{BatchSize, Criterion, black_box};
use vello_common::peniko::ImageAlphaType;
use vello_common::pixmap::{PixelMetadata, Pixmap};

const WIDTH: u16 = 1920;
const HEIGHT: u16 = 1080;
const OPAQUE_BLUE: [u8; 4] = [0, 0, 255, 255];
const TRANSLUCENT_BLUE: [u8; 4] = [0, 0, 255, 128];

pub fn pixmap(c: &mut Criterion) {
    let pixel_count = usize::from(WIDTH) * usize::from(HEIGHT);
    let mut inputs = vec![
        ("opaque", OPAQUE_BLUE.repeat(pixel_count)),
        ("translucent", TRANSLUCENT_BLUE.repeat(pixel_count)),
    ];

    if crate::EXTENDED {
        inputs.push(("interleaved", interleaved_pixels(pixel_count)));
        inputs.push(("mixed_lanes", mixed_lane_pixels(pixel_count)));
    }

    let mut group = c.benchmark_group("pixmap/premultiply");
    for (name, rgba) in &inputs {
        group.bench_function(*name, |b| {
            b.iter_batched(
                || rgba.clone(),
                |rgba| {
                    black_box(Pixmap::from_parts(
                        rgba,
                        WIDTH,
                        HEIGHT,
                        PixelMetadata::new(ImageAlphaType::Alpha, true),
                    ));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();

    let pixmaps = inputs
        .into_iter()
        .map(|(name, rgba)| {
            (
                name,
                Pixmap::from_parts(
                    rgba,
                    WIDTH,
                    HEIGHT,
                    PixelMetadata::new(ImageAlphaType::Alpha, true),
                ),
            )
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("pixmap/unpremultiply");
    for (name, pixmap) in &pixmaps {
        group.bench_function(*name, |b| {
            b.iter_batched(
                || pixmap.clone(),
                |pixmap| black_box(pixmap.take_unpremultiplied()),
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn interleaved_pixels(pixel_count: usize) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(pixel_count * 4);
    for index in 0..pixel_count {
        let pixel = if (index / 16).is_multiple_of(2) {
            TRANSLUCENT_BLUE
        } else {
            OPAQUE_BLUE
        };
        rgba.extend_from_slice(&pixel);
    }
    rgba
}

fn mixed_lane_pixels(pixel_count: usize) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(pixel_count * 4);
    for index in 0..pixel_count {
        let pixel = if (index / 8).is_multiple_of(2) {
            TRANSLUCENT_BLUE
        } else {
            OPAQUE_BLUE
        };
        rgba.extend_from_slice(&pixel);
    }
    rgba
}
