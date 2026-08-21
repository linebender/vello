// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use criterion::{BatchSize, Criterion, black_box};
use vello_common::peniko::ImageAlphaType;
use vello_common::pixmap::{PixelMetadata, Pixmap};

const WIDTH: u16 = 1920;
const HEIGHT: u16 = 1080;

pub fn pixmap(c: &mut Criterion) {
    let pixel_count = usize::from(WIDTH) * usize::from(HEIGHT);
    let mut rgba = Vec::with_capacity(pixel_count * 4);
    for index in 0..pixel_count {
        let value = index.to_le_bytes()[0];
        let alpha = if index.is_multiple_of(8) { 128 } else { 255 };
        rgba.extend_from_slice(&[value, 255 - value, value / 2, alpha]);
    }

    let mut group = c.benchmark_group("pixmap");
    group.bench_function("new", |b| {
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
    let pixmap = Pixmap::from_parts(
        rgba,
        WIDTH,
        HEIGHT,
        PixelMetadata::new(ImageAlphaType::Alpha, true),
    );
    group.bench_function("take_unpremultiplied", |b| {
        b.iter_batched(
            || pixmap.clone(),
            |pixmap| black_box(pixmap.take(ImageAlphaType::Alpha)),
            BatchSize::LargeInput,
        );
    });
    group.finish();
}
