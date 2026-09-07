// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::harness::Registry;
use vello_common::peniko::ImageAlphaType;
use vello_common::pixmap::{PixelMetadata, Pixmap};

const WIDTH: u16 = 1920;
const HEIGHT: u16 = 1080;
const BATCH_SIZE: usize = 1;
const OPAQUE_BLUE: [u8; 4] = [0, 0, 255, 255];
const TRANSLUCENT_BLUE: [u8; 4] = [0, 0, 255, 128];

pub fn register(registry: &mut Registry) {
    let pixel_count = usize::from(WIDTH) * usize::from(HEIGHT);
    register_premultiply(registry, "opaque", OPAQUE_BLUE.repeat(pixel_count));
    register_input(
        registry,
        "translucent",
        TRANSLUCENT_BLUE.repeat(pixel_count),
    );

    registry.extended(|registry| {
        register_input(registry, "interleaved", interleaved_pixels(pixel_count));
        register_input(registry, "mixed_lanes", mixed_lane_pixels(pixel_count));
    });

    let opaque_pixmap = Pixmap::from_parts(
        OPAQUE_BLUE.repeat(pixel_count),
        WIDTH,
        HEIGHT,
        PixelMetadata::new(ImageAlphaType::AlphaPremultiplied, false),
    );
    registry.add("pixmap/rgba_to_rgb/opaque", move |b| {
        b.iter_batched(
            || opaque_pixmap.clone(),
            |pixmap| pixmap.try_take_rgb8(ImageAlphaType::Alpha),
            BATCH_SIZE,
        );
    });
}

fn register_input(registry: &mut Registry, name: &'static str, rgba: Vec<u8>) {
    register_premultiply(registry, name, rgba.clone());

    let pixmap = Pixmap::from_parts(
        rgba,
        WIDTH,
        HEIGHT,
        PixelMetadata::new(ImageAlphaType::Alpha, true),
    );
    registry.add(format!("pixmap/unpremultiply/{name}"), move |b| {
        b.iter_batched(
            || pixmap.clone(),
            |pixmap| pixmap.take_rgba8(ImageAlphaType::Alpha),
            BATCH_SIZE,
        );
    });
}

fn register_premultiply(registry: &mut Registry, name: &'static str, rgba: Vec<u8>) {
    registry.add(format!("pixmap/premultiply/{name}"), move |b| {
        b.iter_batched(
            || rgba.clone(),
            |rgba| {
                Pixmap::from_parts(
                    rgba,
                    WIDTH,
                    HEIGHT,
                    PixelMetadata::new(ImageAlphaType::Alpha, true),
                )
            },
            BATCH_SIZE,
        );
    });
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
