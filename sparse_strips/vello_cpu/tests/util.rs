// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "we don't need docs for testing")]
#![allow(
    unused,
    reason = "cargo reports the functions/variables here are unused
when running `cargo test` because they are not be used in every test module."
)]
#![allow(clippy::cast_possible_truncation, reason = "not critical for testing")]

use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder, Rgba, RgbaImage, load_from_memory};
use std::cmp::max;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::LazyLock;
use vello_common::color::palette;
use vello_common::kurbo::{Rect, Shape};
use vello_common::pixmap::Pixmap;
use vello_cpu::RenderContext;

static REFS_PATH: LazyLock<PathBuf> =
    LazyLock::new(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("snapshots"));
static DIFFS_PATH: LazyLock<PathBuf> =
    LazyLock::new(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("diffs"));

pub(crate) fn get_ctx(width: u16, height: u16, transparent: bool) -> RenderContext {
    let mut ctx = RenderContext::new(width, height);
    if !transparent {
        let path = Rect::new(0.0, 0.0, width as f64, height as f64).to_path(0.1);

        ctx.set_paint(palette::css::WHITE.into());
        ctx.fill_path(&path);
    }

    ctx
}

pub(crate) fn render_pixmap(ctx: &RenderContext) -> Pixmap {
    let mut pixmap = Pixmap::new(ctx.width(), ctx.height());
    ctx.render_to_pixmap(&mut pixmap);

    pixmap
}

pub(crate) fn check_ref(ctx: &RenderContext, name: &str) {
    let mut pixmap = render_pixmap(ctx);
    pixmap.unpremultiply();

    let encoded_image = {
        let mut png_data = Vec::new();
        let cursor = Cursor::new(&mut png_data);
        let encoder = PngEncoder::new(cursor);
        encoder
            .write_image(
                pixmap.data(),
                ctx.width() as u32,
                ctx.height() as u32,
                ExtendedColorType::Rgba8,
            )
            .expect("Failed to encode image");
        png_data
    };

    let ref_path = REFS_PATH.join(format!("{}.png", name));

    let write_ref_image = || {
        let optimized =
            oxipng::optimize_from_memory(&encoded_image, &oxipng::Options::max_compression())
                .unwrap();
        std::fs::write(&ref_path, optimized).unwrap();
    };

    if !ref_path.exists() {
        write_ref_image();
        panic!("new reference image was created");
    }

    let ref_image = load_from_memory(&std::fs::read(&ref_path).unwrap())
        .unwrap()
        .into_rgba8();
    let actual = load_from_memory(&encoded_image).unwrap().into_rgba8();

    let diff_image = get_diff(&ref_image, &actual);

    if let Some(diff_image) = diff_image {
        if std::env::var("REPLACE").is_ok() {
            write_ref_image();
            panic!("test was replaced");
        }

        if !DIFFS_PATH.exists() {
            let _ = std::fs::create_dir_all(DIFFS_PATH.as_path());
        }

        let diff_path = DIFFS_PATH.join(format!("{}.png", name));
        diff_image
            .save_with_format(&diff_path, image::ImageFormat::Png)
            .unwrap();

        panic!("test didnt match reference image");
    }
}

fn get_diff(expected_image: &RgbaImage, actual_image: &RgbaImage) -> Option<RgbaImage> {
    let width = max(expected_image.width(), actual_image.width());
    let height = max(expected_image.height(), actual_image.height());

    let mut diff_image = RgbaImage::new(width * 3, height);

    let mut pixel_diff = 0;

    for x in 0..width {
        for y in 0..height {
            let actual_pixel = actual_image.get_pixel_checked(x, y);
            let expected_pixel = expected_image.get_pixel_checked(x, y);

            match (actual_pixel, expected_pixel) {
                (Some(actual), Some(expected)) => {
                    diff_image.put_pixel(x, y, *expected);
                    diff_image.put_pixel(x + 2 * width, y, *actual);
                    if is_pix_diff(expected, actual) {
                        pixel_diff += 1;
                        diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                    } else {
                        diff_image.put_pixel(x + width, y, Rgba([0, 0, 0, 255]));
                    }
                }
                (Some(actual), None) => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x + 2 * width, y, *actual);
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                }
                (None, Some(expected)) => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x, y, *expected);
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                }
                _ => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x, y, Rgba([255, 0, 0, 255]));
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                }
            }
        }
    }

    if pixel_diff > 0 {
        Some(diff_image)
    } else {
        None
    }
}

fn is_pix_diff(pixel1: &Rgba<u8>, pixel2: &Rgba<u8>) -> bool {
    if pixel1.0[3] == 0 && pixel2.0[3] == 0 {
        return false;
    }

    pixel1 != pixel2
}
