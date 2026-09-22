// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pixel comparison shared by the snapshot suite and the fuzz targets.

use image::{Rgba, RgbaImage};
use serde::Serializer;
use std::cmp::max;

/// Aggregate diff report with statistics and individual pixel differences.
#[derive(Debug, serde::Serialize)]
pub struct DiffReport {
    /// Total number of pixels that differ.
    pub pixel_count: usize,
    /// Maximum absolute difference per channel [R, G, B, A].
    pub max_difference: [i16; 4],
    /// Individual pixel differences.
    pub pixels: Vec<PixelDiff>,
}

impl DiffReport {
    pub fn new(pixels: Vec<PixelDiff>) -> Self {
        let max_difference = pixels.iter().fold([0; 4], |mut max, p| {
            for (m, d) in max.iter_mut().zip(&p.difference) {
                *m = (*m).max(d.abs());
            }
            max
        });
        Self {
            pixel_count: pixels.len(),
            max_difference,
            pixels,
        }
    }
}

/// Represents a single pixel difference between reference and actual images.
#[derive(Debug, serde::Serialize)]
pub struct PixelDiff {
    /// The x coordinate of the differing pixel.
    pub x: u32,
    /// The y coordinate of the differing pixel.
    pub y: u32,
    /// The RGBA values from the target image (i.e. the saved reference).
    // Note that this field name is chosen to be the same length as `actual`
    // That makes it easier to compare the results in the printed JSON.
    #[serde(serialize_with = "hex_string")]
    pub target: [u8; 4],
    /// The RGBA values from the actual image.
    #[serde(serialize_with = "hex_string")]
    pub actual: [u8; 4],
    /// Per-channel difference (actual - target) as signed values.
    pub difference: [i16; 4],
}

/// Serialize a [`[u8; 4]`](primitive@core::array) pixel as a hex string through serde.
///
/// E.g. `[0, 255, 0, 255]` becomes #00ff00. Notice that the alpha is not included if fully opaque.
fn hex_string<S>([r, g, b, a]: &[u8; 4], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if *a != 255 {
        serializer.collect_str(&format_args!("#{r:02x}{g:02x}{b:02x}{a:02x}"))
    } else {
        serializer.collect_str(&format_args!("#{r:02x}{g:02x}{b:02x}"))
    }
}

/// Compares two images and, if more than `diff_pixels` pixels differ by more than `threshold`
/// in any colour channel, returns an `expected | diff | actual` image and the differing pixels.
pub fn get_diff(
    expected_image: &RgbaImage,
    actual_image: &RgbaImage,
    threshold: u8,
    diff_pixels: u32,
) -> Option<(RgbaImage, Vec<PixelDiff>)> {
    let width = max(expected_image.width(), actual_image.width());
    let height = max(expected_image.height(), actual_image.height());

    let mut diff_image = RgbaImage::new(width * 3, height);
    let mut diff_data = Vec::new();

    let mut pixel_diff = 0;

    for x in 0..width {
        for y in 0..height {
            let actual_pixel = actual_image.get_pixel_checked(x, y);
            let expected_pixel = expected_image.get_pixel_checked(x, y);

            match (actual_pixel, expected_pixel) {
                (Some(actual), Some(expected)) => {
                    diff_image.put_pixel(x, y, *expected);
                    diff_image.put_pixel(x + 2 * width, y, *actual);
                    if is_pix_diff(expected, actual, threshold) {
                        pixel_diff += 1;
                        diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                        diff_data.push(PixelDiff {
                            x,
                            y,
                            target: expected.0,
                            actual: actual.0,
                            difference: [
                                i16::from(actual.0[0]) - i16::from(expected.0[0]),
                                i16::from(actual.0[1]) - i16::from(expected.0[1]),
                                i16::from(actual.0[2]) - i16::from(expected.0[2]),
                                i16::from(actual.0[3]) - i16::from(expected.0[3]),
                            ],
                        });
                    } else {
                        diff_image.put_pixel(x + width, y, Rgba([0, 0, 0, 255]));
                    }
                }
                (Some(actual), None) => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x + 2 * width, y, *actual);
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                    diff_data.push(PixelDiff {
                        x,
                        y,
                        target: [0, 0, 0, 0],
                        actual: actual.0,
                        difference: [
                            i16::from(actual.0[0]),
                            i16::from(actual.0[1]),
                            i16::from(actual.0[2]),
                            i16::from(actual.0[3]),
                        ],
                    });
                }
                (None, Some(expected)) => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x, y, *expected);
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                    diff_data.push(PixelDiff {
                        x,
                        y,
                        target: expected.0,
                        actual: [0, 0, 0, 0],
                        difference: [
                            -i16::from(expected.0[0]),
                            -i16::from(expected.0[1]),
                            -i16::from(expected.0[2]),
                            -i16::from(expected.0[3]),
                        ],
                    });
                }
                _ => {
                    pixel_diff += 1;
                    diff_image.put_pixel(x, y, Rgba([255, 0, 0, 255]));
                    diff_image.put_pixel(x + width, y, Rgba([255, 0, 0, 255]));
                    diff_data.push(PixelDiff {
                        x,
                        y,
                        target: [0, 0, 0, 0],
                        actual: [0, 0, 0, 0],
                        difference: [0, 0, 0, 0],
                    });
                }
            }
        }
    }

    if pixel_diff > diff_pixels {
        Some((diff_image, diff_data))
    } else {
        None
    }
}

fn is_pix_diff(pixel1: &Rgba<u8>, pixel2: &Rgba<u8>, threshold: u8) -> bool {
    if pixel1.0[3] == 0 && pixel2.0[3] == 0 {
        return false;
    }

    let mut different = false;

    for i in 0..3 {
        let difference = pixel1.0[i].abs_diff(pixel2.0[i]);
        different |= difference > threshold;
    }

    different
}

/// Writes `<stem>.png` and `<stem>.json`, creating the parent directory if needed, and returns
/// both paths. The extensions are appended so a stem containing dots is preserved.
#[cfg(not(target_arch = "wasm32"))]
pub fn write_diff(
    stem: &std::path::Path,
    diff_image: &RgbaImage,
    report: &DiffReport,
) -> (std::path::PathBuf, std::path::PathBuf) {
    if let Some(parent) = stem.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let with_suffix = |suffix: &str| {
        let mut path = stem.as_os_str().to_owned();
        path.push(suffix);
        std::path::PathBuf::from(path)
    };
    let image_path = with_suffix(".png");
    diff_image
        .save_with_format(&image_path, image::ImageFormat::Png)
        .unwrap();
    let json_path = with_suffix(".json");
    std::fs::write(&json_path, serde_json::to_string_pretty(report).unwrap()).unwrap();
    (image_path, json_path)
}
