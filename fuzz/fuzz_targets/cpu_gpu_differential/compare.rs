// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Decides whether two rendered images agree and produces diff reports when they do not.

use crate::config::TOLERANCE;
use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use vello_common::pixmap::Pixmap;
use vello_tests::diff::{DiffReport, get_diff, write_diff};

pub(crate) struct Mismatch {
    /// Pixel indices exceeding the channel tolerance, in image order.
    outliers: Vec<usize>,
    max_channel_delta: u8,
    first_outlier: Option<(usize, [u8; 4], [u8; 4], u8)>,
}

impl Mismatch {
    /// Fingerprint of which pixels differ; inputs that render the same divergence share it even
    /// when the bytes differ.
    pub(crate) fn signature(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.outliers.hash(&mut hasher);
        hasher.finish()
    }
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CPU/GPU image mismatch: {} pixels exceeded the channel tolerance of {}; maximum \
             delta was {}; first outlier: {:?}",
            self.outliers.len(),
            TOLERANCE.channel,
            self.max_channel_delta,
            self.first_outlier
        )
    }
}

/// Panics if the images have different dimensions; that is a harness bug, not a rendering one.
pub(crate) fn compare_images(cpu: &Pixmap, gpu: &Pixmap) -> Result<(), Mismatch> {
    assert_eq!(
        (cpu.width(), cpu.height()),
        (gpu.width(), gpu.height()),
        "CPU and GPU returned different image dimensions"
    );

    let mut outliers = Vec::new();
    let mut max_channel_delta = 0;
    let mut first_outlier = None;
    let (cpu_pixels, _) = cpu.data_as_u8_slice().as_chunks::<4>();
    let (gpu_pixels, _) = gpu.data_as_u8_slice().as_chunks::<4>();
    for (index, (cpu_pixel, gpu_pixel)) in cpu_pixels.iter().zip(gpu_pixels).enumerate() {
        let mut pixel_delta = 0;
        for (&cpu_channel, &gpu_channel) in cpu_pixel.iter().zip(gpu_pixel) {
            pixel_delta = pixel_delta.max(cpu_channel.abs_diff(gpu_channel));
        }
        max_channel_delta = max_channel_delta.max(pixel_delta);
        if pixel_delta > TOLERANCE.channel {
            outliers.push(index);
            first_outlier.get_or_insert((index, *cpu_pixel, *gpu_pixel, pixel_delta));
        }
    }

    if outliers.len() > TOLERANCE.max_outlier_pixels {
        return Err(Mismatch {
            outliers,
            max_channel_delta,
            first_outlier,
        });
    }
    Ok(())
}

/// Writes `<stem>.png` and `<stem>.json` for two rendered images, comparing them exactly as the
/// snapshot suite would. A side-by-side image is written even when no pixel exceeds the tolerance.
pub(crate) fn write_diff_report(
    cpu_image: Pixmap,
    gpu_image: Pixmap,
    stem: &Path,
) -> (PathBuf, PathBuf, DiffReport) {
    let to_rgba = |pixmap: Pixmap| {
        image::load_from_memory(&pixmap.into_png().unwrap())
            .unwrap()
            .into_rgba8()
    };
    let cpu_image = to_rgba(cpu_image);
    let gpu_image = to_rgba(gpu_image);

    let (diff_image, pixels) = get_diff(&cpu_image, &gpu_image, TOLERANCE.channel, 0)
        .unwrap_or_else(|| {
            let mut side_by_side = image::RgbaImage::new(cpu_image.width() * 3, cpu_image.height());
            image::imageops::replace(&mut side_by_side, &cpu_image, 0, 0);
            image::imageops::replace(
                &mut side_by_side,
                &gpu_image,
                i64::from(cpu_image.width()) * 2,
                0,
            );
            (side_by_side, Vec::new())
        });
    let report = DiffReport::new(pixels);
    let (image_path, json_path) = write_diff(stem, &diff_image, &report);
    (image_path, json_path, report)
}
