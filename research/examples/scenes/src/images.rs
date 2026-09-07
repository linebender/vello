// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use vello::peniko::{Blob, ImageBrush, ImageData, ImageFormat};

/// Simple hack to support loading images for examples.
#[derive(Default)]
pub struct ImageCache {
    files: HashMap<PathBuf, ImageData>,
    bytes: HashMap<usize, ImageData>,
}

impl ImageCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_file(&mut self, path: impl AsRef<Path>) -> anyhow::Result<ImageBrush> {
        let path = path.as_ref();
        if let Some(image) = self.files.get(path) {
            Ok(image.clone().into())
        } else {
            let data = std::fs::read(path)?;
            let image = decode_image(&data)?;
            self.files.insert(path.to_owned(), image.clone());
            Ok(image.into())
        }
    }

    pub fn from_bytes(&mut self, key: usize, bytes: &[u8]) -> anyhow::Result<ImageBrush> {
        if let Some(image) = self.bytes.get(&key) {
            Ok(image.clone().into())
        } else {
            let image = decode_image(bytes)?;
            self.bytes.insert(key, image.clone());
            Ok(image.into())
        }
    }
}

fn decode_image(data: &[u8]) -> anyhow::Result<ImageData> {
    let image = image::ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()?
        .decode()?;
    let width = image.width();
    let height = image.height();
    let data = Arc::new(image.into_rgba8().into_vec());
    let blob = Blob::new(data);
    Ok(ImageData {
        data: blob,
        format: ImageFormat::Rgba8,
        width,
        height,
        alpha_type: vello::peniko::ImageAlphaType::Alpha,
    })
}
