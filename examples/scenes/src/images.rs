use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use vello::peniko::{Blob, Format, Image};

/// Simple hack to support loading images for examples.
#[derive(Default)]
pub struct ImageCache {
    files: HashMap<PathBuf, Image>,
    bytes: HashMap<usize, Image>,
}

impl ImageCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_file(&mut self, path: impl AsRef<Path>) -> Option<Image> {
        let path = path.as_ref();
        if let Some(image) = self.files.get(path) {
            Some(image.clone())
        } else {
            let data = std::fs::read(path).ok()?;
            let image = decode_image(&data)?;
            self.files.insert(path.to_owned(), image.clone());
            Some(image)
        }
    }

    pub fn from_bytes(&mut self, key: usize, bytes: &[u8]) -> Option<Image> {
        if let Some(image) = self.bytes.get(&key) {
            Some(image.clone())
        } else {
            let image = decode_image(bytes)?;
            self.bytes.insert(key, image.clone());
            Some(image)
        }
    }
}

fn decode_image(data: &[u8]) -> Option<Image> {
    let image = image::io::Reader::new(std::io::Cursor::new(data))
        .with_guessed_format()
        .ok()?
        .decode()
        .ok()?;
    let width = image.width();
    let height = image.height();
    let data = Arc::new(image.into_rgba8().into_vec());
    let blob = Blob::new(data);
    Some(Image::new(blob, Format::Rgba8, width, height))
}
