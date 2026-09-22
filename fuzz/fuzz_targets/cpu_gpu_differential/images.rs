// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Image assets shared with the snapshot suite, registered once per renderer.

use crate::scene::FuzzImage;
use std::io::Cursor;
use std::sync::{Arc, LazyLock};
use vello_common::paint::ImageSource;
use vello_common::pixmap::Pixmap;
use vello_tests::renderer::Renderer;

static PIXMAPS: LazyLock<Vec<Arc<Pixmap>>> = LazyLock::new(|| {
    FuzzImage::ALL
        .iter()
        .map(|image| load_asset(*image))
        .collect()
});

/// Image sources of one renderer, in `FuzzImage::ALL` order.
///
/// Sources are renderer-specific handles (the GPU uploads a texture), so a table must only be
/// used with the renderer it was registered on, and a rebuilt renderer needs a new table.
pub(crate) struct ImageTable {
    sources: Vec<ImageSource>,
}

impl ImageTable {
    pub(crate) fn register(renderer: &mut impl Renderer) -> Self {
        let sources = PIXMAPS
            .iter()
            .map(|pixmap| renderer.get_image_source(pixmap.clone()))
            .collect();
        Self { sources }
    }

    pub(crate) fn source(&self, image: FuzzImage) -> &ImageSource {
        &self.sources[image as usize]
    }
}

fn load_asset(image: FuzzImage) -> Arc<Pixmap> {
    let path = format!(
        "{}/../vello_tests/tests/assets/{}.png",
        env!("CARGO_MANIFEST_DIR"),
        image.asset_name()
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("reading {path}: {error}"));
    let pixmap = Pixmap::from_png(Cursor::new(bytes))
        .unwrap_or_else(|error| panic!("decoding {path}: {error}"));
    Arc::new(pixmap)
}
