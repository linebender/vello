// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Debug helpers for glyph atlas and CPU text resources.

#![allow(dead_code, reason = "only used for debugging purposes")]

#[cfg(feature = "png")]
use crate::Pixmap;
#[cfg(feature = "png")]
use crate::render::Resources;
use crate::text::GlyphAtlasResources;
#[cfg(feature = "png")]
use alloc::format;
use glifo::GlyphCacheKey;
use glifo::atlas::GlyphCacheStats;

#[cfg(feature = "png")]
impl GlyphAtlasResources {
    /// Save all atlas pages to PNG files with a custom path prefix.
    ///
    /// Files are saved as `{path_prefix}_atlas_page_{index}.png`.
    pub(crate) fn save_atlas_pages_to(&self, path_prefix: &str) {
        for (i, pixmap) in self.pixmaps.iter().enumerate() {
            let path = format!("{path_prefix}_atlas_page_{i}.png");
            let _ = save_pixmap_to_png(pixmap, std::path::Path::new(&path));
        }
    }

    /// Save all atlas pages to PNG files for debugging.
    ///
    /// Files are saved to `examples/_output/vello_cpu_atlas_page_{index}.png`.
    pub(crate) fn save_atlas_pages(&self) {
        for (i, pixmap) in self.pixmaps.iter().enumerate() {
            let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.pop(); // up from vello_cpu to sparse_strips
            path.pop(); // up from sparse_strips to workspace root
            path.push("examples");
            path.push("_output");
            let _ = std::fs::create_dir_all(&path);
            path.push(format!("vello_cpu_atlas_page_{i}.png"));
            let _ = save_pixmap_to_png(pixmap, &path);
        }
    }
}

impl GlyphAtlasResources {
    /// Get detailed statistics about cached glyphs.
    pub(crate) fn stats(&self) -> GlyphCacheStats {
        self.glyph_atlas.stats(self.pixmaps.len())
    }

    /// Log detailed atlas statistics at info level.
    pub(crate) fn log_atlas_stats(&self) {
        self.glyph_atlas.log_atlas_stats(self.pixmaps.len());
    }

    /// Returns all cached glyph keys (for debugging).
    pub(crate) fn all_keys(&self) -> impl Iterator<Item = &GlyphCacheKey> {
        self.glyph_atlas.all_keys()
    }

    /// Log all cached keys grouped by glyph ID at info level.
    pub(crate) fn log_keys_grouped(&self) {
        self.glyph_atlas.log_keys_grouped();
    }
}

#[cfg(feature = "png")]
impl Resources {
    pub(crate) fn save_glyph_atlas_pages(&self) {
        if let Some(glyph_resources) = &self.glyph_resources {
            glyph_resources.save_atlas_pages();
        }
    }

    pub(crate) fn save_glyph_atlas_pages_to(&self, path_prefix: &str) {
        if let Some(glyph_resources) = &self.glyph_resources {
            glyph_resources.save_atlas_pages_to(path_prefix);
        }
    }
}

/// Save a pixmap to a PNG file (diagnostic utility).
#[cfg(feature = "png")]
pub(crate) fn save_pixmap_to_png(pixmap: &Pixmap, path: &std::path::Path) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::BufWriter;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = File::create(path)?;
    let w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, pixmap.width() as u32, pixmap.height() as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);

    let mut writer = encoder.write_header().map_err(std::io::Error::other)?;

    writer
        .write_image_data(pixmap.data_as_u8_slice())
        .map_err(std::io::Error::other)?;

    Ok(())
}
