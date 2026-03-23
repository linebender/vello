// Copyright 2026 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Debug helpers for [`CpuGlyphAtlas`] and [`CpuGlyphCaches`].

#[cfg(feature = "png")]
use crate::Pixmap;
use crate::atlas::GlyphCacheKey;
use crate::atlas::GlyphCacheStats;
use crate::renderers::vello_cpu::CpuGlyphAtlas;
#[cfg(feature = "png")]
use crate::renderers::vello_cpu::CpuGlyphCaches;
#[cfg(feature = "png")]
use alloc::format;

#[cfg(feature = "png")]
impl CpuGlyphAtlas {
    /// Save every atlas page as `{path_prefix}_atlas_page_{index}.png`.
    pub fn save_atlas_pages_to(&self, path_prefix: &str) {
        for (i, pixmap) in self.pixmaps.iter().enumerate() {
            let path = format!("{path_prefix}_atlas_page_{i}.png");
            let _ = save_pixmap_to_png(pixmap, std::path::Path::new(&path));
        }
    }

    /// Save every atlas page under `examples/_output/vello_cpu_atlas_page_{index}.png`.
    pub fn save_atlas_pages(&self) {
        for (i, pixmap) in self.pixmaps.iter().enumerate() {
            let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.pop(); // up from parley_draw to workspace root
            path.push("examples");
            path.push("_output");
            let _ = std::fs::create_dir_all(&path);
            path.push(format!("vello_cpu_atlas_page_{i}.png"));
            let _ = save_pixmap_to_png(pixmap, &path);
        }
    }
}

impl CpuGlyphAtlas {
    /// Get detailed statistics about cached glyphs.
    pub fn stats(&self) -> GlyphCacheStats {
        self.inner.stats(self.pixmaps.len())
    }

    /// Log detailed atlas statistics at info level.
    pub fn log_atlas_stats(&self) {
        self.inner.log_atlas_stats(self.pixmaps.len());
    }

    /// Returns all cached glyph keys (for debugging).
    pub fn all_keys(&self) -> impl Iterator<Item = &GlyphCacheKey> {
        self.inner.all_keys()
    }

    /// Log all cached keys grouped by glyph ID at info level.
    pub fn log_keys_grouped(&self) {
        self.inner.log_keys_grouped();
    }
}

#[cfg(feature = "png")]
impl CpuGlyphCaches {
    /// Save all atlas pages to PNG files for debugging.
    ///
    /// Files are saved to `examples/_output/vello_cpu_atlas_page_{index}.png`.
    pub fn save_atlas_pages(&self) {
        self.glyph_atlas.save_atlas_pages();
    }

    /// Save all atlas pages to PNG files with a custom path prefix.
    ///
    /// Files are saved as `{path_prefix}_atlas_page_{index}.png`.
    pub fn save_atlas_pages_to(&self, path_prefix: &str) {
        self.glyph_atlas.save_atlas_pages_to(path_prefix);
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
