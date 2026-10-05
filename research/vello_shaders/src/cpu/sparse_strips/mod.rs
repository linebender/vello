// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU sparse strips with MSAA16 sample masks.
//!
//! A port of Skia's CPU sparse strips pipeline (`src/gpu/graphite/sparse_strips`), used as a
//! temporary stand-in for the GPU tile / sort / strip allocate / MSAA render / seam join stages.
//! Instead of reduced alpha, it writes a 16-bit sample mask per pixel, which the GPU coarse and
//! fine stages consume (`AaConfig::SparseMsaa16`).
//!
//! Pipeline, per scene:
//! 1. [`front::flatten_scene`]: the existing CPU pathtag reduce/scan, `bbox_clear` and flatten
//!    stages (Euler-spiral fills and strokes), producing per-path line slices and `PathBbox`es.
//! 2. Per path: tile ([`tiler`]), sort, then traverse the sorted tiles ([`make_strips`]),
//!    accumulating per-sample winding ([`processor`]) and resolving it to masks with the fill rule.
//! 3. Output through [`sink::StripSink`] as [`StripRecord`]s plus mask blocks.
//!
//! ## CPU → GPU contract
//!
//! - Tiles are 16×16 pixels, matching `TILE_WIDTH`/`TILE_HEIGHT` in the shaders.
//! - Records: see [`StripRecord`]. The GPU `strip_scatter` stage writes mask tiles into the
//!   `Tile` array allocated by `tile_alloc` (`segment_count_or_ix = block + 1`), and adds backdrop
//!   deltas. `backdrop_dyn` turns those into backdrops (1 inside solid spans, 0 elsewhere).
//! - Masks: [`MASK_WORDS_PER_TILE`] `u32`s per block. Pixel `(x, y)` of the tile is the `u16` at
//!   index `y * 16 + x` (word `y * 8 + x / 2`, low half for even `x`). Bit `k` is set when sample
//!   `k` is inside, and sample `k` sits at `((MSAA16_PATTERN[k] + 0.5) / 16, (k + 0.5) / 16)`
//!   within the pixel. The fill rule has already been applied.

pub mod front;
pub mod reference;
pub mod sink;
pub mod tile;

mod lut;
mod make_strips;
mod processor;
mod swar;
mod tiler;

#[cfg(test)]
mod tests;

use vello_encoding::{ConfigUniform, DRAW_INFO_FLAGS_FILL_RULE_BIT, PathBbox};

pub use sink::{StripRecord, StripSink};

/// Tile width and height in pixels.
pub const TILE_SIZE: u16 = 16;
/// Samples per pixel.
pub const SAMPLES: usize = 16;
/// `u32` words per mask block: 16×16 pixels × 16 bits / 32.
pub const MASK_WORDS_PER_TILE: usize = TILE_SIZE as usize * TILE_SIZE as usize / 2;
/// Set in [`StripRecord::payload`] for backdrop delta records.
pub const KIND_DELTA: u32 = 1 << 31;
/// Sample `k` of a pixel is at `((MSAA16_PATTERN[k] + 0.5) / 16, (k + 0.5) / 16)`.
///
/// This is Skia's `kMsaaPattern<uint16_t>`: an n-rooks pattern.
pub const MSAA16_PATTERN: [u8; 16] = [1, 8, 4, 11, 15, 7, 3, 12, 0, 9, 5, 13, 2, 10, 6, 14];

/// Which rasterizer produces the masks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// The Skia port (tiler + strip processor).
    #[default]
    Skia,
    /// The exact, slow scanline reference.
    Reference,
}

/// Output of [`render_sparse_strips`].
///
/// `records` and `masks` may be empty; the GPU side must pad its uploads.
#[derive(Debug, Default)]
pub struct SparseStripsOutput {
    /// From CPU flatten: bbox, `draw_flags` (fill rule) and `trans_ix` per path.
    pub path_bboxes: Vec<PathBbox>,
    /// Mask tiles and backdrop deltas, in no particular order.
    pub records: Vec<StripRecord>,
    /// Mask blocks, [`MASK_WORDS_PER_TILE`] words each.
    pub masks: Vec<u32>,
}

/// Run the CPU sparse strips pipeline on a packed scene.
pub fn render_sparse_strips(config: &ConfigUniform, scene: &[u32]) -> SparseStripsOutput {
    render_sparse_strips_with(config, scene, Backend::default())
}

/// [`render_sparse_strips`] with an explicit [`Backend`].
pub fn render_sparse_strips_with(
    config: &ConfigUniform,
    scene: &[u32],
    backend: Backend,
) -> SparseStripsOutput {
    let flat = front::flatten_scene(config, scene);
    let width = config.target_width;
    let height = config.target_height;
    let mut sink = StripSink::new(width.div_ceil(TILE_SIZE as u32));
    let mut rasterizer = make_strips::Rasterizer::new(width, height);
    for (path_ix, bbox) in flat.path_bboxes.iter().enumerate() {
        let lines = flat.path_lines(path_ix);
        if lines.is_empty() {
            continue;
        }
        let even_odd = (bbox.draw_flags & DRAW_INFO_FLAGS_FILL_RULE_BIT) != 0;
        sink.begin_path(path_ix as u32);
        match backend {
            Backend::Skia => rasterizer.rasterize_path(lines, even_odd, &mut sink),
            Backend::Reference => {
                reference::rasterize_path_reference(lines, even_odd, width, height, &mut sink);
            }
        }
    }
    SparseStripsOutput {
        path_bboxes: flat.path_bboxes,
        records: sink.records,
        masks: sink.masks,
    }
}
