// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Abstraction for generating strips from paths.

use crate::clip::{PathDataRef, intersect};
use crate::fearless_simd::Level;
use crate::flatten::{FlattenCtx, Line};
use crate::kurbo::{Affine, PathEl, Stroke};
use crate::peniko::Fill;
use crate::strip::{Strip, PreMergeTile};
use crate::tile::Tiles;
use crate::{flatten, strip};
use alloc::vec::Vec;
use peniko::kurbo::StrokeCtx;

/// A storage for storing strip-related data.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct StripStorage {
    /// The strips in the storage.
    pub strips: Vec<Strip>,
    /// The alphas in the storage.
    pub alphas: Vec<u8>,
    /// Asdasd
    pub pre_merge_tiles: Vec<PreMergeTile>,
    generation_mode: GenerationMode,
}

/// The generation mode of the strip storage.
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq)]
pub enum GenerationMode {
    #[default]
    /// Clear strips before generating the new ones.
    Replace,
    /// Don't clear strips, append to the existing buffer.
    Append,
}

const MASK_WIDTH: u32 = 64;
const MASK_HEIGHT: u32 = 64;
const PACKING_SCALE: f32 = 0.5;
const PATTERN: [u8; 8] = [0, 5, 3, 7, 1, 4, 6, 2];

const SUB_X: [f32; 8] = [
    (PATTERN[0] as f32 + 0.5) * 0.125,
    (PATTERN[1] as f32 + 0.5) * 0.125,
    (PATTERN[2] as f32 + 0.5) * 0.125,
    (PATTERN[3] as f32 + 0.5) * 0.125,
    (PATTERN[4] as f32 + 0.5) * 0.125,
    (PATTERN[5] as f32 + 0.5) * 0.125,
    (PATTERN[6] as f32 + 0.5) * 0.125,
    (PATTERN[7] as f32 + 0.5) * 0.125,
];
const SUB_Y: [f32; 8] = [
    (0.0 + 0.5) * 0.125,
    (1.0 + 0.5) * 0.125,
    (2.0 + 0.5) * 0.125,
    (3.0 + 0.5) * 0.125,
    (4.0 + 0.5) * 0.125,
    (5.0 + 0.5) * 0.125,
    (6.0 + 0.5) * 0.125,
    (7.0 + 0.5) * 0.125,
];

impl StripStorage {
    /// Reset the storage.
    pub fn clear(&mut self) {
        self.strips.clear();
        self.alphas.clear();
        self.pre_merge_tiles.clear()
    }

    /// Set the generation mode of the storage.
    pub fn set_generation_mode(&mut self, mode: GenerationMode) {
        self.generation_mode = mode;
    }

    /// Whether the strip storage is empty.
    pub fn is_empty(&self) -> bool {
        self.strips.is_empty() && self.alphas.is_empty()
    }

    /// Extend the current strip storage with the data from another storage.
    pub fn extend(&mut self, other: &Self) {
        self.strips.extend(&other.strips);
        self.alphas.extend(&other.alphas);
    }
}

/// An object for easily generating strips for a filled/stroked path.
#[derive(Debug)]
pub struct StripGenerator {
    pub(crate) level: Level,
    line_buf: Vec<Line>,
    flatten_ctx: FlattenCtx,
    stroke_ctx: StrokeCtx,
    temp_storage: StripStorage,
    tiles: Tiles,
    width: u16,
    height: u16,
    mask_lut: Vec<u8>
}

impl StripGenerator {
    /// Internal helper to generate the half_plane lut
    fn make_mask_lut_half_plane() -> Vec<u8> {
        let mut lut = Vec::with_capacity((MASK_WIDTH * MASK_HEIGHT) as usize);

        for j in 0..MASK_HEIGHT {
            for i in 0..MASK_WIDTH {
                let xf = (i as f32 + 0.5) / MASK_WIDTH as f32;
                let yf = (j as f32 + 0.5) / MASK_HEIGHT as f32;

                let n_rev = (2.0 * (xf - 0.5), 2.0 * (yf - 0.5));

                let mut lg_rev = n_rev.0.hypot(n_rev.1);
                if lg_rev < 1e-9 {
                    lg_rev = 1e-9;
                }

                let n_lookup = (n_rev.0 / lg_rev, n_rev.1 / lg_rev);

                let c_dist_unsigned = (1.0 - lg_rev).max(0.0) * (1.0 / PACKING_SCALE);

                let mut n_canonical = n_lookup;
                let mut c_signed_dist = c_dist_unsigned;

                if n_lookup.0 < 0.0 {
                    n_canonical.0 = -n_lookup.0;
                    n_canonical.1 = -n_lookup.1;
                    c_signed_dist = -c_dist_unsigned;
                }

                let c_plane = c_signed_dist + 0.5 * (n_canonical.0 + n_canonical.1);

                let mut mask: u8 = 0;
                for k in 0..8 {
                    let p = (SUB_X[k], SUB_Y[k]);
                    if n_canonical.0 * p.0 + n_canonical.1 * p.1 - c_plane > 0.0 {
                        mask |= 1 << k;
                    }
                }
                lut.push(mask);
            }
        }
        lut
    }

    /// Create a new strip generator.
    pub fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            level,
            line_buf: Vec::new(),
            tiles: Tiles::new(level),
            flatten_ctx: FlattenCtx::default(),
            stroke_ctx: StrokeCtx::default(),
            temp_storage: StripStorage::default(),
            width,
            height,
            mask_lut: Self::make_mask_lut_half_plane(),
        }
    }

    /// Generate the strips for a filled path.
    pub fn generate_filled_path(
        &mut self,
        path: impl IntoIterator<Item = PathEl>,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
        strip_storage: &mut StripStorage,
        clip_path: Option<PathDataRef<'_>>,
    ) {
        flatten::fill(
            self.level,
            path,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );

        self.generate_with_clip(aliasing_threshold, strip_storage, fill_rule, clip_path);
    }

    /// Generate the strips for a stroked path.
    pub fn generate_stroked_path(
        &mut self,
        path: impl IntoIterator<Item = PathEl>,
        stroke: &Stroke,
        transform: Affine,
        aliasing_threshold: Option<u8>,
        strip_storage: &mut StripStorage,
        clip_path: Option<PathDataRef<'_>>,
    ) {
        flatten::stroke(
            self.level,
            path,
            stroke,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
            &mut self.stroke_ctx,
        );
        self.generate_with_clip(aliasing_threshold, strip_storage, Fill::NonZero, clip_path);
    }

    fn generate_with_clip(
        &mut self,
        aliasing_threshold: Option<u8>,
        strip_storage: &mut StripStorage,
        fill_rule: Fill,
        clip_path: Option<PathDataRef<'_>>,
    ) {
        if strip_storage.generation_mode == GenerationMode::Replace {
            strip_storage.strips.clear();
        }

        self.tiles
            .make_tiles_msaa(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();

        if let Some(clip_path) = clip_path {
            self.temp_storage.clear();

            strip::render(
                self.level,
                &self.tiles,
                &mut strip_storage.strips,
                &mut strip_storage.pre_merge_tiles,
                &mut strip_storage.alphas,
                fill_rule,
                &self.line_buf,
                &self.mask_lut,
            );
            // let path_data = PathDataRef {
            //     strips: &self.temp_storage.strips,
            //     alphas: &self.temp_storage.alphas,
            // };
            // intersect(self.level, clip_path, path_data, strip_storage);
        } else {
            strip::render(
                self.level,
                &self.tiles,
                &mut strip_storage.strips,
                &mut strip_storage.pre_merge_tiles,
                &mut strip_storage.alphas,
                fill_rule,
                &self.line_buf,
                &self.mask_lut,
            );
        }
    }

    /// Reset the strip generator.
    pub fn reset(&mut self) {
        self.line_buf.clear();
        self.tiles.reset();
        self.temp_storage.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::fearless_simd::Level;
    use crate::kurbo::{Affine, Rect, Shape};
    use crate::peniko::Fill;
    use crate::strip_generator::{StripGenerator, StripStorage};

    #[test]
    fn reset() {
        let mut generator = StripGenerator::new(100, 100, Level::fallback());
        let mut storage = StripStorage::default();
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);

        generator.generate_filled_path(
            rect.to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            None,
            &mut storage,
            None,
        );

        assert!(!generator.line_buf.is_empty());
        assert!(!storage.is_empty());

        generator.reset();
        storage.clear();

        assert!(generator.line_buf.is_empty());
        assert!(storage.is_empty());
    }
}
