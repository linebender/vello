// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Abstraction for generating strips from paths.

use crate::clip::{PathDataRef, intersect};
use crate::fearless_simd::Level;
use crate::flatten::{FlattenCtx, Line};
use crate::kurbo::{Affine, PathEl, Stroke};
use crate::peniko::Fill;
use crate::peniko::kurbo::StrokeCtx;
use crate::strip::{MsaaMask, Strip};
use crate::tile::Tiles;
use crate::{flatten, strip};
use alloc::vec::Vec;
use core::fmt::Debug;

/// a
pub trait AliasingConfig: Debug + Default + 'static {
    /// The type of mask used for this aliasing mode (u8 or u16).
    type Mask: MsaaMask;

    /// Returns the Lookup Table (LUT) for this mode.
    fn get_lut(&self) -> &[Self::Mask];

    /// dispatch method to call the correct tile generation logic.
    fn make_tiles(tiles: &mut Tiles, lines: &[Line], width: u16, height: u16);
}

/// Analytic Anti-Aliasing (Default).
#[derive(Debug, Default, Clone, Copy)]
pub struct Analytic;

impl AliasingConfig for Analytic {
    type Mask = u8;

    fn get_lut(&self) -> &[Self::Mask] {
        &[]
    }

    fn make_tiles(tiles: &mut Tiles, lines: &[Line], width: u16, height: u16) {
        tiles.make_tiles_analytic_aa(lines, width, height);
    }
}

/// 8-sample Multi-Sample Anti-Aliasing.
#[derive(Debug, Clone)]
pub struct Msaa8 {
    lut: Vec<u8>,
}

impl Default for Msaa8 {
    fn default() -> Self {
        Self {
            lut: strip::generate_mask_lut_msaa8(),
        }
    }
}

impl AliasingConfig for Msaa8 {
    type Mask = u8;

    fn get_lut(&self) -> &[Self::Mask] {
        &self.lut
    }

    fn make_tiles(tiles: &mut Tiles, lines: &[Line], width: u16, height: u16) {
        tiles.make_tiles_msaa(lines, width, height);
    }
}

/// 16-sample Multi-Sample Anti-Aliasing.
#[derive(Debug, Clone)]
pub struct Msaa16 {
    lut: Vec<u16>,
}

impl Default for Msaa16 {
    fn default() -> Self {
        Self {
            lut: strip::generate_mask_lut_msaa16(),
        }
    }
}

impl AliasingConfig for Msaa16 {
    type Mask = u16;

    fn get_lut(&self) -> &[Self::Mask] {
        &self.lut
    }

    fn make_tiles(tiles: &mut Tiles, lines: &[Line], width: u16, height: u16) {
        tiles.make_tiles_msaa(lines, width, height);
    }
}

/// A storage for storing strip-related data.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct StripStorage {
    /// The strips in the storage.
    pub strips: Vec<Strip>,
    /// The alphas in the storage.
    pub alphas: Vec<u8>,
    /// The generation mode of the strip storage.
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

impl StripStorage {
    /// Reset the storage.
    pub fn clear(&mut self) {
        self.strips.clear();
        self.alphas.clear();
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
pub struct StripGenerator<C: AliasingConfig> {
    pub(crate) level: Level,
    line_buf: Vec<Line>,
    flatten_ctx: FlattenCtx,
    stroke_ctx: StrokeCtx,
    temp_storage: StripStorage,
    tiles: Tiles,
    width: u16,
    height: u16,
    config: C,
}

impl<C: AliasingConfig> StripGenerator<C> {
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
            config: C::default(),
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

        C::make_tiles(&mut self.tiles, &self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();

        let mask_lut = self.config.get_lut();

        if let Some(clip_path) = clip_path {
            self.temp_storage.clear();

            strip::render(
                self.level,
                &self.tiles,
                &mut self.temp_storage.strips,
                &mut self.temp_storage.alphas,
                fill_rule,
                aliasing_threshold,
                &self.line_buf,
                mask_lut,
            );
            let path_data = PathDataRef {
                strips: &self.temp_storage.strips,
                alphas: &self.temp_storage.alphas,
            };

            intersect(self.level, clip_path, path_data, strip_storage);
        } else {
            strip::render(
                self.level,
                &self.tiles,
                &mut strip_storage.strips,
                &mut strip_storage.alphas,
                fill_rule,
                aliasing_threshold,
                &self.line_buf,
                mask_lut,
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
    use crate::strip_generator::{Analytic, Msaa16, StripGenerator, StripStorage};

    #[test]
    fn reset_analytic() {
        let mut generator = StripGenerator::<Analytic>::new(100, 100, Level::fallback());
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

    #[test]
    fn msaa_generation() {
        let mut generator = StripGenerator::<Msaa16>::new(100, 100, Level::fallback());
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
        assert!(!storage.is_empty());
    }
}
