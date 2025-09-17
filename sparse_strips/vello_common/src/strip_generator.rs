// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Abstraction for generating strips from paths.

use crate::fearless_simd::Level;
use crate::flatten::{FlattenCtx, Line};
use crate::kurbo::{Affine, PathEl, Stroke};
use crate::peniko::Fill;
use crate::strip::Strip;
use crate::tile::Tiles;
use crate::{flatten, strip};
use alloc::vec::Vec;
use peniko::kurbo::StrokeCtx;

/// A storage for storing strip-related data.
#[derive(Debug, Default)]
pub struct StripStorage {
    /// The strips in the storage.
    pub strips: Vec<Strip>,
    /// The alphas in the storage.
    pub alphas: Vec<u8>,
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
}

/// An object for easily generating strips for a filled/stroked path.
#[derive(Debug)]
pub struct StripGenerator {
    level: Level,
    line_buf: Vec<Line>,
    flatten_ctx: FlattenCtx,
    stroke_ctx: StrokeCtx,
    tiles: Tiles,
    width: u16,
    height: u16,
}

impl StripGenerator {
    /// Create a new strip generator.
    pub fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            level,
            line_buf: Vec::new(),
            tiles: Tiles::new(level),
            flatten_ctx: FlattenCtx::default(),
            stroke_ctx: StrokeCtx::default(),
            width,
            height,
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
    ) {
        flatten::fill(
            self.level,
            path,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );
        self.make_strips(strip_storage, fill_rule, aliasing_threshold);
    }

    /// Generate the strips for a stroked path.
    pub fn generate_stroked_path(
        &mut self,
        path: impl IntoIterator<Item = PathEl>,
        stroke: &Stroke,
        transform: Affine,
        aliasing_threshold: Option<u8>,
        strip_storage: &mut StripStorage,
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
        self.make_strips(strip_storage, Fill::NonZero, aliasing_threshold);
    }

    /// Reset the strip generator.
    pub fn reset(&mut self) {
        self.line_buf.clear();
        self.tiles.reset();
    }

    fn make_strips(
        &mut self,
        strip_storage: &mut StripStorage,
        fill_rule: Fill,
        aliasing_threshold: Option<u8>,
    ) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();

        if strip_storage.generation_mode == GenerationMode::Replace {
            strip_storage.strips.clear();
        }

        strip::render(
            self.level,
            &self.tiles,
            &mut strip_storage.strips,
            &mut strip_storage.alphas,
            fill_rule,
            aliasing_threshold,
            &self.line_buf,
        );
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
        );

        assert!(!generator.line_buf.is_empty());
        assert!(!storage.is_empty());

        generator.reset();
        storage.clear();

        assert!(generator.line_buf.is_empty());
        assert!(storage.is_empty());
    }
}
