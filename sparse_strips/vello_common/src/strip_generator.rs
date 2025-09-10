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

/// A storage for storing strip-related data.
#[derive(Debug)]
#[derive(Default)]
pub struct StripStorage {
    /// The strips in the storage.
    pub strips: Vec<Strip>,
    /// The alphas in the storage.
    pub alphas: Vec<u8>,
}

impl StripStorage {
    /// Reset the storage.
    pub fn reset(&mut self) {
        self.strips.clear();
        self.alphas.clear();
    }
}


/// An object for easily generating strips for a filled/stroked path.
#[derive(Debug)]
pub struct StripGenerator {
    level: Level,
    line_buf: Vec<Line>,
    flatten_ctx: FlattenCtx,
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
        clear_strips: bool,
    ) {
        flatten::fill(
            self.level,
            path,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );
        self.make_strips(strip_storage, fill_rule, aliasing_threshold, clear_strips);
    }

    /// Generate the strips for a stroked path.
    pub fn generate_stroked_path(
        &mut self,
        path: impl IntoIterator<Item = PathEl>,
        stroke: &Stroke,
        transform: Affine,
        aliasing_threshold: Option<u8>,
        strip_storage: &mut StripStorage,
        clear_strips: bool,
    ) {
        flatten::stroke(
            self.level,
            path,
            stroke,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );
        self.make_strips(
            strip_storage,
            Fill::NonZero,
            aliasing_threshold,
            clear_strips,
        );
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
        clear_strips: bool,
    ) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();

        if clear_strips {
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
            true,
        );

        assert!(!generator.line_buf.is_empty());
        assert!(!storage.strips.is_empty());
        assert!(!storage.alphas.is_empty());

        generator.reset();
        storage.reset();

        assert!(generator.line_buf.is_empty());
        assert!(storage.strips.is_empty());
        assert!(storage.alphas.is_empty());
    }
}
