// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Abstraction for generating strips from paths.

use crate::clip::{PathDataRef, intersect};
use crate::fearless_simd::Level;
use crate::flatten::{FlattenCtx, Line};
use crate::kurbo::{Affine, PathEl, Rect, Stroke};
use crate::peniko::Fill;
use crate::strip::Strip;
use crate::tile::Tiles;
use crate::{flatten, rect, strip};
use alloc::vec::Vec;
use peniko::kurbo::StrokeCtx;

/// A storage for storing strip-related data.
#[derive(Debug, Default, PartialEq, Eq)]
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
    /// Create a new strip storage with the given generation mode.
    pub fn new(generation_mode: GenerationMode) -> Self {
        Self {
            strips: Vec::new(),
            alphas: Vec::new(),
            generation_mode,
        }
    }

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
pub struct StripGenerator {
    pub(crate) level: Level,
    line_buf: Vec<Line>,
    flatten_ctx: FlattenCtx,
    stroke_ctx: StrokeCtx,
    temp_storage: StripStorage,
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
            temp_storage: StripStorage::default(),
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
        clip_path: Option<PathDataRef<'_>>,
    ) {
        flatten::fill(
            self.level,
            path,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
            self.width,
            self.height,
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
            self.width,
            self.height,
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
        self.tiles
            .make_tiles_analytic_aa(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();

        let level = self.level;
        let tiles = &self.tiles;
        let line_buf = &self.line_buf;
        render_with_clip(
            level,
            &mut self.temp_storage,
            strip_storage,
            clip_path,
            |strips, alphas| {
                strip::render(
                    level,
                    tiles,
                    strips,
                    alphas,
                    fill_rule,
                    aliasing_threshold,
                    line_buf,
                );
            },
        );
    }

    /// Generate strips directly for a pixel-aligned rectangle.
    ///
    /// This bypasses the full path processing pipeline (flatten -> tiles -> strips)
    /// by directly creating strip coverage data for the rectangle.
    pub fn generate_filled_rect_fast(
        &mut self,
        rect: &Rect,
        strip_storage: &mut StripStorage,
        clip_path: Option<PathDataRef<'_>>,
    ) {
        let viewport = Rect::new(0.0, 0.0, self.width as f64, self.height as f64);
        let clamped = rect.intersect(viewport);

        let level = self.level;
        render_with_clip(
            level,
            &mut self.temp_storage,
            strip_storage,
            clip_path,
            |strips, alphas| {
                rect::render(level, clamped, strips, alphas);
            },
        );
    }

    /// Reset the strip generator.
    pub fn reset(&mut self) {
        self.line_buf.clear();
        self.tiles.reset();
        self.temp_storage.clear();
    }
}

/// Render strips via `render_fn` with optional clip intersection.
///
/// When `clip_path` is `Some`, strips are rendered into `temp_storage` first, then
/// intersected with the clip mask into `strip_storage`. Otherwise strips are rendered
/// directly into `strip_storage`.
fn render_with_clip(
    level: Level,
    temp_storage: &mut StripStorage,
    strip_storage: &mut StripStorage,
    clip_path: Option<PathDataRef<'_>>,
    render_fn: impl FnOnce(&mut Vec<Strip>, &mut Vec<u8>),
) {
    if strip_storage.generation_mode == GenerationMode::Replace {
        strip_storage.strips.clear();
    }

    if let Some(clip_path) = clip_path {
        temp_storage.clear();

        render_fn(&mut temp_storage.strips, &mut temp_storage.alphas);

        let path_data = PathDataRef {
            strips: &temp_storage.strips,
            alphas: &temp_storage.alphas,
        };
        intersect(level, clip_path, path_data, strip_storage);
    } else {
        render_fn(&mut strip_storage.strips, &mut strip_storage.alphas);
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
        let mut generator = StripGenerator::new(100, 100, Level::baseline());
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

    /// Assert that `generate_filled_rect_fast` produces the same strips as the
    /// path-based pipeline for the given pixel-aligned rectangle.
    fn assert_rect_fast_eq_path(rect: Rect, test_name: &str) {
        let mut generator = StripGenerator::new(100, 100, Level::baseline());
        let mut storage_path = StripStorage::default();
        let mut storage_rect = StripStorage::default();

        generator.generate_filled_path(
            rect.to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            None,
            &mut storage_path,
            None,
        );
        generator.reset();

        generator.generate_filled_rect_fast(&rect, &mut storage_rect, None);

        assert_eq!(
            storage_path.strips, storage_rect.strips,
            "{test_name}: strips mismatch",
        );
        assert_eq!(
            storage_path.alphas, storage_rect.alphas,
            "{test_name}: alphas mismatch",
        );
    }

    #[test]
    fn rect_small_single_tile() {
        assert_rect_fast_eq_path(Rect::new(1.0, 1.0, 3.0, 3.0), "small_single_tile");
    }

    #[test]
    fn rect_spanning_multiple_tiles_horizontally() {
        assert_rect_fast_eq_path(Rect::new(2.0, 1.0, 14.0, 3.0), "spanning_horizontal");
    }

    #[test]
    fn rect_spanning_multiple_tiles_vertically() {
        assert_rect_fast_eq_path(Rect::new(1.0, 2.0, 3.0, 14.0), "spanning_vertical");
    }

    #[test]
    fn rect_spanning_multiple_tiles_both_directions() {
        assert_rect_fast_eq_path(Rect::new(2.0, 2.0, 18.0, 18.0), "spanning_both");
    }

    #[test]
    fn rect_tile_aligned() {
        assert_rect_fast_eq_path(Rect::new(0.0, 0.0, 8.0, 8.0), "tile_aligned");
    }

    #[test]
    fn rect_one_pixel_wide() {
        assert_rect_fast_eq_path(Rect::new(5.0, 2.0, 6.0, 12.0), "one_pixel_wide");
    }

    #[test]
    fn rect_one_pixel_tall() {
        assert_rect_fast_eq_path(Rect::new(2.0, 5.0, 12.0, 6.0), "one_pixel_tall");
    }
}
