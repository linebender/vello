// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Abstraction for generating strips from paths.

use crate::fearless_simd::Level;
use crate::flatten::{FlattenCtx, Line};
use crate::kurbo::{Affine, PathEl, Stroke};
use crate::peniko::Fill;
use crate::strip::{PathDataMut, PathDataOwned, PathDataRef, Strip, intersect};
use crate::tile::Tiles;
use crate::{flatten, strip};
use alloc::sync::Arc;
use alloc::vec::Vec;

/// An object for easily generating strips for a filled/stroked path.
#[derive(Debug)]
pub struct StripGenerator {
    level: Level,
    line_buf: Vec<Line>,
    flatten_ctx: FlattenCtx,
    tiles: Tiles,
    /// The main global alpha buffer.
    main_alpha_buf: Vec<u8>,
    /// The main strip buffer.
    main_strip_buf: Vec<Strip>,
    /// A temporary alpha buffer used for storing intermediate results when clipping.
    temp_alpha_buf: Vec<u8>,
    /// A temporary strip buffer used for storing intermediate results when clipping.
    temp_strip_buf: Vec<Strip>,
    width: u16,
    height: u16,
}

impl StripGenerator {
    /// Create a new strip generator.
    pub fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            main_alpha_buf: Vec::new(),
            temp_alpha_buf: Vec::new(),
            level,
            line_buf: Vec::new(),
            tiles: Tiles::new(level),
            main_strip_buf: Vec::new(),
            temp_strip_buf: Vec::new(),
            flatten_ctx: FlattenCtx::default(),
            width,
            height,
        }
    }

    /// Generate the strips for a filled path.
    pub fn generate_filled_path<'a>(
        &'a mut self,
        path: impl IntoIterator<Item = PathEl>,
        fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
        clip_path: Option<Arc<PathDataOwned>>,
        func: impl FnOnce(&'a [Strip], &'a [u8]),
    ) {
        flatten::fill(
            self.level,
            path,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );

        self.generate_with_clip_path(fill_rule, aliasing_threshold, clip_path);

        func(&self.main_strip_buf, &self.main_alpha_buf);
    }

    /// Generate the strips for a stroked path.
    pub fn generate_stroked_path<'a>(
        &'a mut self,
        path: impl IntoIterator<Item = PathEl>,
        stroke: &Stroke,
        transform: Affine,
        aliasing_threshold: Option<u8>,
        clip_path: Option<Arc<PathDataOwned>>,
        func: impl FnOnce(&'a [Strip]),
    ) {
        let fill_rule = Fill::NonZero;

        flatten::stroke(
            self.level,
            path,
            stroke,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );

        self.generate_with_clip_path(fill_rule, aliasing_threshold, clip_path);

        func(&self.main_strip_buf);
    }

    fn generate_with_clip_path(
        &mut self,
        mut fill_rule: Fill,
        aliasing_threshold: Option<u8>,
        clip_path: Option<Arc<PathDataOwned>>,
    ) {
        if let Some(clip_path) = clip_path {
            self.make_strips(fill_rule, aliasing_threshold, true);

            let input_path = PathDataRef {
                strips: &self.temp_strip_buf,
                alphas: &self.temp_alpha_buf,
                fill: fill_rule,
            };

            let target = PathDataMut {
                strips: &mut self.main_strip_buf,
                alphas: &mut self.main_alpha_buf,
                fill: &mut fill_rule,
            };

            intersect(self.level, clip_path.as_path_data_ref(), input_path, target);
        } else {
            self.make_strips(fill_rule, aliasing_threshold, false);
        }
    }

    /// Return a reference to the current alpha buffer of the strip generator.
    pub fn alpha_buf(&self) -> &[u8] {
        &self.main_alpha_buf
    }

    /// Extend the alpha buffer with the given alphas.
    pub fn extend_alpha_buf(&mut self, alphas: &[u8]) {
        self.main_alpha_buf.extend_from_slice(alphas);
    }

    /// Set the alpha buffer.
    pub fn set_alpha_buf(&mut self, alpha_buf: Vec<u8>) {
        self.main_alpha_buf = alpha_buf;
    }

    /// Take the alpha buffer and set it to an empty one.
    pub fn take_alpha_buf(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.main_alpha_buf)
    }

    /// Swap the alpha buffer with the given one.
    pub fn replace_alpha_buf(&mut self, alphas: Vec<u8>) -> Vec<u8> {
        core::mem::replace(&mut self.main_alpha_buf, alphas)
    }

    /// Reset the strip generator.
    pub fn reset(&mut self) {
        self.line_buf.clear();
        self.tiles.reset();
        self.main_alpha_buf.clear();
        self.temp_alpha_buf.clear();
        self.main_strip_buf.clear();
        self.temp_strip_buf.clear();
    }

    fn make_strips(&mut self, fill_rule: Fill, aliasing_threshold: Option<u8>, temp: bool) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();

        let (alphas, strip_buf) = if temp {
            self.temp_alpha_buf.clear();
            self.temp_strip_buf.clear();

            (&mut self.temp_alpha_buf, &mut self.temp_strip_buf)
        } else {
            (&mut self.main_alpha_buf, &mut self.main_strip_buf)
        };

        strip::render(
            self.level,
            &self.tiles,
            strip_buf,
            alphas,
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
    use crate::strip::Strip;
    use crate::strip_generator::StripGenerator;

    #[test]
    fn reset_strip_generator() {
        let mut generator = StripGenerator::new(100, 100, Level::fallback());
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);

        generator.generate_filled_path(
            rect.to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            None,
            None,
            |_, _| {},
        );

        generator.temp_alpha_buf.extend_from_slice(&[0; 16]);
        generator.temp_strip_buf.push(Strip {
            x: 0,
            y: 0,
            alpha_idx: 0,
            winding: 0,
        });

        assert!(!generator.line_buf.is_empty());
        assert!(!generator.main_strip_buf.is_empty());
        assert!(!generator.temp_strip_buf.is_empty());
        assert!(!generator.main_alpha_buf.is_empty());
        assert!(!generator.temp_alpha_buf.is_empty());

        generator.reset();

        assert!(generator.line_buf.is_empty());
        assert!(generator.main_strip_buf.is_empty());
        assert!(generator.temp_strip_buf.is_empty());
        assert!(generator.main_alpha_buf.is_empty());
        assert!(generator.main_alpha_buf.is_empty());
    }
}
