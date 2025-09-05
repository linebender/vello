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

/// An object for easily generating strips for a filled/stroked path.
#[derive(Debug)]
pub struct StripGenerator {
    level: Level,
    alphas: Vec<u8>,
    line_buf: Vec<Line>,
    flatten_ctx: FlattenCtx,
    tiles: Tiles,
    strip_buf: Vec<Strip>,
    width: u16,
    height: u16,
}

impl StripGenerator {
    /// Create a new strip generator.
    pub fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            alphas: Vec::new(),
            level,
            line_buf: Vec::new(),
            tiles: Tiles::new(level),
            strip_buf: Vec::new(),
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
        func: impl FnOnce(&'a [Strip]),
    ) {
        flatten::fill(
            self.level,
            path,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );
        self.make_strips(fill_rule, aliasing_threshold);
        func(&mut self.strip_buf);
    }

    /// Generate the strips for a stroked path.
    pub fn generate_stroked_path<'a>(
        &'a mut self,
        path: impl IntoIterator<Item = PathEl>,
        stroke: &Stroke,
        transform: Affine,
        aliasing_threshold: Option<u8>,
        func: impl FnOnce(&'a [Strip]),
    ) {
        flatten::stroke(
            self.level,
            path,
            stroke,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );
        self.make_strips(Fill::NonZero, aliasing_threshold);
        func(&mut self.strip_buf);
    }

    /// Return a reference to the current alpha buffer of the strip generator.
    pub fn alpha_buf(&self) -> &[u8] {
        &self.alphas
    }

    /// Extend the alpha buffer with the given alphas.
    pub fn extend_alpha_buf(&mut self, alphas: &[u8]) {
        self.alphas.extend_from_slice(alphas);
    }

    /// Set the alpha buffer.
    pub fn set_alpha_buf(&mut self, alpha_buf: Vec<u8>) {
        self.alphas = alpha_buf;
    }

    /// Take the alpha buffer and set it to an empty one.
    pub fn take_alpha_buf(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.alphas)
    }

    /// Swap the alpha buffer with the given one.
    pub fn replace_alpha_buf(&mut self, alphas: Vec<u8>) -> Vec<u8> {
        core::mem::replace(&mut self.alphas, alphas)
    }

    /// Reset the strip generator.
    pub fn reset(&mut self) {
        self.line_buf.clear();
        self.tiles.reset();
        self.alphas.clear();
        self.strip_buf.clear();
    }

    fn make_strips(&mut self, fill_rule: Fill, aliasing_threshold: Option<u8>) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();
        strip::render(
            self.level,
            &self.tiles,
            &mut self.strip_buf,
            &mut self.alphas,
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
            |_| {},
        );

        assert!(!generator.line_buf.is_empty());
        assert!(!generator.strip_buf.is_empty());
        assert!(!generator.alphas.is_empty());

        generator.reset();

        assert!(generator.line_buf.is_empty());
        assert!(generator.strip_buf.is_empty());
        assert!(generator.alphas.is_empty());
    }
}
