// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Abstraction for generating strips from paths.

use crate::fearless_simd::Level;
use crate::flatten::{FlattenCtx, Line};
use crate::kurbo::{Affine, PathEl, Stroke};
use crate::peniko::Fill;
use crate::strip::{intersect, PathDataOwned, PathDataRef, PathDataMut, Strip};
use crate::tile::Tiles;
use crate::{flatten, strip};
use alloc::vec::Vec;

/// An object for easily generating strips for a filled/stroked path.
#[derive(Debug)]
pub struct StripGenerator {
    level: Level,
    global_alphas: Vec<u8>,
    temp_alphas: Vec<u8>,
    line_buf: Vec<Line>,
    flatten_ctx: FlattenCtx,
    tiles: Tiles,
    strip_buf1: Vec<Strip>,
    strip_buf2: Vec<Strip>,
    width: u16,
    height: u16,
}

impl StripGenerator {
    /// Create a new strip generator.
    pub fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            global_alphas: Vec::new(),
            temp_alphas: Vec::new(),
            level,
            line_buf: Vec::new(),
            tiles: Tiles::new(level),
            strip_buf1: Vec::new(),
            strip_buf2: Vec::new(),
            flatten_ctx: FlattenCtx::default(),
            width,
            height,
        }
    }

    /// Generate the strips for a filled path.
    pub fn generate_filled_path<'a>(
        &'a mut self,
        path: impl IntoIterator<Item=PathEl>,
        mut fill_rule: Fill,
        transform: Affine,
        aliasing_threshold: Option<u8>,
        clip_path: Option<PathDataRef<'_>>,
        func: impl FnOnce(&'a [Strip], &'a [u8]),
    ) {
        flatten::fill(
            self.level,
            path,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );
        
        if let Some(clip_path) = clip_path {
            self.make_strips(fill_rule, aliasing_threshold, true);
            
            let input_path = PathDataRef {
                strips: &self.strip_buf2,
                alphas: &self.temp_alphas,
                fill: fill_rule,
            };
            
            let target = PathDataMut {
                strips: &mut self.strip_buf1,
                alphas: &mut self.global_alphas,
                fill: &mut fill_rule,
            };

            intersect(
                self.level,
                clip_path,
                input_path,
                target
            );
        }   else {
            self.make_strips(fill_rule, aliasing_threshold, false);
        }

        func(&self.strip_buf1, &self.global_alphas);
    }

    /// Generate the strips for a stroked path.
    pub fn generate_stroked_path<'a>(
        &'a mut self,
        path: impl IntoIterator<Item = PathEl>,
        stroke: &Stroke,
        transform: Affine,
        aliasing_threshold: Option<u8>,
        clip_path: Option<PathDataRef<'_>>,
        func: impl FnOnce(&'a [Strip]),
    ) {
        let mut fill_rule = Fill::NonZero;
        
        flatten::stroke(
            self.level,
            path,
            stroke,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );

        if let Some(clip_path) = clip_path {
            self.make_strips(fill_rule, aliasing_threshold, true);

            let input_path = PathDataRef {
                strips: &self.strip_buf2,
                alphas: &self.temp_alphas,
                fill: fill_rule,
            };

            let target = PathDataMut {
                strips: &mut self.strip_buf1,
                alphas: &mut self.global_alphas,
                fill: &mut fill_rule,
            };

            intersect(
                self.level,
                clip_path,
                input_path,
                target
            );
        }   else {
            self.make_strips(fill_rule, aliasing_threshold, false);
        }

        func(&self.strip_buf1);
    }

    /// Return a reference to the current alpha buffer of the strip generator.
    pub fn alpha_buf(&self) -> &[u8] {
        &self.global_alphas
    }

    /// Extend the alpha buffer with the given alphas.
    pub fn extend_alpha_buf(&mut self, alphas: &[u8]) {
        self.global_alphas.extend_from_slice(alphas);
    }

    /// Set the alpha buffer.
    pub fn set_alpha_buf(&mut self, alpha_buf: Vec<u8>) {
        self.global_alphas = alpha_buf;
    }

    /// Take the alpha buffer and set it to an empty one.
    pub fn take_alpha_buf(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.global_alphas)
    }

    /// Swap the alpha buffer with the given one.
    pub fn replace_alpha_buf(&mut self, alphas: Vec<u8>) -> Vec<u8> {
        core::mem::replace(&mut self.global_alphas, alphas)
    }

    /// Reset the strip generator.
    pub fn reset(&mut self) {
        self.line_buf.clear();
        self.tiles.reset();
        self.global_alphas.clear();
        self.temp_alphas.clear();
        self.strip_buf1.clear();
        self.strip_buf2.clear();
    }

    fn make_strips(&mut self, fill_rule: Fill, aliasing_threshold: Option<u8>, temp: bool) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();
        
        let (alphas, strip_buf) = if temp {
            self.temp_alphas.clear();
            (&mut self.temp_alphas, &mut self.strip_buf2)
        }   else {
            (&mut self.global_alphas, &mut self.strip_buf1)
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

        assert!(!generator.line_buf.is_empty());
        assert!(!generator.strip_buf1.is_empty());
        assert!(!generator.global_alphas.is_empty());

        generator.reset();

        assert!(generator.line_buf.is_empty());
        assert!(generator.strip_buf1.is_empty());
        assert!(generator.global_alphas.is_empty());
    }
}
