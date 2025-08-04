// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::kurbo::{Affine, PathEl, Stroke};
use crate::peniko::Fill;
use alloc::vec::Vec;
use vello_common::fearless_simd::Level;
use vello_common::flatten::{FlattenCtx, Line};
use vello_common::strip::Strip;
use vello_common::tile::Tiles;
use vello_common::{flatten, strip};

#[derive(Debug)]
pub(crate) struct StripGenerator {
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
    pub(crate) fn new(width: u16, height: u16, level: Level) -> Self {
        Self {
            alphas: Vec::new(),
            level,
            line_buf: Vec::new(),
            tiles: Tiles::new(),
            strip_buf: Vec::new(),
            flatten_ctx: FlattenCtx::default(),
            width,
            height,
        }
    }

    pub(crate) fn generate_filled_path<'a>(
        &'a mut self,
        path: impl IntoIterator<Item = PathEl>,
        fill_rule: Fill,
        transform: Affine,
        anti_alias: bool,
        func: impl FnOnce(&'a [Strip]),
    ) {
        flatten::fill(
            self.level,
            path,
            transform,
            &mut self.line_buf,
            &mut self.flatten_ctx,
        );
        self.make_strips(fill_rule, anti_alias);
        func(&mut self.strip_buf);
    }

    pub(crate) fn generate_stroked_path<'a>(
        &'a mut self,
        path: impl IntoIterator<Item = PathEl>,
        stroke: &Stroke,
        transform: Affine,
        anti_alias: bool,
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
        self.make_strips(Fill::NonZero, anti_alias);
        func(&mut self.strip_buf);
    }

    pub(crate) fn alpha_buf(&self) -> &[u8] {
        &self.alphas
    }

    #[cfg(feature = "multithreading")]
    pub(crate) fn set_alpha_buf(&mut self, alpha_buf: Vec<u8>) {
        self.alphas = alpha_buf;
    }

    #[cfg(feature = "multithreading")]
    pub(crate) fn take_alpha_buf(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.alphas)
    }

    pub(crate) fn reset(&mut self) {
        self.line_buf.clear();
        self.tiles.reset();
        self.alphas.clear();
        self.strip_buf.clear();
    }

    fn make_strips(&mut self, fill_rule: Fill, anti_alias: bool) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();
        strip::render(
            self.level,
            &self.tiles,
            &mut self.strip_buf,
            &mut self.alphas,
            fill_rule,
            anti_alias,
            &self.line_buf,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::kurbo::{Affine, Rect, Shape};
    use crate::strip_generator::StripGenerator;
    use vello_common::fearless_simd::Level;
    use vello_common::peniko::Fill;

    #[test]
    fn reset_strip_generator() {
        let mut generator = StripGenerator::new(100, 100, Level::fallback());
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);

        generator.generate_filled_path(
            rect.to_path(0.1),
            Fill::NonZero,
            Affine::IDENTITY,
            true,
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
