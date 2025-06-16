// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::kurbo::{Affine, BezPath, Stroke};
use crate::peniko::Fill;
use alloc::vec::Vec;
use vello_common::fearless_simd::Level;
use vello_common::flatten::Line;
use vello_common::strip::Strip;
use vello_common::tile::Tiles;
use vello_common::{flatten, strip};

#[derive(Debug)]
pub(crate) struct StripGenerator {
    level: Level,
    alphas: Vec<u8>,
    line_buf: Vec<Line>,
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
            width,
            height,
        }
    }

    pub(crate) fn generate_filled_path<'a>(
        &'a mut self,
        path: &BezPath,
        fill_rule: Fill,
        transform: Affine,
        func: impl FnOnce(&'a [Strip]),
    ) {
        flatten::fill(path, transform, &mut self.line_buf);
        self.make_strips(fill_rule);
        func(&mut self.strip_buf);
    }

    pub(crate) fn generate_stroked_path<'a>(
        &'a mut self,
        path: &BezPath,
        stroke: &Stroke,
        transform: Affine,
        func: impl FnOnce(&'a [Strip]),
    ) {
        flatten::stroke(path, stroke, transform, &mut self.line_buf);
        self.make_strips(Fill::NonZero);
        func(&mut self.strip_buf);
    }

    pub(crate) fn alpha_buf(&self) -> &[u8] {
        &self.alphas
    }

    pub(crate) fn reset(&mut self) {
        self.line_buf.clear();
        self.tiles.reset();
        self.alphas.clear();
        self.strip_buf.clear();
    }

    fn make_strips(&mut self, fill_rule: Fill) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();
        strip::render(
            self.level,
            &self.tiles,
            &mut self.strip_buf,
            &mut self.alphas,
            fill_rule,
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

        generator.generate_filled_path(&rect.to_path(0.1), Fill::NonZero, Affine::IDENTITY, |_| {});

        assert!(!generator.line_buf.is_empty());
        assert!(!generator.strip_buf.is_empty());
        assert!(!generator.alphas.is_empty());

        generator.reset();

        assert!(generator.line_buf.is_empty());
        assert!(generator.strip_buf.is_empty());
        assert!(generator.alphas.is_empty());
    }
}
