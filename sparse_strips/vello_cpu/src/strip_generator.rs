// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::kurbo::{Affine, BezPath, Stroke};
use crate::peniko::Fill;
use alloc::vec::Vec;
use vello_common::flatten::Line;
use vello_common::strip::Strip;
use vello_common::tile::Tiles;
use vello_common::{flatten, strip};

#[derive(Debug)]
pub(crate) struct StripGenerator {
    alphas: Vec<u8>,
    line_buf: Vec<Line>,
    tiles: Tiles,
    strip_buf: Vec<Strip>,
    width: u16,
    height: u16,
}

impl StripGenerator {
    pub(crate) fn new(width: u16, height: u16) -> Self {
        Self {
            alphas: Vec::new(),
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
            &self.tiles,
            &mut self.strip_buf,
            &mut self.alphas,
            fill_rule,
            &self.line_buf,
        );
    }
}
