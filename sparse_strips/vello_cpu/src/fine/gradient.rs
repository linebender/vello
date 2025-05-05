// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering linear gradients.

use crate::fine::{COLOR_COMPONENTS, Painter, TILE_HEIGHT_COMPONENTS};
use vello_common::encode::{EncodedGradient, GradientLike, GradientRange};
use vello_common::kurbo::Point;

#[derive(Debug)]
pub(crate) struct GradientFiller<'a, T: GradientLike> {
    /// The current position that should be processed.
    cur_pos: Point,
    /// The index of the current range.
    range_idx: usize,
    /// The underlying gradient.
    gradient: &'a EncodedGradient,
    /// The underlying gradient kind.
    kind: &'a T,
    /// The current gradient range (pointed to by `range_idx`).
    cur_range: &'a GradientRange,
}

impl<'a, T: GradientLike> GradientFiller<'a, T> {
    pub(crate) fn new(
        gradient: &'a EncodedGradient,
        kind: &'a T,
        start_x: u16,
        start_y: u16,
    ) -> Self {
        Self {
            cur_pos: gradient.transform * Point::new(start_x as f64, start_y as f64),
            range_idx: 0,
            cur_range: &gradient.ranges[0],
            gradient,
            kind,
        }
    }

    fn advance(&mut self, target_pos: f32) {
        while target_pos > self.cur_range.x1 || target_pos < self.cur_range.x0 {
            if self.range_idx == 0 {
                self.range_idx = self.gradient.ranges.len() - 1;
            } else {
                self.range_idx -= 1;
            }

            self.cur_range = &self.gradient.ranges[self.range_idx];
        }
    }

    pub(super) fn run(mut self, target: &mut [u8]) {
        let original_pos = self.cur_pos;

        target
            .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
            .for_each(|column| {
                self.run_column(column);
                self.cur_pos += self.gradient.x_advance;
            });

        // Radial gradients can have positions that are undefined and thus shouldn't be
        // painted at all. Checking for this inside of the main filling logic would be
        // an unnecessary overhead for the general case, while this is really just an edge
        // case. Because of this, in the first run we will fill it using a dummy color, and
        // in case the gradient might have undefined locations, we do another run over
        // the buffer and override the positions with a transparent fill. This way, we need
        // 2x as long to handle such gradients, but for the common case we don't incur any
        // additional overhead.
        if self.kind.has_undefined() {
            self.cur_pos = original_pos;
            self.run_undefined(target);
        }
    }

    fn run_column(&mut self, col: &mut [u8]) {
        let pad = self.gradient.pad;
        let extend = |val| extend(val, pad, self.gradient.clamp_range);
        let mut pos = self.cur_pos;

        for pixel in col.chunks_exact_mut(COLOR_COMPONENTS) {
            let dist = extend(self.kind.cur_pos(&pos));
            self.advance(dist);
            let range = self.cur_range;

            for (comp_idx, comp) in pixel.iter_mut().enumerate() {
                let factor = (range.factors[comp_idx] * (dist - range.x0) + 0.5) as i16;

                *comp = (range.c0[comp_idx] as i16 + factor) as u8;
            }

            pos += self.gradient.y_advance;
        }
    }

    fn run_undefined(mut self, target: &mut [u8]) {
        target
            .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
            .for_each(|column| {
                let mut pos = self.cur_pos;

                for pixel in column.chunks_exact_mut(COLOR_COMPONENTS) {
                    let actual_pos = pos;

                    if !self.kind.is_defined(&actual_pos) {
                        pixel.copy_from_slice(&[0, 0, 0, 0]);
                    }

                    pos += self.gradient.y_advance;
                }

                self.cur_pos += self.gradient.x_advance;
            });
    }
}

impl<T: GradientLike> Painter for GradientFiller<'_, T> {
    fn paint(self, target: &mut [u8]) {
        self.run(target);
    }
}

pub(crate) fn extend(mut val: f32, pad: bool, clamp_range: (f32, f32)) -> f32 {
    let start = clamp_range.0;
    let end = clamp_range.1;

    if pad {
        val
    } else {
        // Avoid using modulo here because it's slower.

        while val < start {
            val += end - start;
        }

        while val > end {
            val -= end - start;
        }

        val
    }
}
