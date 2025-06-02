// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering linear gradients.

use crate::fine::{COLOR_COMPONENTS, FineType, Painter, TILE_HEIGHT_COMPONENTS};
use vello_common::encode::{EncodedGradient, GradientLike, GradientRange};
use vello_common::kurbo::Point;

#[derive(Debug)]
pub(crate) struct GradientFiller<'a, T: GradientLike> {
    /// The current position that should be processed.
    cur_pos: Point,
    /// The underlying gradient.
    gradient: &'a EncodedGradient,
    /// The underlying gradient kind.
    kind: &'a T,
}

impl<'a, T: GradientLike> GradientFiller<'a, T> {
    pub(crate) fn new(
        gradient: &'a EncodedGradient,
        kind: &'a T,
        start_x: u16,
        start_y: u16,
    ) -> Self {
        Self {
            cur_pos: gradient.transform * Point::new(f64::from(start_x), f64::from(start_y)),
            gradient,
            kind,
        }
    }

    fn advance(&mut self, target_pos: f32) -> &GradientRange {
        let mut idx = 0;

        while target_pos > self.gradient.ranges[idx].x1 {
            idx += 1;
        }

        &self.gradient.ranges[idx]
    }

    pub(super) fn run<F: FineType>(mut self, target: &mut [F]) {
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

    fn run_column<F: FineType>(&mut self, col: &mut [F]) {
        let pad = self.gradient.pad;
        let extend = |val| extend(val, pad);
        let mut pos = self.cur_pos;

        for pixel in col.chunks_exact_mut(COLOR_COMPONENTS) {
            let t = extend(self.kind.cur_pos(pos));
            let range = self.advance(t);
            let bias = range.bias;

            for (comp_idx, comp) in pixel.iter_mut().enumerate() {
                *comp = F::from_normalized_f32(bias[comp_idx] + range.scale[comp_idx] * t);
            }

            pos += self.gradient.y_advance;
        }
    }

    fn run_undefined<F: FineType>(mut self, target: &mut [F]) {
        target
            .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
            .for_each(|column| {
                let mut pos = self.cur_pos;

                for pixel in column.chunks_exact_mut(COLOR_COMPONENTS) {
                    let actual_pos = pos;

                    if !self.kind.is_defined(actual_pos) {
                        pixel.copy_from_slice(&[F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
                    }

                    pos += self.gradient.y_advance;
                }

                self.cur_pos += self.gradient.x_advance;
            });
    }
}

impl<T: GradientLike> Painter for GradientFiller<'_, T> {
    fn paint<F: FineType>(self, target: &mut [F]) {
        self.run(target);
    }
}

pub(crate) fn extend(mut val: f32, pad: bool) -> f32 {
    if pad {
        // Gradient ranges are constructed such that values outside [0.0, 1.0] are accepted as well.
        val
    } else {
        while val < 0.0 {
            val += 1.0;
        }

        while val > 1.0 {
            val -= 1.0;
        }

        val
    }
}
