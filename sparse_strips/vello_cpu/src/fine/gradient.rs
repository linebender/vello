// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Rendering linear gradients.

use crate::fine::{COLOR_COMPONENTS, FineType, Painter, TILE_HEIGHT_COMPONENTS};
use vello_common::encode::{EncodedGradient, GradientLike, GradientLut};
use vello_common::kurbo::Point;

// This will be removed once this crate is ported to SIMD, so for now just duplicating those.

#[cfg(all(feature = "libm", not(feature = "std")))]
trait FloatExt {
    fn floor(self) -> f32;
    fn fract(self) -> f32;
    fn trunc(self) -> f32;
}

#[cfg(all(feature = "libm", not(feature = "std")))]
impl FloatExt for f32 {
    #[inline(always)]
    fn floor(self) -> f32 {
        libm::floorf(self)
    }
    #[inline(always)]
    fn fract(self) -> f32 {
        self - self.trunc()
    }
    #[inline(always)]
    fn trunc(self) -> f32 {
        libm::truncf(self)
    }
}

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

    pub(super) fn run<F: FineType>(mut self, target: &mut [F]) {
        let original_pos = self.cur_pos;
        let lut = F::get_lut(self.gradient);

        target
            .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
            .for_each(|column| {
                self.run_column(column, lut);
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

    fn run_column<F: FineType>(&mut self, col: &mut [F], lut: &GradientLut<F>) {
        let pad = self.gradient.pad;
        let extend = |val| extend(val, pad);
        let mut pos = self.cur_pos;

        for pixel in col.chunks_exact_mut(COLOR_COMPONENTS) {
            let t = extend(self.kind.cur_pos(pos));
            let idx = (t * lut.scale_factor()) as usize;
            pixel.copy_from_slice(&lut.get(idx));

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

pub(crate) fn extend(val: f32, pad: bool) -> f32 {
    if pad {
        #[allow(clippy::manual_clamp, reason = "better performance")]
        val.min(1.0).max(0.0)
    } else {
        (val - val.floor()).fract()
    }
}
