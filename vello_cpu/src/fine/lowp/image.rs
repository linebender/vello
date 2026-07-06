// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::common::image::{ImagePainterData, extend_mode, fract_floor, sample};
use crate::fine::macros::u8x16_painter;
use crate::fine::{PaintPositions, PlainPaintPositions, PosExt, PositionIterator};
use vello_common::encode::EncodedImage;
use vello_common::fearless_simd::{f32x4, prelude::*, u8x16, u16x16};
use vello_common::pixmap::Pixmap;
use vello_common::simd::element_wise_splat;
use vello_common::util::{Div255Ext, f32_to_u8, narrow, widen};

/// A faster bilinear image renderer for the u8 pipeline.
#[derive(Debug)]
pub(crate) struct BilinearImagePainter<'a, S: Simd> {
    data: ImagePainterData<'a, S>,
    simd: S,
}

impl<'a, S: Simd> BilinearImagePainter<'a, S> {
    pub(crate) fn new(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> Self {
        let data = ImagePainterData::new(simd, image, pixmap, start_x, start_y);

        Self { data, simd }
    }
}

impl<S: Simd> BilinearImagePainter<'_, S> {
    #[inline(always)]
    fn next(&self, positions: &PaintPositions<S, f32x4<S>>) -> u8x16<S> {
        let (x_positions, y_positions) = positions.current();

        let extend_x = |x_pos: f32x4<S>| {
            extend_mode(
                x_pos,
                self.data.image.sampler.x_extend,
                self.data.width,
                self.data.width_inv,
            )
        };

        let extend_y = |y_pos: f32x4<S>| {
            extend_mode(
                y_pos,
                self.data.image.sampler.y_extend,
                self.data.height,
                self.data.height_inv,
            )
        };

        let fx = f32_to_u8(element_wise_splat(
            self.simd,
            fract_floor(x_positions + 0.5).mul_add(255.0, 0.5),
        ));
        let fy = f32_to_u8(element_wise_splat(
            self.simd,
            fract_floor(y_positions + 0.5).mul_add(255.0, 0.5),
        ));

        let fx = widen(fx);
        let fy = widen(fy);
        let fx_inv = u16x16::splat(self.simd, 255) - fx;
        let fy_inv = u16x16::splat(self.simd, 255) - fy;

        let x_pos1 = extend_x(x_positions - 0.5);
        let x_pos2 = extend_x(x_positions + 0.5);
        let y_pos1 = extend_y(y_positions - 0.5);
        let y_pos2 = extend_y(y_positions + 0.5);

        let p00 = widen(sample(self.simd, &self.data, x_pos1, y_pos1));
        let p10 = widen(sample(self.simd, &self.data, x_pos2, y_pos1));
        let p01 = widen(sample(self.simd, &self.data, x_pos1, y_pos2));
        let p11 = widen(sample(self.simd, &self.data, x_pos2, y_pos2));

        let ip1 = (p00 * fx_inv + p10 * fx).div_255();
        let ip2 = (p01 * fx_inv + p11 * fx).div_255();

        narrow((ip1 * fy_inv + ip2 * fy).div_255())
    }
}

u8x16_painter!(
    BilinearImagePainter<'_, S>,
    painter,
    painter.data.positions::<f32x4<S>>(painter.simd)
);

/// A faster bilinear image renderer for axis-aligned images (no skew) in the u8 pipeline.
///
/// This is an optimized version of `BilinearImagePainter` that pre-computes y positions
/// and interpolation weights since they don't change when there's no skew.
#[derive(Debug)]
pub(crate) struct PlainBilinearImagePainter<'a, S: Simd> {
    data: ImagePainterData<'a, S>,
    simd: S,
    /// Pre-computed y sample positions (top row for bilinear grid)
    y_pos1: f32x4<S>,
    /// Pre-computed y sample positions (bottom row for bilinear grid)
    y_pos2: f32x4<S>,
    /// Pre-computed y interpolation weight
    fy: u16x16<S>,
    /// Pre-computed inverse y interpolation weight
    fy_inv: u16x16<S>,
}

impl<'a, S: Simd> PlainBilinearImagePainter<'a, S> {
    pub(crate) fn new(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: f64,
        start_y: f64,
    ) -> Self {
        let data = ImagePainterData::new(simd, image, pixmap, start_x, start_y);

        simd.vectorize(
            #[inline(always)]
            || {
                // For axis-aligned images, y doesn't change across the strip
                let y_positions = f32x4::splat_pos(
                    simd,
                    data.cur_pos.y as f32,
                    data.x_advances.1,
                    data.y_advances.1,
                );

                // Pre-compute y extend positions
                let y_pos1 = extend_mode(
                    y_positions - 0.5,
                    image.sampler.y_extend,
                    data.height,
                    data.height_inv,
                );
                let y_pos2 = extend_mode(
                    y_positions + 0.5,
                    image.sampler.y_extend,
                    data.height,
                    data.height_inv,
                );

                // Pre-compute y interpolation weights
                let fy = f32_to_u8(element_wise_splat(
                    simd,
                    fract_floor(y_positions + 0.5).mul_add(255.0, 0.5),
                ));
                let fy = widen(fy);
                let fy_inv = u16x16::splat(simd, 255) - fy;

                Self {
                    data,
                    y_pos1,
                    y_pos2,
                    fy,
                    fy_inv,
                    simd,
                }
            },
        )
    }
}

impl<S: Simd> PlainBilinearImagePainter<'_, S> {
    #[inline(always)]
    fn next(&self, positions: &PlainPaintPositions<S, f32x4<S>>) -> u8x16<S> {
        let x_positions = positions.current();
        let x_minus_half = x_positions - 0.5;
        let x_plus_half = x_positions + 0.5;

        // Only x needs to be extended per-iteration
        let x_pos1 = extend_mode(
            x_minus_half,
            self.data.image.sampler.x_extend,
            self.data.width,
            self.data.width_inv,
        );
        let x_pos2 = extend_mode(
            x_plus_half,
            self.data.image.sampler.x_extend,
            self.data.width,
            self.data.width_inv,
        );

        // Compute x interpolation weights
        let fx = f32_to_u8(element_wise_splat(
            self.simd,
            fract_floor(x_plus_half).mul_add(255.0, 0.5),
        ));
        let fx = widen(fx);
        let fx_inv = u16x16::splat(self.simd, 255) - fx;

        // Sample the 4 corners using pre-computed y positions
        let p00 = widen(sample(self.simd, &self.data, x_pos1, self.y_pos1));
        let p10 = widen(sample(self.simd, &self.data, x_pos2, self.y_pos1));
        let p01 = widen(sample(self.simd, &self.data, x_pos1, self.y_pos2));
        let p11 = widen(sample(self.simd, &self.data, x_pos2, self.y_pos2));

        // Bilinear interpolation
        let ip1 = (p00 * fx_inv + p10 * fx).div_255();
        let ip2 = (p01 * fx_inv + p11 * fx).div_255();

        narrow((ip1 * self.fy_inv + ip2 * self.fy).div_255())
    }
}

u8x16_painter!(
    PlainBilinearImagePainter<'_, S>,
    painter,
    painter.data.plain_positions::<f32x4<S>>(painter.simd)
);
