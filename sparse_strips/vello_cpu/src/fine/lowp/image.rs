// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::PosExt;
use crate::fine::common::image::{ImagePainterData, extend, sample};
use crate::fine::macros::u8x16_painter;
use vello_common::encode::EncodedImage;
use vello_common::fearless_simd::{Simd, SimdBase, f32x4, u8x16};
use vello_common::pixmap::Pixmap;
use vello_common::simd::element_wise_splat;
use vello_common::util::f32_to_u8;

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
        start_x: u16,
        start_y: u16,
    ) -> Self {
        let data = ImagePainterData::new(simd, image, pixmap, start_x, start_y);

        Self { data, simd }
    }
}

impl<S: Simd> Iterator for BilinearImagePainter<'_, S> {
    type Item = u8x16<S>;

    fn next(&mut self) -> Option<Self::Item> {
        let x_positions = f32x4::splat_pos(
            self.simd,
            self.data.cur_pos.x as f32,
            self.data.x_advances.0,
            self.data.y_advances.0,
        );

        let y_positions = f32x4::splat_pos(
            self.simd,
            self.data.cur_pos.y as f32,
            self.data.x_advances.1,
            self.data.y_advances.1,
        );

        // Note that this `fract` has different behavior for negative numbers than the normal,
        // one.
        #[inline(always)]
        fn fract<S: Simd>(val: f32x4<S>) -> f32x4<S> {
            val - val.floor()
        }

        let extend_x = |x_pos: f32x4<S>| {
            extend(
                self.simd,
                x_pos,
                self.data.image.sampler.x_extend,
                self.data.width,
                self.data.width_inv,
            )
        };

        let extend_y = |y_pos: f32x4<S>| {
            extend(
                self.simd,
                y_pos,
                self.data.image.sampler.y_extend,
                self.data.height,
                self.data.height_inv,
            )
        };

        let fx = f32_to_u8(element_wise_splat(
            self.simd,
            fract(x_positions + 0.5) * 256.0,
        ));
        let fy = f32_to_u8(element_wise_splat(
            self.simd,
            fract(y_positions + 0.5) * 256.0,
        ));
        let fx_inv = self.simd.widen_u8x16(u8x16::splat(self.simd, 255) - fx);
        let fy_inv = self.simd.widen_u8x16(u8x16::splat(self.simd, 255) - fy);

        let fx = self.simd.widen_u8x16(fx);
        let fy = self.simd.widen_u8x16(fy);

        let x_pos1 = extend_x(x_positions - 0.5);
        let x_pos2 = extend_x(x_positions + 0.5);
        let y_pos1 = extend_y(y_positions - 0.5);
        let y_pos2 = extend_y(y_positions + 0.5);

        let p00 = self
            .simd
            .widen_u8x16(sample(self.simd, &self.data, x_pos1, y_pos1));
        let p10 = self
            .simd
            .widen_u8x16(sample(self.simd, &self.data, x_pos2, y_pos1));
        let p01 = self
            .simd
            .widen_u8x16(sample(self.simd, &self.data, x_pos1, y_pos2));
        let p11 = self
            .simd
            .widen_u8x16(sample(self.simd, &self.data, x_pos2, y_pos2));

        let ip1 = (p00 * fx_inv + p10 * fx).shr(8);
        let ip2 = (p01 * fx_inv + p11 * fx).shr(8);
        let res = self.simd.narrow_u16x16((ip1 * fy_inv + ip2 * fy).shr(8));

        self.data.cur_pos += self.data.image.x_advance;

        Some(res)
    }
}

u8x16_painter!(BilinearImagePainter<'_, S>);
