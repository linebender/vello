// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::macros::{f32x16_painter, u8x16_painter};
use crate::fine::{PosExt, Splat4thExt, u8_to_f32};
use crate::kurbo::Point;
use vello_common::encode::EncodedImage;
use vello_common::fearless_simd::{Bytes, Simd, SimdBase, SimdFloat, f32x4, f32x16, u8x16, u32x4};
use vello_common::pixmap::Pixmap;
use vello_common::simd::element_wise_splat;

/// A painter for nearest-neighbor images with no skewing.
#[derive(Debug)]
pub(crate) struct PlainNNImagePainter<'a, S: Simd> {
    data: ImagePainterData<'a, S>,
    y_positions: f32x4<S>,
    cur_x_pos: f32x4<S>,
    advance: f32,
    simd: S,
}

impl<'a, S: Simd> PlainNNImagePainter<'a, S> {
    pub(crate) fn new(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: u16,
        start_y: u16,
    ) -> Self {
        let data = ImagePainterData::new(simd, image, pixmap, start_x, start_y);

        let y_positions = extend(
            simd,
            f32x4::splat_pos(
                simd,
                data.cur_pos.y as f32,
                data.x_advances.1,
                data.y_advances.1,
            ),
            image.sampler.y_extend,
            data.height,
            data.height_inv,
        );

        let cur_x_pos = f32x4::splat_pos(
            simd,
            data.cur_pos.x as f32,
            data.x_advances.0,
            data.y_advances.0,
        );

        Self {
            data,
            advance: image.x_advance.x as f32,
            y_positions,
            cur_x_pos,
            simd,
        }
    }
}

impl<S: Simd> Iterator for PlainNNImagePainter<'_, S> {
    type Item = u8x16<S>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let x_pos = extend(
            self.simd,
            self.cur_x_pos,
            self.data.image.sampler.x_extend,
            self.data.width,
            self.data.width_inv,
        );

        let samples = sample(self.simd, &self.data, x_pos, self.y_positions);

        self.cur_x_pos += self.advance;

        Some(samples)
    }
}

u8x16_painter!(PlainNNImagePainter<'_, S>);

/// A painter for nearest-neighbor images with arbitrary transforms.
#[derive(Debug)]
pub(crate) struct NNImagePainter<'a, S: Simd> {
    data: ImagePainterData<'a, S>,
    simd: S,
}

impl<'a, S: Simd> NNImagePainter<'a, S> {
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

impl<S: Simd> Iterator for NNImagePainter<'_, S> {
    type Item = u8x16<S>;

    fn next(&mut self) -> Option<Self::Item> {
        let x_positions = extend(
            self.simd,
            f32x4::splat_pos(
                self.simd,
                self.data.cur_pos.x as f32,
                self.data.x_advances.0,
                self.data.y_advances.0,
            ),
            self.data.image.sampler.x_extend,
            self.data.width,
            self.data.width_inv,
        );

        let y_positions = extend(
            self.simd,
            f32x4::splat_pos(
                self.simd,
                self.data.cur_pos.y as f32,
                self.data.x_advances.1,
                self.data.y_advances.1,
            ),
            self.data.image.sampler.y_extend,
            self.data.height,
            self.data.height_inv,
        );

        let samples = sample(self.simd, &self.data, x_positions, y_positions);

        self.data.cur_pos += self.data.image.x_advance;

        Some(samples)
    }
}

u8x16_painter!(NNImagePainter<'_, S>);

/// A painter for images with bilinear or bicubic filtering.
///
/// The painter is generic over sampler quality using the const-generic `QUALITY` parameter.
///
/// - Set `QUALITY` to `1` for bilinear sampling; or
/// - set `QUALITY` to `2` for bicubic sampling.
///
/// These values for `QUALITY` are the same numeric values as defined by
/// [`crate::peniko::ImageQuality`].
#[derive(Debug)]
pub(crate) struct FilteredImagePainter<'a, S: Simd, const QUALITY: u8> {
    data: ImagePainterData<'a, S>,
    simd: S,
}

impl<'a, S: Simd, const QUALITY: u8> FilteredImagePainter<'a, S, QUALITY> {
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

impl<S: Simd, const QUALITY: u8> Iterator for FilteredImagePainter<'_, S, QUALITY> {
    type Item = f32x16<S>;

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

        // We have two versions of filtering: `Medium` (bilinear filtering) and
        // `High` (bicubic filtering).

        // In bilinear filtering, we sample the pixels of the rectangle that spans the
        // locations (-0.5, -0.5) and (0.5, 0.5), and weight them by the fractional
        // x/y position using simple linear interpolation in both dimensions.
        // In bicubic filtering, we instead span a 4x4 grid around the
        // center of the location we are sampling, and sample those points
        // using a cubic filter to weight each location's contribution.

        let x_fract = fract_floor(x_positions + 0.5);
        let y_fract = fract_floor(y_positions + 0.5);

        let mut interpolated_color = f32x16::splat(self.simd, 0.0);

        let sample = |x_pos: f32x4<S>, y_pos: f32x4<S>| {
            u8_to_f32(sample(self.simd, &self.data, x_pos, y_pos))
        };

        macro_rules! extend_x {
            ($idx:expr,$offsets:expr) => {
                extend(
                    self.simd,
                    x_positions + $offsets[$idx],
                    self.data.image.sampler.y_extend,
                    self.data.width,
                    self.data.width_inv,
                )
            };
        }

        macro_rules! extend_y {
            ($idx:expr,$offsets:expr) => {
                extend(
                    self.simd,
                    y_positions + $offsets[$idx],
                    self.data.image.sampler.y_extend,
                    self.data.height,
                    self.data.height_inv,
                )
            };
        }

        match QUALITY {
            // medium quality: bilinear
            1 => {
                // <https://github.com/google/skia/blob/220738774f7a0ce4a6c7bd17519a336e5e5dea5b/src/opts/SkRasterPipeline_opts.h#L5039-L5078>
                let cx = [1.0 - x_fract, x_fract];
                let cy = [1.0 - y_fract, y_fract];

                // Note that the sum of all cx*cy combinations also yields 1.0 again
                // (modulo some floating point number impreciseness), ensuring the
                // colors stay in range.

                const OFFSETS: [f32; 2] = [-0.5, 0.5];

                let x_positions = [extend_x!(0, OFFSETS), extend_x!(1, OFFSETS)];

                let y_positions = [extend_y!(0, OFFSETS), extend_y!(1, OFFSETS)];

                // We sample the corners of rectangle that covers our current position.
                for x_idx in 0..2 {
                    let x_positions = x_positions[x_idx];

                    for y_idx in 0..2 {
                        let y_positions = y_positions[y_idx];
                        let color_sample = sample(x_positions, y_positions);
                        let w = element_wise_splat(self.simd, cx[x_idx] * cy[y_idx]);

                        interpolated_color = w.madd(color_sample, interpolated_color);
                    }
                }

                interpolated_color *= f32x16::splat(self.simd, 1.0 / 255.0);
            }
            // high quality: bicubic
            2 => {
                // Compare to <https://github.com/google/skia/blob/84ff153b0093fc83f6c77cd10b025c06a12c5604/src/opts/SkRasterPipeline_opts.h#L5030-L5075>.
                let cx = weights(self.simd, x_fract);
                let cy = weights(self.simd, y_fract);

                const OFFSETS: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];

                let x_positions = [
                    extend_x!(0, OFFSETS),
                    extend_x!(1, OFFSETS),
                    extend_x!(2, OFFSETS),
                    extend_x!(3, OFFSETS),
                ];

                let y_positions = [
                    extend_y!(0, OFFSETS),
                    extend_y!(1, OFFSETS),
                    extend_y!(2, OFFSETS),
                    extend_y!(3, OFFSETS),
                ];

                // Note in particular that it is guaranteed that, similarly to bilinear filtering,
                // the sum of all cx*cy is 1 (modulo some edge cases).

                // We sample the 4x4 grid around the position we are currently looking at.
                for x_idx in 0..4 {
                    let x_positions = x_positions[x_idx];
                    for y_idx in 0..4 {
                        let y_positions = y_positions[y_idx];

                        let color_sample = sample(x_positions, y_positions);
                        let w = element_wise_splat(self.simd, cx[x_idx] * cy[y_idx]);

                        interpolated_color = w.madd(color_sample, interpolated_color);
                    }
                }

                interpolated_color *= f32x16::splat(self.simd, 1.0 / 255.0);

                let alphas = interpolated_color.splat_4th();

                // Due to the nature of the cubic filter, it can happen in certain situations
                // that one of the color components ends up with a higher value than the
                // alpha component, which isn't permissible because the color is
                // premultiplied and would lead to overflows when doing source over
                // compositing with u8-based values. Because of this, we need to clamp
                // to the alpha value.
                interpolated_color = interpolated_color
                    .min(f32x16::splat(self.simd, 1.0))
                    .max(f32x16::splat(self.simd, 0.0))
                    .min(alphas);
            }
            _ => panic!(
                "Unknown value for `FilteredImagePainter`'s const-generic `QUALITY` parameter. Expected `1` for bilinear or `2` for bicubic, got: `{QUALITY}`."
            ),
        }

        self.data.cur_pos += self.data.image.x_advance;

        Some(interpolated_color)
    }
}

// Bilinear
f32x16_painter!(FilteredImagePainter<'_, S, 1>);
// Bicubic
f32x16_painter!(FilteredImagePainter<'_, S, 2>);

/// Computes the positive fractional part of a value: `val - val.floor()`.
///
/// Unlike `f32::fract()`, this always returns a value in [0, 1),
/// even for negative inputs.
#[inline(always)]
pub(crate) fn fract_floor<S: Simd>(val: f32x4<S>) -> f32x4<S> {
    val - val.floor()
}

/// Common data used by different image painters
#[derive(Debug)]
pub(crate) struct ImagePainterData<'a, S: Simd> {
    pub(crate) cur_pos: Point,
    pub(crate) image: &'a EncodedImage,
    pub(crate) pixmap: &'a Pixmap,
    pub(crate) x_advances: (f32, f32),
    pub(crate) y_advances: (f32, f32),
    pub(crate) height: f32x4<S>,
    pub(crate) height_inv: f32x4<S>,
    pub(crate) width: f32x4<S>,
    pub(crate) width_inv: f32x4<S>,
    pub(crate) width_u32: u32x4<S>,
}

impl<'a, S: Simd> ImagePainterData<'a, S> {
    pub(crate) fn new(
        simd: S,
        image: &'a EncodedImage,
        pixmap: &'a Pixmap,
        start_x: u16,
        start_y: u16,
    ) -> Self {
        let width = pixmap.width() as f32;
        let height = pixmap.height() as f32;
        let start_pos = image.transform * Point::new(f64::from(start_x), f64::from(start_y));

        let width_inv = f32x4::splat(simd, 1.0 / width);
        let height_inv = f32x4::splat(simd, 1.0 / height);
        let width = f32x4::splat(simd, width);
        let width_u32 = u32x4::splat(simd, pixmap.width() as u32);
        let height = f32x4::splat(simd, height);

        let x_advances = (image.x_advance.x as f32, image.x_advance.y as f32);
        let y_advances = (image.y_advance.x as f32, image.y_advance.y as f32);

        Self {
            cur_pos: start_pos,
            pixmap,
            x_advances,
            y_advances,
            image,
            width,
            height,
            width_u32,
            width_inv,
            height_inv,
        }
    }
}

#[inline(always)]
pub(crate) fn sample<S: Simd>(
    simd: S,
    data: &ImagePainterData<'_, S>,
    x_positions: f32x4<S>,
    y_positions: f32x4<S>,
) -> u8x16<S> {
    let idx = x_positions.to_int::<u32x4<S>>() + y_positions.to_int::<u32x4<S>>() * data.width_u32;

    u32x4::from_slice(
        simd,
        &[
            data.pixmap.sample_idx(idx[0]).to_u32(),
            data.pixmap.sample_idx(idx[1]).to_u32(),
            data.pixmap.sample_idx(idx[2]).to_u32(),
            data.pixmap.sample_idx(idx[3]).to_u32(),
        ],
    )
    .to_bytes()
}

#[inline(always)]
pub(crate) fn extend<S: Simd>(
    simd: S,
    val: f32x4<S>,
    extend: crate::peniko::Extend,
    max: f32x4<S>,
    inv_max: f32x4<S>,
) -> f32x4<S> {
    match extend {
        // Note that max should be exclusive, so subtract one to enforce that.
        // Since the maximum image dimensions we support is u16::MAX, subtracting 1 in f32
        // is enough to ensure that all numbers are subtracted correctly.
        crate::peniko::Extend::Pad => val.min(max - 1.0).max(f32x4::splat(simd, 0.0)),
        crate::peniko::Extend::Repeat => {
            // floor := (val * inv_max).floor() * max is the nearest multiple of `max` below val.
            max.madd(-(val * inv_max).floor(), val)
                // In certain edge cases, we might still end up with a higher number.
                .min(max - 1.0)
        }
        // <https://github.com/google/skia/blob/220738774f7a0ce4a6c7bd17519a336e5e5dea5b/src/opts/SkRasterPipeline_opts.h#L3274-L3290>
        crate::peniko::Extend::Reflect => {
            let u = val
                - (val * inv_max * f32x4::splat(simd, 0.5)).floor() * f32x4::splat(simd, 2.0) * max;
            let s = (u * inv_max).floor();
            let m = u - f32x4::splat(simd, 2.0) * s * (u - max);

            let bias_in_ulps = s.trunc();

            let m_bits = u32x4::from_bytes(m.to_bytes());
            // This would yield NaN if `m` is 0 and `bias_in_ulps` > 0, but since
            // our `max` is always an integer number, u and s must also be an integer number
            // and thus `m_bits` must be 0.
            // Note that this is a wrapping sub!
            let biased_bits = m_bits - bias_in_ulps.to_int::<u32x4<S>>();
            f32x4::from_bytes(biased_bits.to_bytes())
                // In certain edge cases, we might still end up with a higher number.
                .min(max - 1.0)
        }
    }
}

/// Calculate the weights for a single fractional value.
fn weights<S: Simd>(simd: S, fract: f32x4<S>) -> [f32x4<S>; 4] {
    simd.vectorize(
        #[inline(always)]
        || {
            let s = fract.simd;
            const MF: [[f32; 4]; 4] = mf_resampler();

            [
                single_weight(
                    fract,
                    f32x4::splat(s, MF[0][0]),
                    f32x4::splat(s, MF[0][1]),
                    f32x4::splat(s, MF[0][2]),
                    f32x4::splat(s, MF[0][3]),
                ),
                single_weight(
                    fract,
                    f32x4::splat(s, MF[1][0]),
                    f32x4::splat(s, MF[1][1]),
                    f32x4::splat(s, MF[1][2]),
                    f32x4::splat(s, MF[1][3]),
                ),
                single_weight(
                    fract,
                    f32x4::splat(s, MF[2][0]),
                    f32x4::splat(s, MF[2][1]),
                    f32x4::splat(s, MF[2][2]),
                    f32x4::splat(s, MF[2][3]),
                ),
                single_weight(
                    fract,
                    f32x4::splat(s, MF[3][0]),
                    f32x4::splat(s, MF[3][1]),
                    f32x4::splat(s, MF[3][2]),
                    f32x4::splat(s, MF[3][3]),
                ),
            ]
        },
    )
}

/// Calculate a weight based on the fractional value t and the cubic coefficients.
#[inline(always)]
fn single_weight<S: Simd>(
    t: f32x4<S>,
    a: f32x4<S>,
    b: f32x4<S>,
    c: f32x4<S>,
    d: f32x4<S>,
) -> f32x4<S> {
    t.madd(d, c).madd(t, b).madd(t, a)
}

/// Mitchell filter with the variables B = 1/3 and C = 1/3.
const fn mf_resampler() -> [[f32; 4]; 4] {
    cubic_resampler(1.0 / 3.0, 1.0 / 3.0)
}

/// Cubic resampling logic is borrowed from Skia. See
/// <https://github.com/google/skia/blob/220fef664978643a47d4559ae9e762b91aba534a/include/core/SkSamplingOptions.h#L33-L50>
/// for some links to understand how this works. In principle, this macro allows us to define a
/// resampler kernel based on two variables B and C which can be between 0 and 1, allowing to
/// change some properties of the cubic interpolation kernel.
///
/// As mentioned above, cubic resampling consists of sampling the 16 surrounding pixels of the
/// target point and interpolating them with a cubic filter.
/// The generated matrix is 4x4 and represent the coefficients of the cubic function used to
/// calculate weights based on the `x_fract` and `y_fract` of the location we are looking at.
const fn cubic_resampler(b: f32, c: f32) -> [[f32; 4]; 4] {
    [
        [
            (1.0 / 6.0) * b,
            -(3.0 / 6.0) * b - c,
            (3.0 / 6.0) * b + 2.0 * c,
            -(1.0 / 6.0) * b - c,
        ],
        [
            1.0 - (2.0 / 6.0) * b,
            0.0,
            -3.0 + (12.0 / 6.0) * b + c,
            2.0 - (9.0 / 6.0) * b - c,
        ],
        [
            (1.0 / 6.0) * b,
            (3.0 / 6.0) * b + c,
            3.0 - (15.0 / 6.0) * b - 2.0 * c,
            -2.0 + (9.0 / 6.0) * b + c,
        ],
        [0.0, 0.0, -c, (1.0 / 6.0) * b + c],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::fearless_simd::Fallback;

    #[test]
    fn extend_overflow() {
        let simd = Fallback::new();
        let max = f32x4::splat(simd, 128.0);
        let max_inv = 1.0 / max;

        let num = f32x4::splat(simd, 127.00001);
        let res = extend(simd, num, crate::peniko::Extend::Repeat, max, max_inv);

        assert!(res[0] <= 127.0);
    }
}
