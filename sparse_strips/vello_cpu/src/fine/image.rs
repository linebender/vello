// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::{COLOR_COMPONENTS, FineType, Painter, TILE_HEIGHT_COMPONENTS};
use vello_common::encode::EncodedImage;
use vello_common::kurbo::{Point, Vec2};
use vello_common::peniko::{Extend, ImageQuality};
use vello_common::tile::Tile;

#[cfg(not(feature = "std"))]
use vello_common::kurbo::common::FloatFuncs as _;

#[cfg(feature = "std")]
fn fract(val: f64) -> f64 {
    val.fract()
}

#[cfg(not(feature = "std"))]
fn fract(val: f64) -> f64 {
    #[cfg(feature = "libm")]
    return val - libm::trunc(val);
    #[cfg(not(feature = "libm"))]
    compile_error!("vello_common requires either the `std` or `libm` feature");
}

// Inlined version of f32::rem_euclid, to allow using abs.
fn rem_euclid(lhs: f32, rhs: f32) -> f32 {
    let r = lhs % rhs;
    if r < 0.0 { r + rhs.abs() } else { r }
}

#[derive(Debug)]
pub(crate) struct ImageFiller<'a> {
    /// The current position that should be processed.
    cur_pos: Point,
    /// The underlying image.
    image: &'a EncodedImage,
}

impl<'a> ImageFiller<'a> {
    pub(crate) fn new(image: &'a EncodedImage, start_x: u16, start_y: u16) -> Self {
        Self {
            // We want to sample values of the pixels at the center, so add an offset of 0.5.
            cur_pos: image.transform
                * Point::new(f64::from(start_x) + 0.5, f64::from(start_y) + 0.5),
            image,
        }
    }

    pub(super) fn run<F: FineType>(mut self, target: &mut [F]) {
        // We currently have two branches for filling images: The first case is used for
        // nearest neighbor filtering and for images with no skewing-transform (this is checked
        // by the first two conditions), which allows us to take a faster path.
        // The second version is the general case for any other image.
        // Once we get to performance optimizations, it's possible that there will be further
        // paths (e.g. one for no scaling transform and only integer translation offsets).
        if self.image.y_advance.x != 0.0
            || self.image.x_advance.y != 0.0
            || self.image.quality != ImageQuality::Low
        {
            // Fallback path.
            target
                .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
                .for_each(|column| {
                    self.run_complex_column(column);
                    self.cur_pos += self.image.x_advance;
                });
        } else {
            // Fast path. Each step in the x/y direction only updates x/y component of the
            // current position, since we have no skewing.
            // Most importantly, the y position is the same across each column, allowing us
            // to precompute it (as well as it's extend).
            let mut x_pos = self.cur_pos.x;
            let x_advance = self.image.x_advance.x;
            let y_advance = self.image.y_advance.y;

            let mut y_positions = [0.0; Tile::HEIGHT as usize];

            for (idx, pos) in y_positions.iter_mut().enumerate() {
                *pos = extend(
                    // Since we already added a 0.5 offset to sample at the center of the pixel,
                    // we always floor to get the target pixel.
                    (self.cur_pos.y + y_advance * idx as f64).floor() as f32,
                    self.image.extends.1,
                    f32::from(self.image.pixmap.height()),
                );
            }

            target
                .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
                .for_each(|column| {
                    let extended_x_pos = extend(
                        // As above, always floor.
                        x_pos.floor() as f32,
                        self.image.extends.0,
                        f32::from(self.image.pixmap.width()),
                    );
                    self.run_simple_column(column, extended_x_pos, &y_positions);
                    x_pos += x_advance;
                });
        }
    }

    // Ideally, we'd add this only on the specific arm, but clippy doesn't support that
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "Tile::HEIGHT is expected to increase later."
    )]
    fn run_simple_column<F: FineType>(
        &mut self,
        col: &mut [F],
        x_pos: f32,
        y_positions: &[f32; Tile::HEIGHT as usize],
    ) {
        for (pixel, y_pos) in col
            .chunks_exact_mut(COLOR_COMPONENTS)
            .zip(y_positions.iter())
        {
            let sample = match self.image.quality {
                ImageQuality::Low => F::from_rgba8(
                    &self
                        .image
                        .pixmap
                        .sample(x_pos as u16, *y_pos as u16)
                        .to_u8_array()[..],
                ),
                ImageQuality::Medium | ImageQuality::High => unimplemented!(),
            };

            pixel.copy_from_slice(&sample);
        }
    }

    fn run_complex_column<F: FineType>(&mut self, col: &mut [F]) {
        let extend_point = |mut point: Point| {
            // For the same reason as mentioned above, we always floor.
            point.x = f64::from(extend(
                point.x.floor() as f32,
                self.image.extends.0,
                f32::from(self.image.pixmap.width()),
            ));
            point.y = f64::from(extend(
                point.y.floor() as f32,
                self.image.extends.1,
                f32::from(self.image.pixmap.height()),
            ));

            point
        };

        let mut pos = self.cur_pos;

        for pixel in col.chunks_exact_mut(COLOR_COMPONENTS) {
            match self.image.quality {
                // Nearest neighbor filtering.
                // Simply takes the nearest pixel to our current position.
                ImageQuality::Low => {
                    let point = extend_point(pos);
                    let sample = F::from_rgba8(
                        &self
                            .image
                            .pixmap
                            .sample(point.x as u16, point.y as u16)
                            .to_u8_array()[..],
                    );
                    pixel.copy_from_slice(&sample);
                }
                ImageQuality::Medium | ImageQuality::High => {
                    // We have two versions of filtering: `Medium` (bilinear filtering) and
                    // `High` (bicubic filtering).

                    // In bilinear filtering, we sample the pixels of the rectangle that spans the
                    // locations (-0.5, -0.5) and (0.5, 0.5), and weight them by the fractional
                    // x/y position using simple linear interpolation in both dimensions.
                    // In bicubic filtering, we instead span a 4x4 grid around the
                    // center of the location we are sampling, and sample those points
                    // using a cubic filter to weight each location's contribution.

                    let fract = |orig_val: f64| {
                        // To give some intuition on why we need that shift, based on bilinear
                        // filtering: If we sample at the position (0.5, 0.5), we are at the center
                        // of the pixel and thus only want the color of the current pixel. Thus, we take
                        // 1.0 * 1.0 from the top left pixel (which still lies on our pixel)
                        // and 0.0 from all other corners (which lie at the start of other pixels).
                        //
                        // If we sample at the position (0.4, 0.4), we want 0.1 * 0.1 = 0.01 from
                        // the top-left pixel, 0.1 * 0.9 = 0.09 from the bottom-left and top-right,
                        // and finally 0.9 * 0.9 = 0.81 from the bottom right position (which still
                        // lies on our pixel, and thus has intuitively should have the highest
                        // contribution). Thus, we need to subtract 0.5 from the position to get
                        // the correct fractional contribution.
                        let start = orig_val - 0.5;
                        let mut res = fract(start) as f32;

                        // In case we are in the negative we need to mirror the result.
                        if res.is_sign_negative() {
                            res += 1.0;
                        }

                        res
                    };

                    let x_fract = fract(pos.x);
                    let y_fract = fract(pos.y);

                    let mut interpolated_color = [0.0_f32; 4];

                    let sample = |p: Point| {
                        let c = |val: u8| f32::from(val) / 255.0;
                        let s = self.image.pixmap.sample(p.x as u16, p.y as u16);

                        [c(s.r), c(s.g), c(s.b), c(s.a)]
                    };

                    if self.image.quality == ImageQuality::Medium {
                        let cx = [1.0 - x_fract, x_fract];
                        let cy = [1.0 - y_fract, y_fract];

                        // Note that the sum of all cx*cy combinations also yields 1.0 again
                        // (modulo some floating point number impreciseness), ensuring the
                        // colors stay in range.

                        // We sample the corners rectangle that covers our current position.
                        for (x_idx, x) in [-0.5, 0.5].into_iter().enumerate() {
                            for (y_idx, y) in [-0.5, 0.5].into_iter().enumerate() {
                                let color_sample = sample(extend_point(pos + Vec2::new(x, y)));
                                let w = cx[x_idx] * cy[y_idx];

                                for (component, component_sample) in
                                    interpolated_color.iter_mut().zip(color_sample)
                                {
                                    *component += w * component_sample;
                                }
                            }
                        }
                    } else {
                        // Compare to https://github.com/google/skia/blob/84ff153b0093fc83f6c77cd10b025c06a12c5604/src/opts/SkRasterPipeline_opts.h#L5030-L5075.
                        let cx = weights(x_fract);
                        let cy = weights(y_fract);

                        // Note in particular that it is guaranteed that, similarly to bilinear filtering,
                        // the sum of all cx*cy is 1.

                        // We sample the 4x4 grid around the position we are currently looking at.
                        for (x_idx, x) in [-1.5, -0.5, 0.5, 1.5].into_iter().enumerate() {
                            for (y_idx, y) in [-1.5, -0.5, 0.5, 1.5].into_iter().enumerate() {
                                let color_sample = sample(extend_point(pos + Vec2::new(x, y)));
                                let c = cx[x_idx] * cy[y_idx];

                                for (component, component_sample) in
                                    interpolated_color.iter_mut().zip(color_sample)
                                {
                                    *component += c * component_sample;
                                }
                            }
                        }
                    }

                    for i in 0..COLOR_COMPONENTS {
                        // Due to the nature of the cubic filter, it can happen in certain situations
                        // that one of the color components ends up with a higher value than the
                        // alpha component, which isn't permissible because the color is
                        // premultiplied and would lead to overflows when doing source over
                        // compositing with u8-based values. Because of this, we need to clamp
                        // to the alpha value.
                        let f32_val = interpolated_color[i]
                            .clamp(0.0, 1.0)
                            .min(interpolated_color[3]);
                        interpolated_color[i] = f32_val;
                    }

                    pixel.copy_from_slice(&F::from_rgbaf32(&interpolated_color[..]));
                }
            };

            pos += self.image.y_advance;
        }
    }
}

fn extend(val: f32, extend: Extend, max: f32) -> f32 {
    match extend {
        Extend::Pad => val.clamp(0.0, max - 1.0),
        // TODO: We need to make repeat and reflect more efficient and branch-less.
        Extend::Repeat => rem_euclid(val, max),
        Extend::Reflect => {
            let period = 2.0 * max;

            let val_mod = rem_euclid(val, period);

            if val_mod < max {
                val_mod
            } else {
                (period - 1.0) - val_mod
            }
        }
    }
}

impl Painter for ImageFiller<'_> {
    fn paint<F: FineType>(self, target: &mut [F]) {
        self.run(target);
    }
}

/// Calculate the weights for a single fractional value.
const fn weights(fract: f32) -> [f32; 4] {
    const MF: [[f32; 4]; 4] = mf_resampler();

    [
        single_weight(fract, MF[0][0], MF[0][1], MF[0][2], MF[0][3]),
        single_weight(fract, MF[1][0], MF[1][1], MF[1][2], MF[1][3]),
        single_weight(fract, MF[2][0], MF[2][1], MF[2][2], MF[2][3]),
        single_weight(fract, MF[3][0], MF[3][1], MF[3][2], MF[3][3]),
    ]
}

/// Calculate a weight based on the fractional value t and the cubic coefficients.
const fn single_weight(t: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    t * (t * (t * d + c) + b) + a
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
