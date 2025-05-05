// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::{COLOR_COMPONENTS, Painter, TILE_HEIGHT_COMPONENTS};
use vello_common::encode::EncodedImage;
use vello_common::kurbo::{Point, Vec2};
use vello_common::peniko::{Extend, ImageQuality};
use vello_common::tile::Tile;

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
            cur_pos: image.transform * Point::new(start_x as f64 + 0.5, start_y as f64 + 0.5),
            image,
        }
    }

    pub(super) fn run(mut self, target: &mut [u8]) {
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
                    self.image.pixmap.height() as f32,
                );
            }

            target
                .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
                .for_each(|column| {
                    let extended_x_pos = extend(
                        // As above, always floor.
                        x_pos.floor() as f32,
                        self.image.extends.0,
                        self.image.pixmap.width() as f32,
                    );
                    self.run_simple_column(column, extended_x_pos, &y_positions);
                    x_pos += x_advance;
                });
        }
    }

    fn run_simple_column(
        &mut self,
        col: &mut [u8],
        x_pos: f32,
        y_positions: &[f32; Tile::HEIGHT as usize],
    ) {
        for (pixel, y_pos) in col
            .chunks_exact_mut(COLOR_COMPONENTS)
            .zip(y_positions.iter())
        {
            let sample = match self.image.quality {
                ImageQuality::Low => self.image.pixmap.sample(x_pos as u16, *y_pos as u16),
                ImageQuality::Medium | ImageQuality::High => unimplemented!(),
            };

            pixel.copy_from_slice(sample);
        }
    }

    fn run_complex_column(&mut self, col: &mut [u8]) {
        let extend_point = |mut point: Point| {
            // For the same reason as mentioned above, we always floor.
            point.x = extend(
                point.x.floor() as f32,
                self.image.extends.0,
                self.image.pixmap.width() as f32,
            ) as f64;
            point.y = extend(
                point.y.floor() as f32,
                self.image.extends.1,
                self.image.pixmap.height() as f32,
            ) as f64;

            point
        };

        let mut pos = self.cur_pos;

        for pixel in col.chunks_exact_mut(COLOR_COMPONENTS) {
            match self.image.quality {
                // Nearest neighbor filtering.
                // Simply takes the nearest pixel to our current position.
                ImageQuality::Low => {
                    let point = extend_point(pos);
                    let sample = self.image.pixmap.sample(point.x as u16, point.y as u16);
                    pixel.copy_from_slice(sample);
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
                        let mut res = start.fract() as f32;

                        // In case we are in the negative we need to mirror the result.
                        if res.is_sign_negative() {
                            res += 1.0;
                        }

                        res
                    };

                    let x_fract = fract(pos.x);
                    let y_fract = fract(pos.y);

                    let mut f32_color = [0.0_f32; 4];

                    let sample = |p: Point| self.image.pixmap.sample(p.x as u16, p.y as u16);

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

                                for i in 0..COLOR_COMPONENTS {
                                    f32_color[i] += w * color_sample[i] as f32;
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

                                for i in 0..COLOR_COMPONENTS {
                                    f32_color[i] += c * color_sample[i] as f32;
                                }
                            }
                        }
                    }

                    let mut u8_color = [0; 4];

                    for i in 0..COLOR_COMPONENTS {
                        u8_color[i] = (f32_color[i] + 0.5) as u8;
                    }

                    pixel.copy_from_slice(&u8_color);
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
        Extend::Repeat => val.rem_euclid(max),
        Extend::Reflect => {
            let period = 2.0 * max;

            let val_mod = val.rem_euclid(period);

            if val_mod < max {
                val_mod
            } else {
                (period - 1.0) - val_mod
            }
        }
    }
}

impl Painter for ImageFiller<'_> {
    fn paint(self, target: &mut [u8]) {
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
