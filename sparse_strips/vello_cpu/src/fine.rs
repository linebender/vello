// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fine rasterization runs the commands in each wide tile to determine the final RGBA value
//! of each pixel and pack it into the pixmap.

use crate::paint::{EncodedLinearGradient, EncodedPaint, EncodedSweepGradient, GradientRange};
use std::f32::consts::PI;
use std::iter;
use vello_common::kurbo::{Affine, Point};
use vello_common::{
    coarse::{Cmd, WideTile},
    paint::Paint,
    tile::Tile,
};

pub(crate) const COLOR_COMPONENTS: usize = 4;
pub(crate) const TILE_HEIGHT_COMPONENTS: usize = Tile::HEIGHT as usize * COLOR_COMPONENTS;
pub(crate) const SCRATCH_BUF_SIZE: usize =
    WideTile::WIDTH as usize * Tile::HEIGHT as usize * COLOR_COMPONENTS;

pub(crate) type ScratchBuf = [u8; SCRATCH_BUF_SIZE];

#[derive(Debug)]
#[doc(hidden)]
/// This is an internal struct, do not access directly.
pub struct Fine<'a> {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) out_buf: &'a mut [u8],
    pub(crate) blend_buf: ScratchBuf,
    pub(crate) color_buf: ScratchBuf,
}

impl<'a> Fine<'a> {
    /// Create a new fine rasterizer.
    pub fn new(width: u16, height: u16, out_buf: &'a mut [u8]) -> Self {
        let scratch = [0; SCRATCH_BUF_SIZE];
        let color_scratch = [0; SCRATCH_BUF_SIZE];

        Self {
            width,
            height,
            out_buf,
            blend_buf: scratch,
            color_buf: color_scratch,
        }
    }

    pub fn clear(&mut self, premul_color: [u8; 4]) {
        if premul_color[0] == premul_color[1]
            && premul_color[1] == premul_color[2]
            && premul_color[2] == premul_color[3]
        {
            // All components are the same, so we can use memset instead.
            self.blend_buf.fill(premul_color[0]);
        } else {
            for z in self.blend_buf.chunks_exact_mut(COLOR_COMPONENTS) {
                z.copy_from_slice(&premul_color);
            }
        }
    }

    pub(crate) fn pack(&mut self, x: u16, y: u16) {
        pack(
            self.out_buf,
            &self.blend_buf,
            self.width.into(),
            self.height.into(),
            x.into(),
            y.into(),
        );
    }

    pub(crate) fn run_cmd(
        &mut self,
        tile_x: u16,
        tile_y: u16,
        cmd: &Cmd,
        alphas: &[u8],
        paints: &[EncodedPaint],
    ) {
        match cmd {
            Cmd::Fill(f) => {
                self.fill(
                    f.x as usize,
                    tile_x,
                    tile_y,
                    f.width as usize,
                    &f.paint,
                    paints,
                );
            }
            Cmd::AlphaFill(s) => {
                let a_slice = &alphas[s.alpha_ix..];
                self.strip(
                    s.x as usize,
                    tile_x,
                    tile_y,
                    s.width as usize,
                    a_slice,
                    &s.paint,
                    paints,
                );
            }
        }
    }

    /// Fill at a given x and with a width using the given paint.
    pub fn fill(
        &mut self,
        x: usize,
        tile_x: u16,
        tile_y: u16,
        width: usize,
        fill: &Paint,
        encoded_paints: &[EncodedPaint],
    ) {
        let blend_buf =
            &mut self.blend_buf[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];
        let color_buf =
            &mut self.color_buf[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];

        let start_x = tile_x * WideTile::WIDTH + x as u16;
        let start_y = tile_y * Tile::HEIGHT;

        match fill {
            Paint::Solid(color) => {
                let color = &color.to_u8_array();

                // If color is completely opaque we can just memcopy the colors.
                if color[3] == 255 {
                    for t in blend_buf.chunks_exact_mut(COLOR_COMPONENTS) {
                        t.copy_from_slice(color);
                    }

                    return;
                }

                fill::src_over(blend_buf, iter::repeat(*color));
            }
            Paint::Indexed(i) => {
                let paint = &encoded_paints[i.index()];

                match paint {
                    EncodedPaint::LinearGradient(g) => {
                        let mut iter = LinearGradientFiller::new(g, start_x, start_y);

                        if g.has_opacities {
                            iter.run(color_buf);
                            fill::src_over(
                                blend_buf,
                                color_buf.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
                            );
                        } else {
                            // Similarly to solid colors we can just override the previous values
                            // if all colors in the gradient are fully opaque.
                            iter.run(blend_buf);
                        }
                    }
                    EncodedPaint::SweepGradient(s) => {
                        let mut iter = SweepGradientFiller::new(s, start_x, start_y);

                        if s.has_opacities {
                            iter.run(color_buf);
                            fill::src_over(
                                blend_buf,
                                color_buf.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
                            );
                        } else {
                            // Similarly to solid colors we can just override the previous values
                            // if all colors in the gradient are fully opaque.
                            iter.run(blend_buf);
                        }
                    }
                }
            }
        }
    }

    /// Strip at a given x and with a width using the given paint and alpha values.
    pub fn strip(
        &mut self,
        x: usize,
        tile_x: u16,
        tile_y: u16,
        width: usize,
        alphas: &[u8],
        fill: &Paint,
        paints: &[EncodedPaint],
    ) {
        debug_assert!(
            alphas.len() >= width,
            "alpha buffer doesn't contain sufficient elements"
        );

        let blend_buf =
            &mut self.blend_buf[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];
        let color_buf =
            &mut self.color_buf[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];

        let start_x = tile_x * WideTile::WIDTH + x as u16;
        let start_y = tile_y * Tile::HEIGHT;

        match fill {
            Paint::Solid(color) => {
                strip::src_over(blend_buf, iter::repeat(color.to_u8_array()), alphas);
            }
            Paint::Indexed(i) => {
                let encoded_paint = &paints[i.index()];

                match encoded_paint {
                    EncodedPaint::LinearGradient(g) => {
                        let mut iter = LinearGradientFiller::new(g, start_x, start_y);
                        iter.run(color_buf);
                        strip::src_over(
                            blend_buf,
                            color_buf.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
                            alphas,
                        );
                    }
                    EncodedPaint::SweepGradient(s) => {
                        // let mut iter = SweepGradientFiller::new(s, start_x, start_y);
                        // iter.run(color_buf);
                        // strip::src_over(
                        //     blend_buf,
                        //     color_buf.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
                        //     alphas,
                        // );
                    }
                }
            }
        }
    }
}

fn pack(out_buf: &mut [u8], scratch: &ScratchBuf, width: usize, height: usize, x: usize, y: usize) {
    let base_ix = (y * usize::from(Tile::HEIGHT) * width + x * usize::from(WideTile::WIDTH))
        * COLOR_COMPONENTS;

    // Make sure we don't process rows outside the range of the pixmap.
    let max_height = (height - y * usize::from(Tile::HEIGHT)).min(usize::from(Tile::HEIGHT));

    for j in 0..max_height {
        let line_ix = base_ix + j * width * COLOR_COMPONENTS;

        // Make sure we don't process columns outside the range of the pixmap.
        let max_width =
            (width - x * usize::from(WideTile::WIDTH)).min(usize::from(WideTile::WIDTH));
        let target_len = max_width * COLOR_COMPONENTS;
        // This helps the compiler to understand that any access to `dest` cannot
        // be out of bounds, and thus saves corresponding checks in the for loop.
        let dest = &mut out_buf[line_ix..][..target_len];

        for i in 0..max_width {
            let src = &scratch[(i * usize::from(Tile::HEIGHT) + j) * COLOR_COMPONENTS..]
                [..COLOR_COMPONENTS];
            dest[i * COLOR_COMPONENTS..][..COLOR_COMPONENTS]
                .copy_from_slice(&src[..COLOR_COMPONENTS]);
        }
    }
}

pub(crate) mod fill {
    // See https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators for the
    // formulas.

    use crate::fine::{COLOR_COMPONENTS, TILE_HEIGHT_COMPONENTS};
    use crate::util::scalar::div_255;

    pub(crate) fn src_over<T: Iterator<Item = [u8; COLOR_COMPONENTS]>>(
        target: &mut [u8],
        mut color_iter: T,
    ) {
        for strip in target.chunks_exact_mut(TILE_HEIGHT_COMPONENTS) {
            for bg_c in strip.chunks_exact_mut(COLOR_COMPONENTS) {
                let src_c = color_iter.next().unwrap();
                for i in 0..COLOR_COMPONENTS {
                    bg_c[i] = src_c[i] + div_255(bg_c[i] as u16 * (255 - src_c[3] as u16)) as u8;
                }
            }
        }
    }
}

pub(crate) mod strip {
    use crate::fine::{COLOR_COMPONENTS, TILE_HEIGHT_COMPONENTS};
    use crate::util::scalar::div_255;
    use vello_common::tile::Tile;

    pub(crate) fn src_over<T: Iterator<Item = [u8; COLOR_COMPONENTS]>>(
        target: &mut [u8],
        mut color_iter: T,
        alphas: &[u8],
    ) {
        for (bg_c, masks) in target
            .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
            .zip(alphas.chunks_exact(usize::from(Tile::HEIGHT)))
        {
            for j in 0..usize::from(Tile::HEIGHT) {
                let src_c = color_iter.next().unwrap();
                let mask_a = u16::from(masks[j]);
                let inv_src_a_mask_a = 255 - div_255(mask_a * src_c[3] as u16);

                for i in 0..COLOR_COMPONENTS {
                    let im1 = bg_c[j * COLOR_COMPONENTS + i] as u16 * inv_src_a_mask_a;
                    let im2 = src_c[i] as u16 * mask_a;
                    let im3 = div_255(im1 + im2);
                    bg_c[j * COLOR_COMPONENTS + i] = im3 as u8;
                }
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct LinearGradientFiller<'a> {
    /// The position of the next x that should be processed.
    cur_pos: f32,
    x_advance: f32,
    y_advance: f32,
    /// The index of the current right stop we are processing.
    range_idx: usize,
    temp_range_idx: usize,
    /// The underlying gradient.
    gradient: &'a EncodedLinearGradient,
    cur_range: &'a GradientRange,
    temp_range: &'a GradientRange,
}

impl<'a> LinearGradientFiller<'a> {
    pub(crate) fn new(
        gradient: &'a EncodedLinearGradient,
        mut start_x: u16,
        mut start_y: u16,
    ) -> Self {
        // The actual starting point of the strip.
        let x0 = start_x as f32 + gradient.offsets.0 + 0.5;
        let y0 = start_y as f32 + gradient.offsets.1 + 0.5;

        let cur_pos = (x0 * gradient.fact1 - y0 * gradient.fact2) / gradient.denom;

        let mut filler = Self {
            cur_pos,
            x_advance: gradient.advances.0,
            y_advance: gradient.advances.1,
            range_idx: 0,
            temp_range_idx: 0,
            cur_range: &gradient.ranges[0],
            temp_range: &gradient.ranges[0],
            gradient,
        };

        filler
    }

    fn advance<T: Sign>(&mut self) {
        while self.cur_pos > self.cur_range.x1 || self.cur_pos < self.cur_range.x0 {
            T::idx_advance(&mut self.range_idx, self.gradient.ranges.len());
            self.cur_range = &self.gradient.ranges[self.range_idx];
        }
    }

    fn advance_temp<T: Sign>(&mut self, target_pos: f32) {
        while target_pos > self.temp_range.x1 || target_pos < self.temp_range.x0 {
            T::idx_advance(&mut self.temp_range_idx, self.gradient.ranges.len());
            self.temp_range = &self.gradient.ranges[self.temp_range_idx];
        }
    }

    fn run(mut self, target: &mut [u8]) {
        if self.gradient.pad {
            if self.gradient.sign == 1 {
                self.run_inner::<Pad, Positive>(target);
            } else {
                self.run_inner::<Pad, Negative>(target);
            }
        } else {
            if self.gradient.sign == 1 {
                self.run_inner::<Repeat, Positive>(target);
            } else {
                self.run_inner::<Repeat, Negative>(target);
            }
        }
    }

    fn run_inner<T: Extend, S: Sign>(mut self, target: &mut [u8]) {
        let mut col_positions = [0.0; Tile::HEIGHT as usize];

        target
            .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
            .for_each(|col| {
                self.cur_pos = T::extend(self.cur_pos, 0.0, self.gradient.end);

                let mut needs_advance = false;
                let range = self.cur_range;
                for i in 0..COLOR_COMPONENTS {
                    let base_pos = self.cur_pos + i as f32 * self.y_advance;
                    needs_advance |= base_pos > range.x1 || base_pos < range.x0;
                    col_positions[i] = base_pos;
                }

                if needs_advance {
                    for i in 0..COLOR_COMPONENTS {
                        col_positions[i] = T::extend(col_positions[i], 0.0, self.gradient.end);
                    }

                    self.run_col::<Advancer, S>(col, &col_positions);
                } else {
                    self.run_col::<NoAdvancer, S>(col, &col_positions);
                }

                self.cur_pos = T::extend(self.cur_pos + self.x_advance, 0.0, self.gradient.end);
                self.advance::<S>()
            })
    }

    #[inline(always)]
    fn run_col<T: Advance, S: Sign>(
        &mut self,
        column: &mut [u8],
        positions: &[f32; Tile::HEIGHT as usize],
    ) {
        for (pixel, target_pos) in column.chunks_exact_mut(COLOR_COMPONENTS).zip(positions) {
            let range = T::get_range::<S>(self, *target_pos);

            for col_idx in 0..COLOR_COMPONENTS {
                let im3 = target_pos - range.x0;
                let combined = (range.im3[col_idx] * im3 + 0.5) as i16;

                pixel[col_idx] = (range.c0[col_idx] as i16 + combined) as u8;
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct SweepGradientFiller<'a> {
    /// The position of the next x that should be processed.
    cur_pos: (f32, f32),
    x0: f32,
    x1: f32,
    /// The color of the left stop.
    c0: [u8; 4],
    /// The color of the right stop.
    c1: [u8; 4],
    im1: [f32; 4],
    im2: f32,
    im3: [f32; 4],
    /// The output buffer for emitting colors from the iterator.
    color_buf: [u8; COLOR_COMPONENTS],
    /// The underlying gradient.
    gradient: &'a EncodedSweepGradient,
}

impl<'a> SweepGradientFiller<'a> {
    pub(crate) fn new(
        gradient: &'a EncodedSweepGradient,
        mut start_x: u16,
        mut start_y: u16,
    ) -> Self {
        let mut start_point = Point::new(
            start_x as f64 + gradient.offsets.0 as f64,
            start_y as f64 + gradient.offsets.1 as f64,
        );
        start_point =
            Affine::rotate_about(gradient.rotation as f64, Point::new(0.0, 0.0)) * start_point;

        let left_stop = &gradient.stops[0];
        let right_stop = &gradient.stops[1];

        let c0 = left_stop.color;
        let c1 = right_stop.color;
        let x0 = 0.0;
        let x1 = gradient.end_angle;

        let mut im1 = [0.0; 4];
        let im2 = x1 - x0;
        let mut im3 = [0.0; 4];

        for i in 0..COLOR_COMPONENTS {
            im1[i] = c1[i] as f32 - c0[i] as f32;
            im3[i] = im1[i] / im2;
        }

        let mut filler = Self {
            cur_pos: (start_point.x as f32, start_point.y as f32 + 0.5),
            c0,
            c1,
            x1,
            x0,
            im1,
            im2,
            im3,
            color_buf: [0; COLOR_COMPONENTS],
            gradient,
        };

        filler
    }

    fn run(mut self, target: &mut [u8]) {
        target
            .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
            .for_each(|column| {
                let mut pos = self.cur_pos;

                for pixel in column.chunks_exact_mut(COLOR_COMPONENTS) {
                    let angle = (-pos.1).atan2(pos.0).rem_euclid(2.0 * PI);

                    for col_idx in 0..COLOR_COMPONENTS {
                        let im3 = angle - self.x0;
                        let combined = (self.im3[col_idx] * im3 + 0.5) as i16;

                        pixel[col_idx] = (self.c0[col_idx] as i16 + combined) as u8;
                    }

                    pos.1 += 1.0;
                }

                self.cur_pos.0 += 1.0;
            })
    }
}

trait Extend {
    fn extend(val: f32, min: f32, max: f32) -> f32;
}

struct Pad;
impl Extend for Pad {
    fn extend(val: f32, _: f32, _: f32) -> f32 {
        val
    }
}

trait Sign {
    fn idx_advance(idx: &mut usize, gradient_len: usize);
}

struct Negative;
impl Sign for Negative {
    fn idx_advance(idx: &mut usize, gradient_len: usize) {
        if *idx >= (gradient_len - 1) {
            *idx = 0;
        } else {
            *idx += 1;
        }
    }
}

struct Positive;
impl Sign for Positive {
    fn idx_advance(idx: &mut usize, gradient_len: usize) {
        if *idx == 0 {
            *idx = gradient_len - 1;
        } else {
            *idx -= 1;
        }
    }
}

struct Repeat;
impl Extend for Repeat {
    fn extend(val: f32, min: f32, max: f32) -> f32 {
        min + val.rem_euclid(max - min)
    }
}

trait Advance {
    fn get_range<'a, S: Sign>(
        gf: &mut LinearGradientFiller<'a>,
        target_pos: f32,
    ) -> &'a GradientRange;
}

struct Advancer;
impl Advance for Advancer {
    #[inline]
    fn get_range<'a, S: Sign>(
        gf: &mut LinearGradientFiller<'a>,
        target_pos: f32,
    ) -> &'a GradientRange {
        // It's possible that we have to skip multiple stops.
        gf.advance_temp::<S>(target_pos);

        gf.temp_range
    }
}

struct NoAdvancer;
impl Advance for NoAdvancer {
    fn get_range<'a, S: Sign>(gf: &mut LinearGradientFiller<'a>, _: f32) -> &'a GradientRange {
        gf.cur_range
    }
}
