// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fine rasterization runs the commands in each wide tile to determine the final RGBA value
//! of each pixel and pack it into the pixmap.

mod linear_gradient;
mod sweep_gradient;

use crate::paint::EncodedPaint;
use linear_gradient::LinearGradientFiller;
use std::iter;
use sweep_gradient::SweepGradientFiller;
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
                        let iter = LinearGradientFiller::new(g, start_x, start_y);

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
                        let iter = SweepGradientFiller::new(s, start_x, start_y);

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
                        let iter = LinearGradientFiller::new(g, start_x, start_y);
                        iter.run(color_buf);
                        strip::src_over(
                            blend_buf,
                            color_buf.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
                            alphas,
                        );
                    }
                    EncodedPaint::SweepGradient(s) => {
                        let mut iter = SweepGradientFiller::new(s, start_x, start_y);
                        iter.run(color_buf);
                        strip::src_over(
                            blend_buf,
                            color_buf.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
                            alphas,
                        );
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

trait Extend {
    fn extend(val: f32, max: f32) -> f32;
}

trait Sign {
    fn needs_advance(base_pos: f32, x0: f32, x1: f32) -> bool;
    fn idx_advance(idx: &mut usize, gradient_len: usize);
}

struct Pad;

impl Extend for Pad {
    fn extend(val: f32, _: f32) -> f32 {
        val
    }
}

struct Repeat;

impl Extend for Repeat {
    fn extend(mut val: f32, max: f32) -> f32 {
        while val < 0.0 {
            val += max;
        }

        while val > max {
            val -= max;
        }

        val
    }
}

struct Negative;

impl Sign for Negative {
    fn needs_advance(base_pos: f32, x0: f32, _: f32) -> bool {
        base_pos < x0
    }

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
    fn needs_advance(base_pos: f32, _: f32, x1: f32) -> bool {
        base_pos > x1
    }

    fn idx_advance(idx: &mut usize, gradient_len: usize) {
        if *idx == 0 {
            *idx = gradient_len - 1;
        } else {
            *idx -= 1;
        }
    }
}

#[inline(always)]
pub(crate) fn extend(mut val: f32, pad: bool, start: f32, end: f32) -> f32 {
    if pad {
        val
    } else {
        while val < start {
            val += end - start;
        }

        while val > end {
            val -= (end - start);
        }

        val
    }
}
