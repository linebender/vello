// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fine rasterization runs the commands in each wide tile to determine the final RGBA value
//! of each pixel and pack it into the pixmap.

mod gradient;
mod image;

use crate::fine::gradient::GradientFiller;
use crate::fine::image::ImageFiller;
use crate::util::scalar::div_255;
use alloc::vec;
use alloc::vec::Vec;
use core::iter;
use vello_common::encode::{EncodedKind, EncodedPaint};
use vello_common::paint::Paint;
use vello_common::{
    coarse::{Cmd, WideTile},
    tile::Tile,
};

pub(crate) const COLOR_COMPONENTS: usize = 4;
pub(crate) const TILE_HEIGHT_COMPONENTS: usize = Tile::HEIGHT as usize * COLOR_COMPONENTS;
#[doc(hidden)]
pub const SCRATCH_BUF_SIZE: usize =
    WideTile::WIDTH as usize * Tile::HEIGHT as usize * COLOR_COMPONENTS;
#[doc(hidden)]
pub type ScratchBuf = [u8; SCRATCH_BUF_SIZE];

#[derive(Debug)]
#[doc(hidden)]
/// This is an internal struct, do not access directly.
pub struct Fine {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) wide_coords: (u16, u16),
    pub(crate) blend_buf: Vec<ScratchBuf>,
    pub(crate) color_buf: ScratchBuf,
}

impl Fine {
    /// Create a new fine rasterizer.
    pub fn new(width: u16, height: u16) -> Self {
        let blend_buf = [0; SCRATCH_BUF_SIZE];
        let color_buf = [0; SCRATCH_BUF_SIZE];

        Self {
            width,
            height,
            wide_coords: (0, 0),
            blend_buf: vec![blend_buf],
            color_buf,
        }
    }

    /// Set the coordinates of the current wide tile that is being processed (in tile units).
    pub fn set_coords(&mut self, x: u16, y: u16) {
        self.wide_coords = (x, y);
    }

    pub fn clear(&mut self, premul_color: [u8; 4]) {
        let blend_buf = self.blend_buf.last_mut().unwrap();

        if premul_color[0] == premul_color[1]
            && premul_color[1] == premul_color[2]
            && premul_color[2] == premul_color[3]
        {
            // All components are the same, so we can use memset instead.
            blend_buf.fill(premul_color[0]);
        } else {
            for z in blend_buf.chunks_exact_mut(COLOR_COMPONENTS) {
                z.copy_from_slice(&premul_color);
            }
        }
    }

    #[doc(hidden)]
    pub fn pack(&mut self, out_buf: &mut [u8]) {
        let blend_buf = self.blend_buf.last_mut().unwrap();

        pack(
            out_buf,
            blend_buf,
            self.width.into(),
            self.height.into(),
            self.wide_coords.0.into(),
            self.wide_coords.1.into(),
        );
    }

    pub(crate) fn run_cmd(&mut self, cmd: &Cmd, alphas: &[u8], paints: &[EncodedPaint]) {
        match cmd {
            Cmd::Fill(f) => {
                self.fill(f.x as usize, f.width as usize, &f.paint, paints);
            }
            Cmd::AlphaFill(s) => {
                let a_slice = &alphas[s.alpha_idx..];
                self.strip(s.x as usize, s.width as usize, a_slice, &s.paint, paints);
            }
            Cmd::PushClip => {
                self.blend_buf.push([0; SCRATCH_BUF_SIZE]);
            }
            Cmd::PopClip => {
                self.blend_buf.pop();
            }
            Cmd::ClipFill(cf) => {
                self.clip_fill(cf.x as usize, cf.width as usize);
            }
            Cmd::ClipStrip(cs) => {
                let aslice = &alphas[cs.alpha_idx..];
                self.clip_strip(cs.x as usize, cs.width as usize, aslice);
            }
        }
    }

    /// Fill at a given x and with a width using the given paint.
    pub fn fill(&mut self, x: usize, width: usize, fill: &Paint, encoded_paints: &[EncodedPaint]) {
        let blend_buf = &mut self.blend_buf.last_mut().unwrap()[x * TILE_HEIGHT_COMPONENTS..]
            [..TILE_HEIGHT_COMPONENTS * width];
        let color_buf =
            &mut self.color_buf[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];

        let start_x = self.wide_coords.0 * WideTile::WIDTH + x as u16;
        let start_y = self.wide_coords.1 * Tile::HEIGHT;

        fn fill_complex_paint(
            color_buf: &mut [u8],
            blend_buf: &mut [u8],
            has_opacities: bool,
            filler: impl Painter,
        ) {
            if has_opacities {
                filler.paint(color_buf);
                fill::src_over(
                    blend_buf,
                    color_buf.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
                );
            } else {
                // Similarly to solid colors we can just override the previous values
                // if all colors in the gradient are fully opaque.
                filler.paint(blend_buf);
            }
        }

        match fill {
            Paint::Solid(color) => {
                let color = color.rbga_u8();

                // If color is completely opaque we can just memcopy the colors.
                if color[3] == 255 {
                    for t in blend_buf.chunks_exact_mut(COLOR_COMPONENTS) {
                        t.copy_from_slice(&color);
                    }

                    return;
                }

                fill::src_over(blend_buf, iter::repeat(color));
            }
            Paint::Indexed(paint) => {
                let encoded_paint = &encoded_paints[paint.index()];

                match encoded_paint {
                    EncodedPaint::Gradient(g) => match &g.kind {
                        EncodedKind::Linear(l) => {
                            let filler = GradientFiller::new(g, l, start_x, start_y);
                            fill_complex_paint(color_buf, blend_buf, g.has_opacities, filler);
                        }
                        EncodedKind::Radial(r) => {
                            let filler = GradientFiller::new(g, r, start_x, start_y);
                            fill_complex_paint(color_buf, blend_buf, g.has_opacities, filler);
                        }
                        EncodedKind::Sweep(s) => {
                            let filler = GradientFiller::new(g, s, start_x, start_y);
                            fill_complex_paint(color_buf, blend_buf, g.has_opacities, filler);
                        }
                    },
                    EncodedPaint::Image(i) => {
                        let filler = ImageFiller::new(i, start_x, start_y);
                        fill_complex_paint(color_buf, blend_buf, i.has_opacities, filler);
                    }
                }
            }
        }
    }

    /// Strip at a given x and with a width using the given paint and alpha values.
    pub fn strip(
        &mut self,
        x: usize,
        width: usize,
        alphas: &[u8],
        fill: &Paint,
        paints: &[EncodedPaint],
    ) {
        debug_assert!(
            alphas.len() >= width,
            "alpha buffer doesn't contain sufficient elements"
        );

        let blend_buf = &mut self.blend_buf.last_mut().unwrap()[x * TILE_HEIGHT_COMPONENTS..]
            [..TILE_HEIGHT_COMPONENTS * width];
        let color_buf =
            &mut self.color_buf[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];

        let start_x = self.wide_coords.0 * WideTile::WIDTH + x as u16;
        let start_y = self.wide_coords.1 * Tile::HEIGHT;

        fn strip_complex_paint(
            color_buf: &mut [u8],
            blend_buf: &mut [u8],
            filler: impl Painter,
            alphas: &[u8],
        ) {
            filler.paint(color_buf);
            strip::src_over(
                blend_buf,
                color_buf.chunks_exact(4).map(|e| [e[0], e[1], e[2], e[3]]),
                alphas,
            );
        }

        match fill {
            Paint::Solid(color) => {
                strip::src_over(blend_buf, iter::repeat(color.rbga_u8()), alphas);
            }
            Paint::Indexed(paint) => {
                let encoded_paint = &paints[paint.index()];

                match encoded_paint {
                    EncodedPaint::Gradient(g) => match &g.kind {
                        EncodedKind::Linear(l) => {
                            let filler = GradientFiller::new(g, l, start_x, start_y);
                            strip_complex_paint(color_buf, blend_buf, filler, alphas);
                        }
                        EncodedKind::Radial(r) => {
                            let filler = GradientFiller::new(g, r, start_x, start_y);
                            strip_complex_paint(color_buf, blend_buf, filler, alphas);
                        }
                        EncodedKind::Sweep(s) => {
                            let filler = GradientFiller::new(g, s, start_x, start_y);
                            strip_complex_paint(color_buf, blend_buf, filler, alphas);
                        }
                    },
                    EncodedPaint::Image(i) => {
                        let filler = ImageFiller::new(i, start_x, start_y);
                        strip_complex_paint(color_buf, blend_buf, filler, alphas);
                    }
                }
            }
        }
    }

    fn clip_fill(&mut self, x: usize, width: usize) {
        let (source_buffer, rest) = self.blend_buf.split_last_mut().unwrap();
        let target_buffer = rest.last_mut().unwrap();

        for col_idx in 0..width {
            for row_idx in 0..usize::from(Tile::HEIGHT) {
                let px_offset = (x + col_idx) * TILE_HEIGHT_COMPONENTS + row_idx * COLOR_COMPONENTS;
                let source_alpha = source_buffer[px_offset + 3] as u16;
                let inverse_alpha = 255 - source_alpha;

                for channel_idx in 0..COLOR_COMPONENTS {
                    let dest = target_buffer[px_offset + channel_idx] as u16;
                    let src = source_buffer[px_offset + channel_idx] as u16;
                    target_buffer[px_offset + channel_idx] =
                        (src + div_255(dest * inverse_alpha)) as u8;
                }
            }
        }
    }

    fn clip_strip(&mut self, x: usize, width: usize, alphas: &[u8]) {
        let (source_buffer, rest) = self.blend_buf.split_last_mut().unwrap();
        let target_buffer = rest.last_mut().unwrap();

        for (col_idx, column_alphas) in alphas
            .chunks_exact(usize::from(Tile::HEIGHT))
            .take(width)
            .enumerate()
        {
            for (row_idx, &alpha) in column_alphas.iter().enumerate() {
                let px_offset = (x + col_idx) * TILE_HEIGHT_COMPONENTS + row_idx * COLOR_COMPONENTS;
                let mask_alpha = alpha as u16;
                let source_alpha = source_buffer[px_offset + 3] as u16;
                let inverse_alpha = 255 - div_255(mask_alpha * source_alpha);

                for channel_idx in 0..COLOR_COMPONENTS {
                    let dest = target_buffer[px_offset + channel_idx] as u16;
                    let source = source_buffer[px_offset + channel_idx] as u16;
                    target_buffer[px_offset + channel_idx] =
                        div_255(dest * inverse_alpha + mask_alpha * source) as u8;
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

trait Painter {
    fn paint(self, target: &mut [u8]);
}
