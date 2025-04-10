// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fine rasterization runs the commands in each wide tile to determine the final RGBA value
//! of each pixel and pack it into the pixmap.

use vello_common::{
    coarse::{Cmd, WideTile},
    paint::Paint,
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
    pub(crate) scratch: Vec<ScratchBuf>,
}

impl Fine {
    /// Create a new fine rasterizer.
    pub fn new(width: u16, height: u16) -> Self {
        let scratch = [0; SCRATCH_BUF_SIZE];

        Self {
            width,
            height,
            scratch: vec![scratch],
        }
    }

    pub fn clear(&mut self, premul_color: [u8; 4]) {
        let scratch = self.scratch.last_mut().unwrap();
        if premul_color[0] == premul_color[1]
            && premul_color[1] == premul_color[2]
            && premul_color[2] == premul_color[3]
        {
            // All components are the same, so we can use memset instead.
            scratch.fill(premul_color[0]);
        } else {
            for z in scratch.chunks_exact_mut(COLOR_COMPONENTS) {
                z.copy_from_slice(&premul_color);
            }
        }
    }

    #[doc(hidden)]
    pub fn pack(&mut self, x: u16, y: u16, out_buf: &mut [u8]) {
        let scratch = self.scratch.last_mut().unwrap();
        pack(
            out_buf,
            scratch,
            self.width.into(),
            self.height.into(),
            x.into(),
            y.into(),
        );
    }

    pub(crate) fn run_cmd(&mut self, cmd: &Cmd, alphas: &[u8]) {
        match cmd {
            Cmd::Fill(f) => {
                self.fill(f.x as usize, f.width as usize, &f.paint);
            }
            Cmd::AlphaFill(s) => {
                let a_slice = &alphas[s.alpha_idx..];
                self.strip(s.x as usize, s.width as usize, a_slice, &s.paint);
            }
            Cmd::PushClip => {
                self.scratch.push([0; SCRATCH_BUF_SIZE]);
            }
            Cmd::PopClip => {
                self.scratch.pop();
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
    pub fn fill(&mut self, x: usize, width: usize, paint: &Paint) {
        let scratch = self.scratch.last_mut().unwrap();
        match paint {
            Paint::Solid(c) => {
                let color = c.to_u8_array();

                let target =
                    &mut scratch[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];

                // If color is completely opaque we can just memcopy the colors.
                if color[3] == 255 {
                    for t in target.chunks_exact_mut(COLOR_COMPONENTS) {
                        t.copy_from_slice(&color);
                    }

                    return;
                }

                fill::src_over(target, &color);
            }
            Paint::Indexed(_) => unimplemented!(),
        }
    }

    /// Strip at a given x and with a width using the given paint and alpha values.
    pub fn strip(&mut self, x: usize, width: usize, alphas: &[u8], paint: &Paint) {
        debug_assert!(
            alphas.len() >= width,
            "alpha buffer doesn't contain sufficient elements"
        );
        let scratch = self.scratch.last_mut().unwrap();

        match paint {
            Paint::Solid(s) => {
                let color = s.to_u8_array();

                let target =
                    &mut scratch[x * TILE_HEIGHT_COMPONENTS..][..TILE_HEIGHT_COMPONENTS * width];

                strip::src_over(target, &color, alphas);
            }
            Paint::Indexed(_) => unimplemented!(),
        }
    }

    fn clip_fill(&mut self, x: usize, width: usize) {
        let (source_buffer, rest) = self.scratch.split_last_mut().unwrap();
        let target_buffer = rest.last_mut().unwrap();

        for col_idx in 0..width {
            for row_idx in 0..usize::from(Tile::HEIGHT) {
                let px_offset = (x + col_idx) * TILE_HEIGHT_COMPONENTS + row_idx * COLOR_COMPONENTS;
                let source_alpha = source_buffer[px_offset + 3] as f32 / 255.0;
                let inverse_alpha = 1.0 - source_alpha;

                for channel_idx in 0..COLOR_COMPONENTS {
                    let dest = target_buffer[px_offset + channel_idx] as f32;
                    let src = source_buffer[px_offset + channel_idx] as f32;
                    target_buffer[px_offset + channel_idx] =
                        (dest * inverse_alpha + src * source_alpha) as u8;
                }
            }
        }
    }

    fn clip_strip(&mut self, x: usize, width: usize, alphas: &[u8]) {
        let (source_buffer, rest) = self.scratch.split_last_mut().unwrap();
        let target_buffer = rest.last_mut().unwrap();

        for (col_idx, column_alphas) in alphas
            .chunks_exact(usize::from(Tile::HEIGHT))
            .take(width)
            .enumerate()
        {
            for (row_idx, &alpha) in column_alphas.iter().enumerate() {
                let px_offset = (x + col_idx) * TILE_HEIGHT_COMPONENTS + row_idx * COLOR_COMPONENTS;
                let mask_alpha = alpha as f32 / 255.0;
                let source_alpha = source_buffer[px_offset + 3] as f32 / 255.0;
                let inverse_alpha = 1.0 - mask_alpha * source_alpha;

                for channel_idx in 0..COLOR_COMPONENTS {
                    let dest = target_buffer[px_offset + channel_idx] as f32;
                    let source = source_buffer[px_offset + channel_idx] as f32;
                    target_buffer[px_offset + channel_idx] =
                        (dest * inverse_alpha + mask_alpha * source) as u8;
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

    pub(crate) fn src_over(target: &mut [u8], src_c: &[u8; COLOR_COMPONENTS]) {
        let src_a = src_c[3] as u16;

        for strip in target.chunks_exact_mut(TILE_HEIGHT_COMPONENTS) {
            for bg_c in strip.chunks_exact_mut(COLOR_COMPONENTS) {
                for i in 0..COLOR_COMPONENTS {
                    bg_c[i] = src_c[i] + div_255(bg_c[i] as u16 * (255 - src_a)) as u8;
                }
            }
        }
    }
}

pub(crate) mod strip {
    use crate::fine::{COLOR_COMPONENTS, TILE_HEIGHT_COMPONENTS};
    use crate::util::scalar::div_255;
    use vello_common::tile::Tile;

    pub(crate) fn src_over(target: &mut [u8], src_c: &[u8; COLOR_COMPONENTS], alphas: &[u8]) {
        let src_a = src_c[3] as u16;

        for (bg_c, masks) in target
            .chunks_exact_mut(TILE_HEIGHT_COMPONENTS)
            .zip(alphas.chunks_exact(usize::from(Tile::HEIGHT)))
        {
            for j in 0..usize::from(Tile::HEIGHT) {
                let mask_a = u16::from(masks[j]);
                let inv_src_a_mask_a = 255 - div_255(mask_a * src_a);

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
