// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fine rasterization runs the commands in each wide tile to determine the final RGBA value
//! of each pixel and pack it into the pixmap.

use crate::util::ColorExt;
use vello_common::coarse::{Cmd, WIDE_TILE_WIDTH};
use vello_common::paint::Paint;
use vello_common::strip::STRIP_HEIGHT;

pub(crate) const COLOR_COMPONENTS: usize = 4;
pub(crate) const STRIP_HEIGHT_COMPONENTS: usize = STRIP_HEIGHT * COLOR_COMPONENTS;
pub(crate) const SCRATCH_BUF_SIZE: usize = WIDE_TILE_WIDTH * STRIP_HEIGHT * COLOR_COMPONENTS;

pub(crate) type ScratchBuf = [u8; SCRATCH_BUF_SIZE];

pub(crate) struct Fine<'a> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) out_buf: &'a mut [u8],
    pub(crate) scratch: ScratchBuf,
}

impl<'a> Fine<'a> {
    pub(crate) fn new(width: usize, height: usize, out_buf: &'a mut [u8]) -> Self {
        let scratch = [0; SCRATCH_BUF_SIZE];

        Self {
            width,
            height,
            out_buf,
            scratch,
        }
    }

    pub(crate) fn clear(&mut self, premul_color: [u8; 4]) {
        if premul_color[0] == premul_color[1]
            && premul_color[1] == premul_color[2]
            && premul_color[2] == premul_color[3]
        {
            // All components are the same, so we can use memset instead.
            self.scratch.fill(premul_color[0]);
        } else {
            for z in self.scratch.chunks_exact_mut(COLOR_COMPONENTS) {
                z.copy_from_slice(&premul_color);
            }
        }
    }

    pub(crate) fn pack(&mut self, x: usize, y: usize) {
        pack(self.out_buf, &self.scratch, self.width, self.height, x, y);
    }

    pub(crate) fn run_cmd(&mut self, cmd: &Cmd, alphas: &[u32]) {
        match cmd {
            Cmd::Fill(f) => {
                self.fill(f.x as usize, f.width as usize, &f.paint);
            }
            Cmd::AlphaFill(s) => {
                let a_slice = &alphas[s.alpha_ix..];
                self.strip(s.x as usize, s.width as usize, a_slice, &s.paint);
            }
        }
    }

    pub(crate) fn fill(&mut self, x: usize, width: usize, paint: &Paint) {
        match paint {
            Paint::Solid(c) => {
                let color = c.premultiply().to_rgba8_fast();

                let target = &mut self.scratch[x * STRIP_HEIGHT_COMPONENTS..]
                    [..STRIP_HEIGHT_COMPONENTS * width];

                // If color is completely opaque we can just memcopy the colors.
                if color[3] == 255 {
                    for t in target.chunks_exact_mut(COLOR_COMPONENTS) {
                        t.copy_from_slice(&color);
                    }

                    return;
                }

                fill::src_over(target, &color);
            }
            _ => unimplemented!(),
        }
    }

    pub(crate) fn strip(&mut self, x: usize, width: usize, alphas: &[u32], paint: &Paint) {
        debug_assert!(
            alphas.len() >= width,
            "alpha buffer doesn't contain sufficient elements"
        );

        match paint {
            Paint::Solid(s) => {
                let color = s.premultiply().to_rgba8_fast();

                let target = &mut self.scratch[x * STRIP_HEIGHT_COMPONENTS..]
                    [..STRIP_HEIGHT_COMPONENTS * width];

                strip::src_over(target, &color, alphas);
            }
            _ => unimplemented!(),
        }
    }
}

fn pack(out_buf: &mut [u8], scratch: &ScratchBuf, width: usize, height: usize, x: usize, y: usize) {
    let base_ix = (y * STRIP_HEIGHT * width + x * WIDE_TILE_WIDTH) * COLOR_COMPONENTS;

    // Make sure we don't process rows outside the range of the pixmap.
    let max_height = (height - y * STRIP_HEIGHT).min(STRIP_HEIGHT);

    for j in 0..max_height {
        let line_ix = base_ix + j * width * COLOR_COMPONENTS;

        // Make sure we don't process columns outside the range of the pixmap.
        let max_width = (width - x * WIDE_TILE_WIDTH).min(WIDE_TILE_WIDTH);
        let target_len = max_width * COLOR_COMPONENTS;
        // This helps the compiler to understand that any access to `dest` cannot
        // be out of bounds, and thus saves corresponding checks in the for loop.
        let dest = &mut out_buf[line_ix..][..target_len];

        for i in 0..max_width {
            let src = &scratch[(i * STRIP_HEIGHT + j) * COLOR_COMPONENTS..][..COLOR_COMPONENTS];
            dest[i * COLOR_COMPONENTS..][..COLOR_COMPONENTS]
                .copy_from_slice(&src[..COLOR_COMPONENTS]);
        }
    }
}

pub(crate) mod fill {
    // See https://www.w3.org/TR/compositing-1/#porterduffcompositingoperators for the
    // formulas.

    use crate::fine::{COLOR_COMPONENTS, STRIP_HEIGHT_COMPONENTS};
    use crate::util::scalar::div_255;

    pub(crate) fn src_over(target: &mut [u8], src_c: &[u8; COLOR_COMPONENTS]) {
        let src_a = src_c[3] as u16;

        for strip in target.chunks_exact_mut(STRIP_HEIGHT_COMPONENTS) {
            for bg_c in strip.chunks_exact_mut(COLOR_COMPONENTS) {
                for i in 0..COLOR_COMPONENTS {
                    bg_c[i] = src_c[i] + div_255(bg_c[i] as u16 * (255 - src_a)) as u8;
                }
            }
        }
    }
}

pub(crate) mod strip {
    use crate::fine::{COLOR_COMPONENTS, STRIP_HEIGHT_COMPONENTS};
    use crate::util::scalar::div_255;
    use vello_common::strip::STRIP_HEIGHT;

    pub(crate) fn src_over(target: &mut [u8], src_c: &[u8; COLOR_COMPONENTS], alphas: &[u32]) {
        let src_a = src_c[3] as u16;

        for (bg_c, masks) in target.chunks_exact_mut(STRIP_HEIGHT_COMPONENTS).zip(alphas) {
            for j in 0..STRIP_HEIGHT {
                let mask_a = ((*masks >> (j * 8)) & 0xff) as u16;
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
