// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Fine rasterization

use crate::wide_tile::{Cmd, STRIP_HEIGHT, WIDE_TILE_WIDTH};

const STRIP_HEIGHT_F32: usize = STRIP_HEIGHT * 4;

pub(crate) struct Fine<'a> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    // rgba pixels
    pub(crate) out_buf: &'a mut [u8],
    // f32 RGBA pixels
    // That said, if we use u8, then this is basically a block of
    // untyped memory.
    pub(crate) scratch: Vec<[f32; WIDE_TILE_WIDTH * STRIP_HEIGHT * 4]>,
    #[allow(clippy::doc_markdown, reason = "false positive for x86_64")]
    /// Whether to use SIMD
    ///
    /// This is useful to toggle for performance evaluation reasons. It also
    /// *must* be false if runtime detection fails, otherwise we have safety
    /// problems. This is important for x86_64, as we'll be targeting Haswell
    /// as the minimum.
    #[allow(unused, reason = "some platforms might not have SIMD")]
    // The allow(unused) lint exception is because some platforms may not have
    // a SIMD implementation, and thus won't check the field.
    pub(crate) use_simd: bool,
}

impl<'a> Fine<'a> {
    pub(crate) fn new(width: usize, height: usize, out_buf: &'a mut [u8]) -> Self {
        let scratch = vec![[0.0; WIDE_TILE_WIDTH * STRIP_HEIGHT * 4]];
        Self {
            width,
            height,
            out_buf,
            scratch,
            use_simd: true,
        }
    }

    pub(crate) fn clear_scalar(&mut self, color: [f32; 4]) {
        let scratch = self.scratch.last_mut().unwrap();
        for z in scratch.chunks_exact_mut(4) {
            z.copy_from_slice(&color);
        }
    }

    pub(crate) fn pack_scalar(&mut self, x: usize, y: usize) {
        // Note that these can trigger if the method is called on a pixmap that
        // is not an integral multiple of the tile.
        assert!(
            (x + 1) * WIDE_TILE_WIDTH <= self.width,
            "overflow of pixmap width"
        );
        assert!(
            (y + 1) * STRIP_HEIGHT <= self.height,
            "overflow of pixmap height"
        );
        let scratch = self.scratch.last_mut().unwrap();
        let base_ix = (y * STRIP_HEIGHT * self.width + x * WIDE_TILE_WIDTH) * 4;
        for j in 0..STRIP_HEIGHT {
            let line_ix = base_ix + j * self.width * 4;
            for i in 0..WIDE_TILE_WIDTH {
                let mut rgba_f32 = [0.0; 4];
                rgba_f32.copy_from_slice(&scratch[(i * STRIP_HEIGHT + j) * 4..][..4]);
                let rgba_u8 = rgba_f32.map(|z| (z * 255.0).round() as u8);
                self.out_buf[line_ix + i * 4..][..4].copy_from_slice(&rgba_u8);
            }
        }
    }

    pub(crate) fn run_cmd(&mut self, cmd: &Cmd, alphas: &[u32]) {
        match cmd {
            Cmd::Fill(f) => {
                self.fill(f.x as usize, f.width as usize, f.color.components);
            }
            Cmd::Strip(s) => {
                let aslice = &alphas[s.alpha_ix..];
                self.strip(s.x as usize, s.width as usize, aslice, s.color.components);
            }
            Cmd::PushClip => self.scratch.push([0.0; WIDE_TILE_WIDTH * STRIP_HEIGHT * 4]),
            Cmd::PopClip => _ = self.scratch.pop(),
            Cmd::ClipFill(f) => {
                self.clip_fill_scalar(f.x as usize, f.width as usize);
            }
            Cmd::ClipStrip(s) => {
                let aslice = &alphas[s.alpha_ix..];
                self.clip_strip_scalar(s.x as usize, s.width as usize, aslice);
            }
        }
    }

    pub(crate) fn fill_scalar(&mut self, x: usize, width: usize, color: [f32; 4]) {
        let scratch = self.scratch.last_mut().unwrap();
        if color[3] == 1.0 {
            for z in scratch[x * STRIP_HEIGHT_F32..][..STRIP_HEIGHT_F32 * width].chunks_exact_mut(4)
            {
                z.copy_from_slice(&color);
            }
        } else {
            let one_minus_alpha = 1.0 - color[3];
            for z in scratch[x * STRIP_HEIGHT_F32..][..STRIP_HEIGHT_F32 * width].chunks_exact_mut(4)
            {
                for i in 0..4 {
                    //z[i] = color[i] + one_minus_alpha * z[i];
                    // Note: the mul_add will perform poorly on x86_64 default cpu target
                    // Probably right thing to do is craft a #cfg that detects fma, fcma, etc.
                    // What we really want is fmuladdf32 from intrinsics!
                    z[i] = z[i].mul_add(one_minus_alpha, color[i]);
                }
            }
        }
    }

    pub(crate) fn strip_scalar(&mut self, x: usize, width: usize, alphas: &[u32], color: [f32; 4]) {
        let scratch = self.scratch.last_mut().unwrap();
        debug_assert!(alphas.len() >= width, "overflow of alphas buffer");
        let cs = color.map(|z| z * (1.0 / 255.0));
        for (z, a) in scratch[x * STRIP_HEIGHT_F32..][..STRIP_HEIGHT_F32 * width]
            .chunks_exact_mut(16)
            .zip(alphas)
        {
            for j in 0..4 {
                let mask_alpha = ((*a >> (j * 8)) & 0xff) as f32;
                let one_minus_alpha = 1.0 - mask_alpha * cs[3];
                for i in 0..4 {
                    z[j * 4 + i] = z[j * 4 + i].mul_add(one_minus_alpha, mask_alpha * cs[i]);
                }
            }
        }
    }

    fn clip_fill_scalar(&mut self, x: usize, width: usize) {
        let (tos, rest) = self.scratch.split_last_mut().unwrap();
        let nos = rest.last_mut().unwrap();
        for i in 0..width {
            for j in 0..4 {
                let ix = (x + i) * STRIP_HEIGHT_F32 + j * 4;
                let one_minus_alpha = 1.0 - tos[ix + 3];
                for k in 0..4 {
                    nos[ix + k] = nos[ix + k].mul_add(one_minus_alpha, tos[ix + k]);
                }
            }
        }
    }

    fn clip_strip_scalar(&mut self, x: usize, width: usize, alphas: &[u32]) {
        let (tos, rest) = self.scratch.split_last_mut().unwrap();
        let nos = rest.last_mut().unwrap();
        for (i, a) in alphas.iter().take(width).enumerate() {
            for j in 0..4 {
                let ix = (x + i) * STRIP_HEIGHT_F32 + j * 4;
                let mask_alpha = ((a >> (j * 8)) & 0xff) as f32 * (1. / 255.);
                let one_minus_alpha = 1.0 - mask_alpha * tos[ix + 3];
                for k in 0..4 {
                    nos[ix + k] = nos[ix + k].mul_add(one_minus_alpha, mask_alpha * tos[ix + k]);
                }
            }
        }
    }
}
