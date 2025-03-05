// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SIMD speedups

use crate::{
    fine::Fine,
    strip::{Strip, Tile},
};

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

// This block is when we have SIMD
#[cfg(target_arch = "aarch64")]
impl Fine<'_> {
    pub(crate) fn pack(&mut self, x: usize, y: usize) {
        if self.use_simd {
            self.pack_simd(x, y);
        } else {
            self.pack_scalar(x, y);
        }
    }

    pub(crate) fn clear(&mut self, color: [f32; 4]) {
        if self.use_simd {
            unsafe {
                self.clear_simd(color);
            }
        } else {
            self.clear_scalar(color);
        }
    }

    pub(crate) fn fill(&mut self, x: usize, width: usize, color: [f32; 4]) {
        if self.use_simd {
            unsafe {
                self.fill_simd(x, width, color);
            }
        } else {
            self.fill_scalar(x, width, color);
        }
    }

    pub(crate) fn strip(&mut self, x: usize, width: usize, alphas: &[u32], color: [f32; 4]) {
        if self.use_simd {
            unsafe {
                self.strip_simd(x, width, alphas, color);
            }
        } else {
            self.strip_scalar(x, width, alphas, color);
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn render_strips(tiles: &[Tile], strip_buf: &mut Vec<Strip>, alpha_buf: &mut Vec<u32>) {
    neon::render_strips_simd(tiles, strip_buf, alpha_buf);
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn render_strips(tiles: &[Tile], strip_buf: &mut Vec<Strip>, alpha_buf: &mut Vec<u32>) {
    crate::strip::render_strips_scalar(tiles, strip_buf, alpha_buf);
}

// This block is the fallback, no SIMD
#[cfg(not(target_arch = "aarch64"))]
impl Fine<'_> {
    pub(crate) fn pack(&mut self, x: usize, y: usize) {
        self.pack_scalar(x, y);
    }

    pub(crate) fn clear(&mut self, color: [f32; 4]) {
        self.clear_scalar(color);
    }

    pub(crate) fn fill(&mut self, x: usize, y: usize, color: [f32; 4]) {
        self.fill_scalar(x, y, color);
    }

    pub(crate) fn strip(&mut self, x: usize, width: usize, alphas: &[u32], color: [f32; 4]) {
        self.strip_scalar(x, width, alphas, color);
    }
}
