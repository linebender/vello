// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SIMD speedups for Neon

use core::arch::aarch64::*;

use crate::{
    fine::Fine,
    strip::{Strip, Tile},
    tiling::Vec2,
    wide_tile::{STRIP_HEIGHT, WIDE_TILE_WIDTH},
};

impl Fine<'_> {
    pub(crate) unsafe fn clear_simd(&mut self, color: [f32; 4]) {
        let scratch = self.scratch.last_mut().unwrap();
        unsafe {
            let v_color = vld1q_f32(color.as_ptr());
            let v_color_4 = float32x4x4_t(v_color, v_color, v_color, v_color);
            for i in 0..WIDE_TILE_WIDTH {
                vst1q_f32_x4(scratch.as_mut_ptr().add(i * 16), v_color_4);
            }
        }
    }

    pub(crate) fn pack_simd(&mut self, x: usize, y: usize) {
        let scratch = self.scratch.last_mut().unwrap();
        unsafe fn cvt(v: float32x4_t) -> uint8x16_t {
            unsafe {
                let clamped = vminq_f32(v, vdupq_n_f32(1.0));
                let scaled = vmulq_f32(clamped, vdupq_n_f32(255.0));
                vreinterpretq_u8_u32(vcvtnq_u32_f32(scaled))
            }
        }

        unsafe fn cvt2(v0: float32x4_t, v1: float32x4_t) -> uint8x16_t {
            unsafe { vuzp1q_u8(cvt(v0), cvt(v1)) }
        }

        unsafe {
            let base_ix = (y * STRIP_HEIGHT * self.width + x * WIDE_TILE_WIDTH) * 4;
            for i in (0..WIDE_TILE_WIDTH).step_by(4) {
                let chunk_ix = base_ix + i * 4;
                let v0 = vld1q_f32_x4(scratch.as_ptr().add(i * 16));
                let v1 = vld1q_f32_x4(scratch.as_ptr().add((i + 1) * 16));
                let x0 = cvt2(v0.0, v1.0);
                let x1 = cvt2(v0.1, v1.1);
                let x2 = cvt2(v0.2, v1.2);
                let x3 = cvt2(v0.3, v1.3);
                let v2 = vld1q_f32_x4(scratch.as_ptr().add((i + 2) * 16));
                let v3 = vld1q_f32_x4(scratch.as_ptr().add((i + 3) * 16));
                let x4 = cvt2(v2.0, v3.0);
                let y0 = vuzp1q_u8(x0, x4);
                vst1q_u8(self.out_buf.as_mut_ptr().add(chunk_ix), y0);
                let x5 = cvt2(v2.1, v3.1);
                let y1 = vuzp1q_u8(x1, x5);
                vst1q_u8(self.out_buf.as_mut_ptr().add(chunk_ix + self.width * 4), y1);
                let x6 = cvt2(v2.2, v3.2);
                let y2 = vuzp1q_u8(x2, x6);
                vst1q_u8(self.out_buf.as_mut_ptr().add(chunk_ix + self.width * 8), y2);
                let x7 = cvt2(v2.3, v3.3);
                let y3 = vuzp1q_u8(x3, x7);
                vst1q_u8(
                    self.out_buf.as_mut_ptr().add(chunk_ix + self.width * 12),
                    y3,
                );
            }
        }
    }

    pub(crate) unsafe fn fill_simd(&mut self, x: usize, width: usize, color: [f32; 4]) {
        let scratch = self.scratch.last_mut().unwrap();
        unsafe {
            let v_color = vld1q_f32(color.as_ptr());
            let alpha = color[3];
            if alpha == 1.0 {
                let v_color_4 = float32x4x4_t(v_color, v_color, v_color, v_color);
                for i in x..x + width {
                    vst1q_f32_x4(scratch.as_mut_ptr().add(i * 16), v_color_4);
                }
            } else {
                let one_minus_alpha = vdupq_n_f32(1.0 - alpha);
                for i in x..x + width {
                    let ix = (x + i) * 16;
                    let mut v = vld1q_f32_x4(scratch.as_ptr().add(ix));
                    v.0 = vfmaq_f32(v_color, v.0, one_minus_alpha);
                    v.1 = vfmaq_f32(v_color, v.1, one_minus_alpha);
                    v.2 = vfmaq_f32(v_color, v.2, one_minus_alpha);
                    v.3 = vfmaq_f32(v_color, v.3, one_minus_alpha);
                    vst1q_f32_x4(scratch.as_mut_ptr().add(ix), v);
                }
            }
        }
    }

    #[inline(never)]
    pub(crate) unsafe fn strip_simd(
        &mut self,
        x: usize,
        width: usize,
        alphas: &[u32],
        color: [f32; 4],
    ) {
        let scratch = self.scratch.last_mut().unwrap();
        unsafe {
            debug_assert!(alphas.len() >= width, "overflow of alphas buffer");
            let v_color = vmulq_f32(vld1q_f32(color.as_ptr()), vdupq_n_f32(1.0 / 255.0));
            for i in 0..width {
                let a = *alphas.get_unchecked(i);
                // all this zipping compiles to tbl, we should probably just write that
                let a1 = vreinterpret_u8_u32(vdup_n_u32(a));
                let a2 = vreinterpret_u16_u8(vzip1_u8(a1, vdup_n_u8(0)));
                let a3 = vcombine_u16(a2, vdup_n_u16(0));
                let a4 = vreinterpretq_u32_u16(vzip1q_u16(a3, vdupq_n_u16(0)));
                let alpha = vcvtq_f32_u32(a4);
                let ix = (x + i) * 16;
                let mut v = vld1q_f32_x4(scratch.as_ptr().add(ix));
                let one_minus_alpha = vfmsq_laneq_f32(vdupq_n_f32(1.0), alpha, v_color, 3);
                v.0 = vfmaq_laneq_f32(vmulq_laneq_f32(v_color, alpha, 0), v.0, one_minus_alpha, 0);
                v.1 = vfmaq_laneq_f32(vmulq_laneq_f32(v_color, alpha, 1), v.1, one_minus_alpha, 1);
                v.2 = vfmaq_laneq_f32(vmulq_laneq_f32(v_color, alpha, 2), v.2, one_minus_alpha, 2);
                v.3 = vfmaq_laneq_f32(vmulq_laneq_f32(v_color, alpha, 3), v.3, one_minus_alpha, 3);
                vst1q_f32_x4(scratch.as_mut_ptr().add(ix), v);
            }
        }
    }
}

#[inline(never)]
pub(crate) fn render_strips_simd(
    tiles: &[Tile],
    strip_buf: &mut Vec<Strip>,
    alpha_buf: &mut Vec<u32>,
) {
    unsafe {
        strip_buf.clear();
        let mut strip_start = true;
        let mut cols = alpha_buf.len() as u32;
        let mut prev_tile = &tiles[0];
        let mut fp = prev_tile.footprint().0;
        let mut seg_start = 0;
        let mut delta = 0;
        // Note: the input should contain a sentinel tile, to avoid having
        // logic here to process the final strip.
        const IOTA: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
        let iota = vld1q_f32(IOTA.as_ptr());
        for i in 1..tiles.len() {
            let tile = &tiles[i];
            if prev_tile.loc() != tile.loc() {
                let start_delta = delta;
                let same_strip = prev_tile.loc().same_strip(&tile.loc());
                if same_strip {
                    fp |= 8;
                }
                let x0 = fp.trailing_zeros();
                let x1 = 32 - fp.leading_zeros();
                let mut areas = [[start_delta as f32; 4]; 4];
                for this_tile in &tiles[seg_start..i] {
                    // small gain possible here to unpack in simd, but llvm goes halfway
                    delta += this_tile.delta();
                    let p0 = Vec2::unpack(this_tile.p0);
                    let p1 = Vec2::unpack(this_tile.p1);
                    let slope = (p1.x - p0.x) / (p1.y - p0.y);
                    let vstarty = vsubq_f32(vdupq_n_f32(p0.y), iota);
                    let vy0 = vminq_f32(vmaxq_f32(vstarty, vdupq_n_f32(0.0)), vdupq_n_f32(1.0));
                    let vy1a = vsubq_f32(vdupq_n_f32(p1.y), iota);
                    let vy1 = vminq_f32(vmaxq_f32(vy1a, vdupq_n_f32(0.0)), vdupq_n_f32(1.0));
                    let vdy = vsubq_f32(vy0, vy1);
                    let mask = vceqzq_f32(vdy);
                    let vslope = vbslq_f32(mask, vdupq_n_f32(0.0), vdupq_n_f32(slope));
                    let vdy0 = vsubq_f32(vy0, vstarty);
                    let vdy1 = vsubq_f32(vy1, vstarty);
                    let mut vyedge = vdupq_n_f32(0.0);
                    if p0.x == 0.0 {
                        let ye = vsubq_f32(vdupq_n_f32(1.0), vstarty);
                        vyedge = vminq_f32(vmaxq_f32(ye, vdupq_n_f32(0.0)), vdupq_n_f32(1.0));
                    } else if p1.x == 0.0 {
                        let ye = vsubq_f32(vy1a, vdupq_n_f32(1.0));
                        vyedge = vminq_f32(vmaxq_f32(ye, vdupq_n_f32(-1.0)), vdupq_n_f32(0.0));
                    }
                    for x in x0..x1 {
                        let mut varea = vld1q_f32(areas.as_ptr().add(x as usize) as *const f32);
                        varea = vaddq_f32(varea, vyedge);
                        let vstartx = vdupq_n_f32(p0.x - x as f32);
                        let vxx0 = vfmaq_f32(vstartx, vdy0, vslope);
                        let vxx1 = vfmaq_f32(vstartx, vdy1, vslope);
                        let vxmin0 = vminq_f32(vxx0, vxx1);
                        let vxmax = vmaxq_f32(vxx0, vxx1);
                        let vxmin =
                            vsubq_f32(vminq_f32(vxmin0, vdupq_n_f32(1.0)), vdupq_n_f32(1e-6));
                        let vb = vminq_f32(vxmax, vdupq_n_f32(1.0));
                        let vc = vmaxq_f32(vb, vdupq_n_f32(0.0));
                        let vd = vmaxq_f32(vxmin, vdupq_n_f32(0.0));
                        let vd2 = vmulq_f32(vd, vd);
                        let vd2c2 = vfmsq_f32(vd2, vc, vc);
                        let vax = vfmaq_f32(vb, vd2c2, vdupq_n_f32(0.5));
                        let va = vdivq_f32(vsubq_f32(vax, vxmin), vsubq_f32(vxmax, vxmin));
                        varea = vfmaq_f32(varea, va, vdy);
                        vst1q_f32(areas.as_mut_ptr().add(x as usize) as *mut f32, varea);
                    }
                }
                for x in x0..x1 {
                    let mut alphas = 0_u32;
                    let varea = vld1q_f32(areas.as_ptr().add(x as usize) as *const f32);
                    let vnzw = vminq_f32(vabsq_f32(varea), vdupq_n_f32(1.0));
                    let vscaled = vmulq_f32(vnzw, vdupq_n_f32(255.0));
                    let vbits = vreinterpretq_u8_u32(vcvtnq_u32_f32(vscaled));
                    let vbits2 = vuzp1q_u8(vbits, vbits);
                    let vbits3 = vreinterpretq_u32_u8(vuzp1q_u8(vbits2, vbits2));
                    vst1q_lane_u32::<0>(&mut alphas, vbits3);
                    alpha_buf.push(alphas);
                }

                if strip_start {
                    let xy = (1 << 18) * prev_tile.y as u32 + 4 * prev_tile.x as u32 + x0;
                    let strip = Strip {
                        xy,
                        col: cols,
                        winding: start_delta,
                    };
                    strip_buf.push(strip);
                }
                cols += x1 - x0;
                fp = if same_strip { 1 } else { 0 };
                strip_start = !same_strip;
                seg_start = i;
                if !prev_tile.loc().same_row(&tile.loc()) {
                    delta = 0;
                }
            }
            fp |= tile.footprint().0;
            prev_tile = tile;
        }
    }
}
