// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Hand-written AVX2 baseline for the strip rasterizer's numeric kernels.

use core::arch::{asm, x86_64::__m128};
use core::mem::transmute;
use fearless_simd::{f32x4, u8x16, x86::Avx2};

#[repr(C, align(16))]
struct CoverageConstants {
    px_top: [f32; 4],
    px_bottom: [f32; 4],
    abs_mask: [u32; 4],
    neg_half: [f32; 4],
    right_1: [f32; 4],
    right_2: [f32; 4],
    right_3: [f32; 4],
    right_4: [f32; 4],
}

static COVERAGE_CONSTANTS: CoverageConstants = CoverageConstants {
    px_top: [0.0, 1.0, 2.0, 3.0],
    px_bottom: [1.0, 2.0, 3.0, 4.0],
    abs_mask: [0x7fff_ffff; 4],
    neg_half: [-0.5; 4],
    right_1: [1.0; 4],
    right_2: [2.0; 4],
    right_3: [3.0; 4],
    right_4: [4.0; 4],
};

#[repr(C)]
struct CoverageParams {
    line_top_y: f32,
    line_top_x: f32,
    line_bottom_y: f32,
    y_slope: f32,
    x_slope: f32,
    sign: f32,
    _padding: [f32; 2],
    line_px_base_yx: [f32; 4],
}

/// Update the four winding vectors without allowing LLVM to spill their live values.
#[inline(always)]
pub(crate) fn update_coverage(
    location_winding: &mut [f32x4<Avx2>; 4],
    line_top_y: f32,
    line_top_x: f32,
    line_bottom_y: f32,
    y_slope: f32,
    x_slope: f32,
    sign: f32,
) -> f32x4<Avx2> {
    let params = CoverageParams {
        line_top_y,
        line_top_x,
        line_bottom_y,
        y_slope,
        x_slope,
        sign,
        _padding: [0.0; 2],
        line_px_base_yx: [line_top_y.mul_add(-x_slope, line_top_x); 4],
    };
    let params = &raw const params;
    let constants = &raw const COVERAGE_CONSTANTS;

    // `Avx2::f32x4` is `__m128`; the SIMD token is zero-sized.
    let mut winding_0: __m128 = unsafe { transmute(location_winding[0]) };
    let mut winding_1: __m128 = unsafe { transmute(location_winding[1]) };
    let mut winding_2: __m128 = unsafe { transmute(location_winding[2]) };
    let mut winding_3: __m128 = unsafe { transmute(location_winding[3]) };
    let acc: __m128;

    macro_rules! column {
        ($winding:literal, $right_offset:literal) => {
            concat!(
                // Calculate this boundary independently. This is required for vertical lines:
                // incrementing an infinite slope would turn all later boundaries into NaN.
                "vmovaps xmm14, xmmword ptr [{constants} + ",
                $right_offset,
                "]\n",
                "vsubps xmm14, xmm14, xmm5\n",
                "vfmadd213ps xmm14, xmm8, xmm4\n",
                "vmaxps xmm14, xmm14, xmm6\n",
                "vminps xmm14, xmm14, xmm7\n",
                // h = abs(next_y - previous_y).
                "vsubps xmm15, xmm14, xmm10\n",
                "vandps xmm15, xmm15, xmmword ptr [{constants} + 32]\n",
                // Convert the clamped intersection to x and form the trapezoid.
                "vmovaps xmm10, xmm14\n",
                "vfmadd213ps xmm14, xmm9, xmmword ptr [{params} + 32]\n",
                "vaddps xmm13, xmm13, xmm14\n",
                "vmulps xmm13, xmm13, xmmword ptr [{constants} + 48]\n",
                "vaddps xmm13, xmm13, xmmword ptr [{constants} + ",
                $right_offset,
                "]\n",
                "vmulps xmm13, xmm13, xmm15\n",
                // winding += area * sign + horizontal-prefix accumulator.
                "vfmadd213ps xmm13, xmm11, xmm12\n",
                "vaddps xmm",
                $winding,
                ", xmm",
                $winding,
                ", xmm13\n",
                "vfmadd231ps xmm12, xmm15, xmm11\n",
                "vmovaps xmm13, xmm14\n",
            )
        };
    }

    unsafe {
        asm!(
            // Inputs and scanline-specific y bounds.
            "vbroadcastss xmm5, dword ptr [{params} + 0]",
            "vmaxps xmm6, xmm5, xmmword ptr [{constants} + 0]",
            "vbroadcastss xmm7, dword ptr [{params} + 8]",
            "vminps xmm7, xmm7, xmmword ptr [{constants} + 16]",
            "vmovaps xmm4, xmm5",
            "vbroadcastss xmm5, dword ptr [{params} + 4]",
            "vbroadcastss xmm8, dword ptr [{params} + 12]",
            "vbroadcastss xmm9, dword ptr [{params} + 16]",
            "vbroadcastss xmm11, dword ptr [{params} + 20]",
            "vxorps xmm12, xmm12, xmm12",
            // Calculate, clamp, and transform the x=0 intersection.
            "vxorps xmm10, xmm10, xmm10",
            "vsubps xmm10, xmm10, xmm5",
            "vfmadd213ps xmm10, xmm8, xmm4",
            "vmaxps xmm10, xmm10, xmm6",
            "vminps xmm10, xmm10, xmm7",
            "vmovaps xmm13, xmm10",
            "vfmadd213ps xmm13, xmm9, xmmword ptr [{params} + 32]",
            column!("0", "64"),
            column!("1", "80"),
            column!("2", "96"),
            column!("3", "112"),
            params = in(reg) params,
            constants = in(reg) constants,
            inout("xmm0") winding_0,
            inout("xmm1") winding_1,
            inout("xmm2") winding_2,
            inout("xmm3") winding_3,
            lateout("xmm4") _,
            lateout("xmm5") _,
            lateout("xmm6") _,
            lateout("xmm7") _,
            lateout("xmm8") _,
            lateout("xmm9") _,
            lateout("xmm10") _,
            lateout("xmm11") _,
            lateout("xmm12") acc,
            lateout("xmm13") _,
            lateout("xmm14") _,
            lateout("xmm15") _,
            options(nostack, readonly, preserves_flags),
        );
    }

    location_winding[0] = unsafe { transmute(winding_0) };
    location_winding[1] = unsafe { transmute(winding_1) };
    location_winding[2] = unsafe { transmute(winding_2) };
    location_winding[3] = unsafe { transmute(winding_3) };
    unsafe { transmute(acc) }
}

/// Convert four clamped f32 vectors directly to packed bytes.
#[inline(always)]
pub(crate) fn alpha_to_u8(values: [f32x4<Avx2>; 4]) -> u8x16<Avx2> {
    let mut value_0: __m128 = unsafe { transmute(values[0]) };
    let mut value_1: __m128 = unsafe { transmute(values[1]) };
    let mut value_2: __m128 = unsafe { transmute(values[2]) };
    let mut value_3: __m128 = unsafe { transmute(values[3]) };

    unsafe {
        asm!(
            "vinsertf128 ymm0, ymm0, xmm1, 1",
            "vinsertf128 ymm2, ymm2, xmm3, 1",
            "vcvttps2dq ymm0, ymm0",
            "vcvttps2dq ymm2, ymm2",
            "vextracti128 xmm1, ymm0, 1",
            "vextracti128 xmm3, ymm2, 1",
            "vpackusdw xmm0, xmm0, xmm1",
            "vpackusdw xmm2, xmm2, xmm3",
            "vpackuswb xmm0, xmm0, xmm2",
            inout("xmm0") value_0,
            inout("xmm1") value_1,
            inout("xmm2") value_2,
            inout("xmm3") value_3,
            options(nostack, nomem, preserves_flags),
        );
    }

    let _ = (value_1, value_2, value_3);
    unsafe { transmute(value_0) }
}
