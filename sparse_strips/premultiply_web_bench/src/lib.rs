// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! WebAssembly entry points for the browser premultiplication benchmark.

use std::cell::RefCell;

use vello_common::fearless_simd::{Level, Simd, SimdBase, SimdInt, SimdMask, dispatch, mask8x16};
use vello_common::util::Div255Ext;

const DEFAULT_PIXEL_COUNT: usize = 800 * 500;
const MAX_PIXEL_COUNT: usize = 3840 * 2160;

thread_local! {
    static DATA: RefCell<Vec<u8>> = RefCell::new(make_source_data(DEFAULT_PIXEL_COUNT));
}

fn make_source_data(pixel_count: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(pixel_count * 4);
    for index in 0..pixel_count {
        let value = index.to_le_bytes()[0];
        let alpha = if index.is_multiple_of(8) { 128 } else { 255 };
        data.extend_from_slice(&[value, 255 - value, value / 2, alpha]);
    }
    data
}

/// Selects the dimensions used by subsequent benchmark calls.
///
/// Returns zero if the dimensions are empty, overflow, or exceed 3840 × 2160.
#[unsafe(no_mangle)]
pub extern "C" fn set_dimensions(width: usize, height: usize) -> u32 {
    let Some(pixel_count) = width.checked_mul(height) else {
        return 0;
    };
    if pixel_count == 0 || pixel_count > MAX_PIXEL_COUNT {
        return 0;
    }
    DATA.with_borrow_mut(|data| *data = make_source_data(pixel_count));
    1
}

/// Restores the benchmark image to its original straight-alpha pixels.
#[unsafe(no_mangle)]
pub extern "C" fn reset() {
    DATA.with_borrow_mut(|data| {
        for (index, pixel) in data.chunks_exact_mut(4).enumerate() {
            let value = index.to_le_bytes()[0];
            let alpha = if index.is_multiple_of(8) { 128 } else { 255 };
            pixel.copy_from_slice(&[value, 255 - value, value / 2, alpha]);
        }
    });
}

/// Runs the previous scalar premultiplication implementation.
#[unsafe(no_mangle)]
pub extern "C" fn run_old_scalar() -> u32 {
    DATA.with_borrow_mut(|data| u32::from(old_premultiply(data)))
}

/// Runs the new interleaved 64-byte premultiplication implementation.
#[unsafe(no_mangle)]
pub extern "C" fn run_new_interleaved_64() -> u32 {
    DATA.with_borrow_mut(|data| u32::from(new_premultiply_interleaved_64(data)))
}

/// Returns a pointer to the benchmark pixels in WebAssembly memory.
#[unsafe(no_mangle)]
pub extern "C" fn data_ptr() -> *const u8 {
    DATA.with_borrow(|data| data.as_ptr())
}

/// Returns the benchmark buffer length in bytes.
#[unsafe(no_mangle)]
pub extern "C" fn data_len() -> usize {
    DATA.with_borrow(|data| data.len())
}

/// Returns whether this module was built with WebAssembly SIMD enabled.
#[unsafe(no_mangle)]
pub extern "C" fn simd_enabled() -> u32 {
    u32::from(cfg!(target_feature = "simd128"))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "premultiplication always produces a value in the u8 range"
)]
#[inline(never)]
fn old_premultiply(data: &mut [u8]) -> bool {
    let mut may_have_transparency = false;
    for pixel in data.chunks_exact_mut(4) {
        let alpha = pixel[3];
        may_have_transparency |= alpha != 255;
        let alpha = u16::from(alpha);
        let premultiply = |component| (u16::from(component) * alpha / 255) as u8;
        pixel[0] = premultiply(pixel[0]);
        pixel[1] = premultiply(pixel[1]);
        pixel[2] = premultiply(pixel[2]);
    }
    may_have_transparency
}

#[inline(never)]
fn new_premultiply_interleaved_64(data: &mut [u8]) -> bool {
    let level = Level::try_detect().unwrap_or(Level::baseline());
    dispatch!(level, simd => new_premultiply_interleaved_64_impl(simd, data))
}

fn new_premultiply_interleaved_64_impl<S: Simd>(simd: S, data: &mut [u8]) -> bool {
    let (body, tail) = data.as_chunks_mut::<64>();
    let mut transparency = mask8x16::splat(simd, 0);
    for chunk in body {
        let rgba = simd.load_interleaved_128_u8x64(chunk);
        let (rg, ba) = simd.split_u8x64(rgba);
        let (r, g) = simd.split_u8x32(rg);
        let (b, a) = simd.split_u8x32(ba);

        transparency |= !a.simd_eq(255);
        let premultiply = |component| {
            let product = simd.widen_u8x16(component) * simd.widen_u8x16(a);
            simd.narrow_u16x16(product.div_255())
        };
        let premultiplied = simd.combine_u8x32(
            simd.combine_u8x16(premultiply(r), premultiply(g)),
            simd.combine_u8x16(premultiply(b), a),
        );
        simd.store_interleaved_128_u8x64(premultiplied, chunk);
    }

    let mut may_have_transparency = transparency.any_true();
    for pixel in tail.chunks_exact_mut(4) {
        let alpha = u16::from(pixel[3]);
        may_have_transparency |= alpha != 255;
        let premultiply = |component| ((u16::from(component) * alpha + 255) >> 8) as u8;
        pixel[0] = premultiply(pixel[0]);
        pixel[1] = premultiply(pixel[1]);
        pixel[2] = premultiply(pixel[2]);
    }
    may_have_transparency
}
