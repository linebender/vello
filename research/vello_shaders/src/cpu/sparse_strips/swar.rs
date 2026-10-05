// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SWAR helpers standing in for the skvx operations used by Skia's strip processor.
//! A pixel holds 16 signed 8-bit per-sample winding lanes.
//!
//! Lane `k` of a [`SwarPixel`] is bits `8 * k .. 8 * k + 8` and holds the winding of sample `k`.
//! Like `skvx::Vec<N, uint8_t>`, lane arithmetic wraps modulo 256 and never carries into the
//! neighboring lane, so a plain `+` or `-` across lanes is never used.

/// The per-sample winding of one pixel, 16 lanes of 8 bits (Skia's `SwarPixel` / `PixelBytes`).
pub(super) type SwarPixel = u128;

/// `0x01` in every lane.
pub(super) const ONES: SwarPixel = SwarPixel::MAX / 0xff;
/// `0x7f` in every lane.
const LOW7: SwarPixel = ONES * 0x7f;
/// `0x80` in every lane.
const HIGH: SwarPixel = ONES * 0x80;

/// `b` in every lane.
#[inline(always)]
pub(super) const fn splat(b: u8) -> SwarPixel {
    ONES * b as SwarPixel
}

/// Lane-wise wrapping `a + b`.
#[inline(always)]
pub(super) const fn add(a: SwarPixel, b: SwarPixel) -> SwarPixel {
    // Add the low 7 bits of every lane, which can't carry out of the lane. The top bit of each
    // lane is then the XOR of both top bits and the carry out of bit 6, which is already in it.
    ((a & LOW7) + (b & LOW7)) ^ ((a ^ b) & HIGH)
}

/// Lane-wise wrapping `a - b`.
#[inline(always)]
pub(super) const fn sub(a: SwarPixel, b: SwarPixel) -> SwarPixel {
    // Subtract the low 7 bits of every lane from `a` with the top bits set, so that no borrow
    // leaves the lane. The top bit of each lane then holds `!borrow`, and the true top bit is
    // `a7 ^ b7 ^ borrow`.
    ((a | HIGH) - (b & LOW7)) ^ ((a ^ !b) & HIGH)
}

/// `EXPAND[b]` has byte `j` set to `0xff` iff bit `j` of `b` is set.
static EXPAND: [u64; 256] = {
    let mut table = [0_u64; 256];
    let mut b = 0;
    while b < 256 {
        let mut j = 0;
        while j < 8 {
            if b & (1 << j) != 0 {
                table[b] |= 0xff_u64 << (8 * j);
            }
            j += 1;
        }
        b += 1;
    }
    table
};

/// Expand a sample mask to lanes: lane `k` is `0xff` if bit `k` of `mask` is set, else `0`.
///
/// Stands in for skvx's `(PixelBytes(maskVal) & vBit) != 0`.
#[inline(always)]
pub(super) fn expand_mask(mask: u16) -> SwarPixel {
    let lo = EXPAND[usize::from(mask & 0xff)];
    let hi = EXPAND[usize::from(mask >> 8)];
    SwarPixel::from(lo) | (SwarPixel::from(hi) << 64)
}

/// Multiplying a `u64` whose bytes are each `0` or `1` by this gathers byte `j` into bit `56 + j`.
/// All partial products land on distinct bits, so nothing carries into the top byte.
const GATHER: u64 = 0x0102_0408_1020_4080;
const ONES64: u64 = 0x0101_0101_0101_0101;
const LOW7_64: u64 = 0x7f7f_7f7f_7f7f_7f7f;
const HIGH64: u64 = 0x8080_8080_8080_8080;

/// Gather 8 byte flags (each byte `0` or `1`) into bits: byte `j` to bit `j`.
#[inline(always)]
const fn gather8(flags: u64) -> u16 {
    (flags.wrapping_mul(GATHER) >> 56) as u16
}

/// Byte flags of a `u64`: `1` where the byte is non-zero.
#[inline(always)]
const fn nonzero8(y: u64) -> u64 {
    // Adding 0x7f to the low 7 bits sets bit 7 iff they are non-zero, without carrying out.
    ((((y & LOW7_64) + LOW7_64) | y) & HIGH64) >> 7
}

/// Bit `k` is set iff lane `k` of `a` differs from lane `k` of `b`.
#[inline(always)]
pub(super) const fn ne_mask(a: SwarPixel, b: SwarPixel) -> u16 {
    let x = a ^ b;
    gather8(nonzero8(x as u64)) | (gather8(nonzero8((x >> 64) as u64)) << 8)
}

/// Bit `k` is set iff lane `k` of `x` is odd.
#[inline(always)]
pub(super) const fn odd_mask(x: SwarPixel) -> u16 {
    gather8(x as u64 & ONES64) | (gather8((x >> 64) as u64 & ONES64) << 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::sparse_strips::tests::XorShift;

    fn lanes(x: SwarPixel) -> [u8; 16] {
        x.to_le_bytes()
    }

    fn random_pixel(rng: &mut XorShift) -> SwarPixel {
        let mut bytes = [0_u8; 16];
        for b in &mut bytes {
            // Bias towards the interesting values around 0, 0x7f/0x80 and 0xff.
            let r = (rng.next_f32() * 256.0) as u8;
            *b = match r % 4 {
                0 => r,
                1 => 0x80_u8.wrapping_add(r % 5).wrapping_sub(2),
                2 => r % 3,
                _ => 0xff_u8.wrapping_sub(r % 3),
            };
        }
        SwarPixel::from_le_bytes(bytes)
    }

    #[test]
    fn add_sub_are_lanewise() {
        let mut rng = XorShift(0x1234_5678_9abc_def1);
        for _ in 0..10_000 {
            let (a, b) = (random_pixel(&mut rng), random_pixel(&mut rng));
            let (sum, diff) = (lanes(add(a, b)), lanes(sub(a, b)));
            for (((&s, &d), &x), &y) in sum.iter().zip(&diff).zip(&lanes(a)).zip(&lanes(b)) {
                assert_eq!(s, x.wrapping_add(y), "{a:x} + {b:x}");
                assert_eq!(d, x.wrapping_sub(y), "{a:x} - {b:x}");
            }
        }
    }

    #[test]
    fn masks_round_trip() {
        for m in 0..=u16::MAX {
            let e = expand_mask(m);
            for (k, lane) in lanes(e).into_iter().enumerate() {
                assert_eq!(lane, if m & (1 << k) != 0 { 0xff } else { 0 });
            }
            assert_eq!(ne_mask(e, 0), m);
            assert_eq!(ne_mask(e ^ splat(0x80), splat(0x80)), m);
            assert_eq!(odd_mask(e), m);
            assert_eq!(odd_mask(e & ONES), m);
        }
    }

    #[test]
    fn ne_and_odd_masks() {
        let mut rng = XorShift(0x0fed_cba9_8765_4321);
        for _ in 0..10_000 {
            let (a, b) = (random_pixel(&mut rng), random_pixel(&mut rng));
            let mut ne = 0_u16;
            let mut odd = 0_u16;
            for (k, (&x, &y)) in lanes(a).iter().zip(&lanes(b)).enumerate() {
                ne |= u16::from(x != y) << k;
                odd |= u16::from(x & 1) << k;
            }
            assert_eq!(ne_mask(a, b), ne);
            assert_eq!(odd_mask(a), odd);
        }
    }
}
