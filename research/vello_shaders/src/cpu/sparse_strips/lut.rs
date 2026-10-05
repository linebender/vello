// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Port of Skia's `MSAA_LUT.h`, instantiated for 16 samples (`u16` masks, `MSAA16_PATTERN`).
//!
//! Naively, calculating MSAA coverage in software is O(n) per each sample location. Instead we use
//! a variation of a lookup technique introduced by Laine and Karras, adapted to 2D by Li et al.,
//! and refined by the Vello project:
//! <https://dl.acm.org/doi/abs/10.1111/j.1467-8659.2010.01728.x>,
//! <https://dl.acm.org/doi/10.1145/2980179.2982434>,
//! <https://github.com/linebender/vello/pull/64>.
//!
//! - To precompute the MSAA coverage of a line segment slicing through a 1×1 pixel, in Li et al.
//!   the line is parameterized by its normal vector (angle) and perpendicular distance from the
//!   center. While effective, angle and distance do not map linearly to the Cartesian pixel grid,
//!   leading to higher quantization errors (especially near corners), and computing the normal
//!   requires expensive square roots during rendering.
//! - Instead, we use a continuous parameterization introduced by the Vello project. We map the
//!   infinite range of standard slopes (m) and y-intercepts (b) into two new variables, s
//!   (normalized slope) and t (normalized translation), strictly bounded between 0.0 and 1.0.
//!
//! ## Squashing the slope (m → s)
//!
//! We squash the positive slope m in \[0, ∞) using `s = 1 / (m + 1)`: a vertical line
//! (m = ∞) maps to s = 0, a 45° line to s = 0.5, and a horizontal line to s = 1. To use this in the
//! half-plane equation `m*x - y + b >= 0`, we solve for `m = (1 - s) / s`. Substituting and
//! multiplying by s gives `x(1 - s) - y*s + b*s >= 0`.
//!
//! ## Squashing the translation (b → t)
//!
//! To keep the spatial offset in \[0, 1\], we scale the y-intercept against the new slope:
//! `t = (b + m) / (m + 1) = b*s + 1 - s`, so `b*s = s + t - 1`.
//!
//! ## Final equation
//!
//! Substituting `b*s` back into the half-plane equation and factoring yields
//! `(x - (1 - t))(1 - s) - (y - t)s >= 0`.
//!
//! ## LUT generation
//!
//! - For each sample location (x, y), we evaluate the final half-plane equation at the centers of
//!   the cells of a 64×64 grid (empirically sufficient for almost all rendering scenarios).
//! - Since the derivation assumes a positive slope (m ≥ 0), the LUT is split in two halves. The
//!   bottom half (`v >= LUT_MASK_HEIGHT / 2`) stores masks for positive slopes, the top half for
//!   negative slopes. For negative slopes, we reuse the same equation but flip the y axis of the
//!   sample points (`y = 1 - y`).
//! - The grid has [`LUT_MASK_WIDTH`] columns for the translation t, and `LUT_MASK_HEIGHT / 2` rows
//!   for the slope s. To minimize the maximum quantization error, s and t are taken from the center
//!   of each grid cell.
//! - Each cell holds a mask with bit k set when sample k is on the non-negative side of the line.
//!
//! The sample positions are [`MSAA16_PATTERN`] (Skia's `kMsaaPattern<uint16_t>`): sample k is at
//! `((MSAA16_PATTERN[k] + 0.5) / 16, (k + 0.5) / 16)`. This is an n-rooks pattern sorted by y, which
//! the truncation masks in the strip processor rely on.

use std::sync::OnceLock;

use super::{MSAA16_PATTERN, SAMPLES};

/// One bit per sample (Skia's `SparseStripConfig::SubSampleType`).
pub(super) type SubSampleType = u16;
/// Skia's `SparseStripConfig::kNumSubSamples`.
pub(super) const NUM_SUB_SAMPLES: usize = SubSampleType::BITS as usize;
const _: () = assert!(NUM_SUB_SAMPLES == SAMPLES && NUM_SUB_SAMPLES == MSAA16_PATTERN.len());

/// Number of t columns.
pub(super) const LUT_MASK_WIDTH: usize = 64;
/// Number of s rows, across both halves.
pub(super) const LUT_MASK_HEIGHT: usize = 64;
/// Largest column index.
pub(super) const LUT_MASK_WIDTH_EXCL: i32 = LUT_MASK_WIDTH as i32 - 1;

/// The LUT, indexed as `lut[v][u]` for slope row `v` and translation column `u`.
///
/// Skia stores it flat (`lut[v * kLutMaskWidth + u]`); the layout in memory is the same.
pub(super) type LutArray = [[SubSampleType; LUT_MASK_WIDTH]; LUT_MASK_HEIGHT];

/// Port of `MSAA_LUT::Make`.
pub(super) fn make() -> LutArray {
    let scale = 1.0 / NUM_SUB_SAMPLES as f32;
    let sub_x: [f32; NUM_SUB_SAMPLES] =
        std::array::from_fn(|k| (f32::from(MSAA16_PATTERN[k]) + 0.5) * scale);
    let sub_y: [f32; NUM_SUB_SAMPLES] = std::array::from_fn(|k| (k as f32 + 0.5) * scale);

    let mut lut = [[0; LUT_MASK_WIDTH]; LUT_MASK_HEIGHT];
    let half_height = LUT_MASK_HEIGHT / 2;
    let half_height_f = half_height as f32;

    for (v, lut_row) in lut.iter_mut().enumerate() {
        let is_pos = v >= half_height;
        for (u, entry) in lut_row.iter_mut().enumerate() {
            // Extract continuous parameters from the center of the grid cells.
            let t = (u as f32 + 0.5) / LUT_MASK_WIDTH as f32;
            let s = ((v % half_height) as f32 + 0.5) / half_height_f;

            let mut mask: SubSampleType = 0;
            for (k, (&x, &y)) in sub_x.iter().zip(&sub_y).enumerate() {
                let y = if is_pos { y } else { 1.0 - y };
                let val = (x - (1.0 - t)) * (1.0 - s) - (y - t) * s;
                if val >= 0.0 {
                    mask |= 1 << k;
                }
            }
            *entry = mask;
        }
    }
    lut
}

/// The LUT, generated on first use (Skia's `GenerateMSAALUT`, held by the caller).
pub(super) fn msaa_lut() -> &'static LutArray {
    static LUT: OnceLock<LutArray> = OnceLock::new();
    LUT.get_or_init(make)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample positions within the pixel.
    fn sample(k: usize) -> (f32, f32) {
        (
            (f32::from(MSAA16_PATTERN[k]) + 0.5) / 16.0,
            (k as f32 + 0.5) / 16.0,
        )
    }

    #[test]
    fn pattern_is_n_rooks() {
        let mut seen = [false; 16];
        for &p in &MSAA16_PATTERN {
            assert!(!seen[usize::from(p)], "column {p} used twice");
            seen[usize::from(p)] = true;
        }
    }

    #[test]
    fn coverage_grows_with_t() {
        // Moving the line left of the samples (larger t) only adds coverage.
        let lut = msaa_lut();
        for row in lut {
            for pair in row.windows(2) {
                assert_eq!(pair[0] & !pair[1], 0, "{row:?}");
            }
            assert_eq!(row[0], 0, "{row:?}");
            assert_eq!(row[LUT_MASK_WIDTH - 1], 0xffff, "{row:?}");
        }
    }

    #[test]
    fn near_vertical_row_is_right_half_plane() {
        // Row 32 (positive slope, s = 1/64) at t covers the samples right of x = 1 - t, with the
        // line tilted by 1/63 across the pixel.
        let row = &msaa_lut()[LUT_MASK_HEIGHT / 2];
        for (u, &mask) in row.iter().enumerate() {
            let t = (u as f32 + 0.5) / 64.0;
            for k in 0..16 {
                let (x, y) = sample(k);
                let threshold = 1.0 - t + (y - t) / 63.0;
                if (x - threshold).abs() < 1e-4 {
                    // Exact ties exist (e.g. u = 22, k = 13); rounding decides those.
                    continue;
                }
                assert_eq!(mask & (1 << k) != 0, x >= threshold, "u {u} k {k}");
            }
        }
        // Half a pixel: exactly the 8 samples with MSAA16_PATTERN[k] >= 8.
        let half = row[32];
        for (k, &p) in MSAA16_PATTERN.iter().enumerate() {
            assert_eq!(half & (1 << k) != 0, p >= 8, "k {k}");
        }
    }
}
