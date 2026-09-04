// Copyright 2025 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility helper functions.

use core::ops::Sub;
#[cfg(not(feature = "std"))]
use core_maths::CoreFloat as _;
use peniko::kurbo::Affine;

// From <https://github.com/linebender/tiny-skia/blob/68b198a7210a6bbf752b43d6bc4db62445730313/path/src/scalar.rs#L12>
const SCALAR_NEARLY_ZERO_F32: f32 = 1.0 / (1 << 12) as f32;
const SCALAR_NEARLY_ZERO_F64: f64 = 1.0 / (1 << 12) as f64;

/// A number of useful methods for f32 numbers.
pub(crate) trait FloatExt: Sized + Sub<f32, Output = f32> {
    /// Whether the number is approximately 0.
    fn is_nearly_zero(&self) -> bool {
        self.is_nearly_zero_within_tolerance(SCALAR_NEARLY_ZERO_F32)
    }

    /// Whether the number is approximately 0, with a given tolerance.
    fn is_nearly_zero_within_tolerance(&self, tolerance: f32) -> bool;
}

impl FloatExt for f32 {
    #[inline(always)]
    fn is_nearly_zero_within_tolerance(&self, tolerance: f32) -> bool {
        debug_assert!(tolerance >= 0.0, "tolerance must be positive");

        self.abs() <= tolerance
    }
}

pub(crate) trait AffineExt {
    /// Whether the transform has any skewing coefficient.
    fn has_skew(&self) -> bool;

    /// Whether the transform has a vertical skew.
    fn has_vertical_skew(&self) -> bool;

    /// Whether the transform has positive, uniform scaling factors and no skew.
    fn is_positive_uniform_scale_without_skew(&self) -> bool;

    /// Whether the transform has positive, uniform scaling factors and no vertical skew.
    fn is_positive_uniform_scale_without_vertical_skew(&self) -> bool;

    /// Whether the transform has non-unit scale or skew.
    ///
    /// Note that negative scales (i.e. -1.0) are explicitly allowed.
    fn has_non_unit_skew_or_scale(&self) -> bool;
}

impl AffineExt for Affine {
    #[inline]
    fn has_skew(&self) -> bool {
        let [_, b, c, _, _, _] = self.as_coeffs();
        b.abs() > SCALAR_NEARLY_ZERO_F64 || c.abs() > SCALAR_NEARLY_ZERO_F64
    }

    #[inline]
    fn is_positive_uniform_scale_without_skew(&self) -> bool {
        let [a, _, _, d, _, _] = self.as_coeffs();
        (a - d).abs() <= SCALAR_NEARLY_ZERO_F64 && a > 0.0 && d > 0.0 && !self.has_skew()
    }

    #[inline]
    fn has_non_unit_skew_or_scale(&self) -> bool {
        let [a, _, _, d, _, _] = self.as_coeffs();
        self.has_skew()
            || (1.0 - a.abs()).abs() > SCALAR_NEARLY_ZERO_F64
            || (1.0 - d.abs()).abs() > SCALAR_NEARLY_ZERO_F64
    }

    /// Whether the transform has a vertical skew.
    #[inline]
    fn has_vertical_skew(&self) -> bool {
        let [_, b, _, _, _, _] = self.as_coeffs();
        b.abs() > SCALAR_NEARLY_ZERO_F64
    }

    #[inline]
    fn is_positive_uniform_scale_without_vertical_skew(&self) -> bool {
        let [a, _, _, d, _, _] = self.as_coeffs();
        (a - d).abs() <= SCALAR_NEARLY_ZERO_F64 && a > 0.0 && d > 0.0 && !self.has_vertical_skew()
    }
}

/// Bias added before every discrete glyph-placement decision.
///
/// Glyph placement makes per-pixel decisions (the floor/fract split behind the
/// atlas quad position and subpixel-bucket key, and the hinting baseline
/// round) from an `f64` translation. A consumer that produces the same content
/// twice — a live pass and a recorded or cached one — computes that
/// translation through different floating-point op orders and gets answers
/// agreeing only to ~1e-12, while glyph origins routinely sit exactly on the
/// decision boundary (integer coordinates; exact halves under fractional
/// scale factors). A bare `floor`/`round` lets that last-ulp noise decide the
/// pixel, so the two passes can place the same glyph one device pixel apart.
///
/// Biasing every decision by 1e-6 — far above the noise, far below the
/// 0.25-px subpixel buckets — moves the boundary off those structured values,
/// so any two evaluations agreeing within 1e-6 decide identically. All sites
/// operate in `f64` (ulp ~1.8e-12 at 16384 px), so the bias survives any
/// realistic coordinate.
pub(crate) const PIXEL_DECISION_BIAS: f64 = 1e-6;

/// Biased floor: the integer half of a glyph pixel decision. Pair with
/// [`biased_fract`] on the same value — mixing a biased and a raw operation
/// re-opens the 1-px gap at integer boundaries.
#[inline]
pub(crate) fn biased_floor(v: f64) -> f64 {
    (v + PIXEL_DECISION_BIAS).floor()
}

/// Biased fractional part in `[0, 1]`: the subpixel half of the decision,
/// consistent with [`biased_floor`] by construction. The closed upper bound
/// is real: for inputs a hair below a multiple of the bias, `b - b.floor()`
/// rounds to exactly `1.0` — consumers must clamp, not wrap.
#[inline]
pub(crate) fn biased_fract(v: f64) -> f64 {
    let b = v + PIXEL_DECISION_BIAS;
    b - b.floor()
}

/// Biased round, half up (`floor(v + 0.5 + bias)`): the hinting baseline
/// snap. Half-up rather than `f64::round`'s half-away-from-zero, so exact
/// negative halves decide the same way as positive ones.
#[inline]
pub(crate) fn biased_round_half_up(v: f64) -> f64 {
    (v + (0.5 + PIXEL_DECISION_BIAS)).floor()
}

#[cfg(test)]
mod tests {
    use super::AffineExt;
    use super::{PIXEL_DECISION_BIAS, biased_floor, biased_fract, biased_round_half_up};
    use peniko::kurbo::Affine;

    #[test]
    fn detects_positive_uniform_scale_without_skew() {
        let transform = Affine::scale(2.0);

        assert!(!transform.has_skew());
        assert!(!transform.has_vertical_skew());
        assert!(transform.is_positive_uniform_scale_without_skew());
        assert!(transform.is_positive_uniform_scale_without_vertical_skew());
    }

    #[test]
    fn rejects_positive_uniform_scale_without_skew_when_horizontally_skewed() {
        let transform = Affine::new([2.0, 0.0, 0.25, 2.0, 0.0, 0.0]);

        assert!(transform.has_skew());
        assert!(!transform.has_vertical_skew());
        assert!(!transform.is_positive_uniform_scale_without_skew());
        assert!(transform.is_positive_uniform_scale_without_vertical_skew());
    }

    #[test]
    fn rejects_positive_uniform_scale_without_vertical_skew_when_vertically_skewed() {
        let transform = Affine::new([2.0, 0.25, 0.0, 2.0, 0.0, 0.0]);

        assert!(transform.has_skew());
        assert!(transform.has_vertical_skew());
        assert!(!transform.is_positive_uniform_scale_without_skew());
        assert!(!transform.is_positive_uniform_scale_without_vertical_skew());
    }

    #[test]
    fn rejects_non_uniform_or_non_positive_scale() {
        let non_uniform = Affine::new([2.0, 0.0, 0.0, 3.0, 0.0, 0.0]);
        let flipped = Affine::new([-2.0, 0.0, 0.0, -2.0, 0.0, 0.0]);

        assert!(non_uniform.has_non_unit_skew_or_scale());
        assert!(!non_uniform.is_positive_uniform_scale_without_skew());
        assert!(!non_uniform.is_positive_uniform_scale_without_vertical_skew());
        assert!(flipped.has_non_unit_skew_or_scale());
        assert!(!flipped.is_positive_uniform_scale_without_skew());
        assert!(!flipped.is_positive_uniform_scale_without_vertical_skew());
    }

    #[test]
    fn allows_unit_axis_flips() {
        let flip_x = Affine::new([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let flip_y = Affine::new([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]);
        let flip_xy = Affine::new([-1.0, 0.0, 0.0, -1.0, 0.0, 0.0]);

        assert!(!flip_x.has_non_unit_skew_or_scale());
        assert!(!flip_y.has_non_unit_skew_or_scale());
        assert!(!flip_xy.has_non_unit_skew_or_scale());
    }

    /// Producing the same content twice (a live pass and a recorded or cached
    /// one) yields translations that differ by last-ulp noise. The discrete
    /// decisions must not flip on that noise at the values glyph origins
    /// actually sit on: exact integers and exact halves.
    #[test]
    fn pixel_decisions_are_stable_under_last_ulp_noise() {
        // Far above the real cross-evaluation divergence, far below the bias.
        const DUST: f64 = 1e-9;
        // Integer boundaries: the floor/fract split.
        for k in [-1234.0, -3.0, 0.0, 7.0, 341.0, 4096.0] {
            for d in [-DUST, 0.0, DUST] {
                assert_eq!(biased_floor(k + d), k, "floor stable at {k} + {d}");
                assert!(
                    biased_fract(k + d) < 0.125,
                    "fract in bucket 0 at {k} + {d}"
                );
            }
        }
        // Half boundaries: the hinting round, half-up, including exact
        // negative halves.
        for k in [-12.5, -0.5, 0.5, 33.5, 2047.5] {
            for d in [-DUST, 0.0, DUST] {
                assert_eq!(
                    biased_round_half_up(k + d),
                    k + 0.5,
                    "half-up stable at {k} + {d}"
                );
            }
        }
        // The split reassembles its input (modulo the bias); one biased and
        // one raw operation would not.
        for v in [-3.0, -0.25, 0.0, 9.0, 9.75, 341.5] {
            let rebuilt = biased_floor(v) + biased_fract(v);
            assert!(
                (rebuilt - v).abs() <= 2.0 * PIXEL_DECISION_BIAS,
                "split reassembles {v}"
            );
        }
    }
}
