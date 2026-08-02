// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for [`CoverageContrast`], the coverage transfer applied to alpha-mask
//! (glyph) coverage:
//!
//! ```text
//! a' = a + c * a * (1 - a) * (2a - 1) + w * a * (1 - a)
//! ```
//!
//! Pure arithmetic; nothing here needs a GPU. `render.wesl` and `vello_cpu`'s
//! fine stages evaluate the same expression in the same order, so the
//! properties proven here hold for all three consumers.

use vello_common::paint::{Color, CoverageContrast};

/// A full `smoothstep`, i.e. the `c = 1` end of the steepening family.
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Weight strengths sampled when sweeping the second axis; the `c` axis is
/// cheap enough to sweep exhaustively.
const WEIGHT_SAMPLES: [u8; 6] = [0, 1, 64, 128, 191, 255];

/// The default must be the identity bit-for-bit, not merely to within a
/// tolerance: the feature is opt-in, and every existing snapshot must keep its
/// exact bytes while it is off.
#[test]
fn none_is_bit_exact_identity() {
    let none = CoverageContrast::NONE;
    assert!(none.is_none());
    assert_eq!(none, CoverageContrast::default());
    assert_eq!(none.contrast_bits(), 0);
    assert_eq!(none.weight_bits(), 0);

    for byte in 0..=u8::MAX {
        assert_eq!(none.apply_u8(byte), byte, "u8 identity at {byte}");
    }

    // Compare bit patterns so a `-0.0` or one-ulp drift would fail.
    for i in 0..=100_000_u32 {
        let a = i as f32 / 100_000.0;
        assert_eq!(none.apply(a).to_bits(), a.to_bits(), "f32 identity at {a}");
    }

    // Either term alone disqualifies `is_none`.
    assert!(!CoverageContrast::from_bits(1, 0).is_none());
    assert!(!CoverageContrast::from_bits(0, 1).is_none());
}

/// Fully-covered and fully-empty pixels must be fixed points at every strength
/// pair: interiors and background stay byte-identical, only the transition
/// band moves.
#[test]
fn endpoints_are_fixed_points_at_every_strength() {
    for contrast in 0..=u8::MAX {
        for weight in WEIGHT_SAMPLES {
            let c = CoverageContrast::from_bits(contrast, weight);
            assert_eq!(c.apply(0.0).to_bits(), 0.0_f32.to_bits());
            assert_eq!(c.apply(1.0).to_bits(), 1.0_f32.to_bits());
            assert_eq!(c.apply_u8(0), 0);
            assert_eq!(c.apply_u8(255), 255);
        }
        // At `w = 0` the midpoint is a pivot: stem weight is preserved. (With
        // `w > 0` it deliberately is not — see `weight_term_adds_weight`.)
        let c = CoverageContrast::from_bits(contrast, 0);
        assert_eq!(c.apply(0.5).to_bits(), 0.5_f32.to_bits());
    }
}

/// The `c` term is symmetric about `a = 0.5`: it steepens the transition
/// without moving the 50% crossing, so stems keep their weight. Weight is the
/// separate, deliberately asymmetric `w` axis.
#[test]
fn steepening_term_is_symmetric_about_the_midpoint() {
    for bits in [0_u8, 1, 64, 128, 200, 255] {
        let c = CoverageContrast::from_bits(bits, 0);
        for i in 0..=1000_u32 {
            let a = i as f32 / 1000.0;
            let lhs = c.apply(1.0 - a);
            let rhs = 1.0 - c.apply(a);
            assert!(
                (lhs - rhs).abs() < 1e-6,
                "symmetry at a={a}, bits={bits}: {lhs} vs {rhs}"
            );
        }
    }
}

/// Monotone and within `[0, 1]` for every constructible pair, so no consumer
/// needs a clamp and coverage can never invert.
#[test]
fn combined_curve_is_monotone_and_in_range() {
    for contrast in 0..=u8::MAX {
        for weight in WEIGHT_SAMPLES {
            let c = CoverageContrast::from_bits(contrast, weight);
            let mut prev = c.apply(0.0);
            for i in 0..=2000_u32 {
                let a = i as f32 / 2000.0;
                let v = c.apply(a);
                assert!(
                    (0.0..=1.0).contains(&v),
                    "range at a={a}, ({contrast}, {weight}): {v}"
                );
                assert!(
                    v >= prev - 1e-7,
                    "monotone at a={a}, ({contrast}, {weight}): {v} < {prev}"
                );
                prev = v;
            }

            let mut prev_u8 = c.apply_u8(0);
            for byte in 0..=u8::MAX {
                let v = c.apply_u8(byte);
                assert!(
                    v >= prev_u8,
                    "u8 monotone at {byte}, ({contrast}, {weight})"
                );
                prev_u8 = v;
            }
        }
    }

    // The derivative `1 + c/2 - 6c*t^2 - 2w*t` (with `t = a - 1/2`) is
    // concave, minimized at the `a = 1` endpoint where it equals `1 - c - w` —
    // exactly zero for every capped boundary pair. Probe densely just below
    // `a = 1`, where the slope vanishes.
    for contrast in [1_u8, 64, 128, 191, 254] {
        let c = CoverageContrast::from_bits(contrast, 255); // caps to the boundary
        assert_eq!(
            u32::from(c.contrast_bits()) + u32::from(c.weight_bits()),
            255,
            "boundary pair"
        );
        for i in 0..=10_000_u32 {
            let a = 1.0 - i as f32 / 1_000_000.0;
            let v = c.apply(a);
            assert!(
                (0.0..=1.0).contains(&v),
                "range near saturation at a={a}, contrast={contrast}: {v}"
            );
        }
    }
}

/// The peak slope of the `c` term (at the 50% crossing) is `1 + c/2`, so the
/// steepening family spans exact-area coverage (1.0 alpha/px) through a full
/// smoothstep (1.5 alpha/px).
#[test]
fn peak_slope_is_one_plus_half_the_strength() {
    const H: f32 = 1e-3;
    for bits in [0_u8, 51, 128, 191, 255] {
        let c = CoverageContrast::from_bits(bits, 0);
        let slope = (c.apply(0.5 + H) - c.apply(0.5 - H)) / (2.0 * H);
        let expected = 1.0 + c.contrast_strength() / 2.0;
        assert!(
            (slope - expected).abs() < 1e-3,
            "peak slope at bits={bits}: {slope} vs {expected}"
        );
    }

    let full = CoverageContrast::from_bits(255, 0);
    assert!((full.contrast_strength() - 1.0).abs() < 1e-6);
    for i in 0..=100_u32 {
        let a = i as f32 / 100.0;
        assert!(
            (full.apply(a) - smoothstep(a)).abs() < 1e-6,
            "c=1 must be a full smoothstep at a={a}"
        );
    }
}

/// Control for the lerp parameterisation: the "rescaled smoothstep" family
/// `S(0.5 + (a - 0.5) * k)` does not contain the identity for any `k`, so it
/// could not be defaulted off. The family's midpoint slope is `1.5 * k`, so
/// the only candidate is `k = 2/3`, which fails away from the midpoint.
#[test]
fn rescaled_smoothstep_family_cannot_express_the_identity() {
    let rescaled = |k: f32, a: f32| smoothstep((0.5 + (a - 0.5) * k).clamp(0.0, 1.0));

    // k = 1 reads as "no rescaling" but is a full smoothstep.
    assert!((rescaled(1.0, 0.25) - 0.15625).abs() < 1e-6);
    // The midpoint-slope candidate still deviates by more than an 8-bit level.
    assert!((rescaled(2.0 / 3.0, 0.25) - 0.25).abs() > 1.0 / 255.0);

    // Exhaustively: no k comes within half an 8-bit level of the identity
    // everywhere (the closest, ~0.031, is at k ≈ 0.79).
    for i in 0..=2000_u32 {
        let k = i as f32 / 1000.0;
        let worst = (0..=100_u32)
            .map(|j| {
                let a = j as f32 / 100.0;
                (rescaled(k, a) - a).abs()
            })
            .fold(0.0_f32, f32::max);
        assert!(worst > 0.5 / 255.0, "k={k} must not be the identity");
    }

    // The family actually used contains the identity exactly.
    assert_eq!(
        CoverageContrast::NONE.apply(0.25).to_bits(),
        0.25_f32.to_bits()
    );
}

#[test]
fn from_strength_clamps_quantises_and_round_trips() {
    assert_eq!(CoverageContrast::from_strength(0.0), CoverageContrast::NONE);
    assert_eq!(CoverageContrast::from_strength(-5.0).contrast_bits(), 0);
    assert_eq!(CoverageContrast::from_strength(5.0).contrast_bits(), 255);
    assert_eq!(CoverageContrast::from_strength(f32::NAN).contrast_bits(), 0);

    // Same on the weight axis — plus the headroom cap: at contrast 128 the
    // weight tops out at 127.
    let base = CoverageContrast::from_strength(0.5);
    assert_eq!(base.with_weight_strength(-5.0).weight_bits(), 0);
    assert_eq!(base.with_weight_strength(5.0).weight_bits(), 127);
    assert_eq!(base.with_weight_strength(f32::NAN).weight_bits(), 0);
    assert_eq!(base.with_weight_strength(1.0).contrast_bits(), 128);

    for contrast in 0..=u8::MAX {
        for weight in WEIGHT_SAMPLES {
            let c = CoverageContrast::from_bits(contrast, weight);
            assert_eq!(c.contrast_bits(), contrast);
            assert_eq!(c.weight_bits(), weight.min(255 - contrast), "headroom cap");
            assert_eq!(
                CoverageContrast::from_strength(c.contrast_strength())
                    .with_weight_strength(c.weight_strength()),
                c
            );
        }
    }
}

/// The `c + w <= 1` invariant, enforced at the [`CoverageContrast::from_bits`]
/// choke point. This is the condition under which the curve is monotone with
/// range `[0, 1]` (the sweeps above run at the capped boundary for every `c`),
/// which is what lets every kernel stay clamp-free per pixel.
#[test]
fn weight_caps_at_contrast_headroom() {
    assert_eq!(CoverageContrast::from_bits(200, 200).weight_bits(), 55);
    assert_eq!(CoverageContrast::from_bits(255, 255).weight_bits(), 0);
    assert_eq!(CoverageContrast::from_bits(0, 255).weight_bits(), 255);

    // The pair that motivates the cap: (1, 255) uncapped overshoots 1.0 just
    // below full coverage (the `w` bump falls away faster there than the
    // steepened ramp rises). Capped to (1, 254) it must not.
    let c = CoverageContrast::from_bits(1, 255);
    assert_eq!((c.contrast_bits(), c.weight_bits()), (1, 254));
    for i in 0..=100_000_u32 {
        let a = i as f32 / 100_000.0;
        assert!(c.apply(a) <= 1.0, "overshoot at a={a}: {}", c.apply(a));
    }
}

/// `w = 0` must leave steepening-only output bit-identical to the plain
/// single-term expression: the appended weight term is exactly `+0.0`, never a
/// rounding change.
#[test]
fn weight_zero_keeps_steepening_only_output_bit_identical() {
    let single_term = |c: f32, a: f32| a + c * a * (1.0 - a) * (2.0 * a - 1.0);

    for bits in 0..=u8::MAX {
        let c = CoverageContrast::from_bits(bits, 0);
        let c_s = c.contrast_strength();
        for i in 0..=10_000_u32 {
            let a = i as f32 / 10_000.0;
            assert_eq!(
                c.apply(a).to_bits(),
                single_term(c_s, a).to_bits(),
                "bit drift at a={a}, bits={bits}"
            );
        }
    }
}

/// With `c = 0` the curve is exactly the mask-contrast form used by Skia,
/// `a' = a + w * a * (1 - a)` — bit-for-bit, since the zeroed steepening term
/// contributes `±0.0`. The curve only ever adds coverage, and raises the 50%
/// crossing by `w/4`: it is the weight knob, asymmetric where the steepening
/// knob is symmetric.
#[test]
fn weight_term_adds_weight() {
    for weight in [1_u8, 64, 128, 191, 255] {
        let c = CoverageContrast::from_bits(0, weight);
        let w_s = c.weight_strength();

        for i in 0..=10_000_u32 {
            let a = i as f32 / 10_000.0;
            assert_eq!(
                c.apply(a).to_bits(),
                (a + w_s * a * (1.0 - a)).to_bits(),
                "mask-contrast form at a={a}, weight={weight}"
            );
            assert!(c.apply(a) >= a, "weight must only add, at a={a}");
        }

        assert!(
            (c.apply(0.5) - (0.5 + w_s * 0.25)).abs() < 1e-6,
            "midpoint shift at weight={weight}"
        );
    }
}

/// [`CoverageContrast::resolve_for_color`] scales the weight by the text
/// color's approximate relative luminance (gamma-2), leaving the contrast
/// untouched: white text keeps the stored strength, black text resolves to
/// the bit-exact `c`-only curve.
#[test]
fn resolve_for_color_scales_weight_by_luminance() {
    // Contrast 55 leaves 200 of headroom, so the requested weight of 200 is
    // stored unchanged and the scaling is observed undistorted.
    let policy = CoverageContrast::from_bits(55, 200);
    assert_eq!(policy.weight_bits(), 200, "cap must not bind in this test");

    let white = policy.resolve_for_color(Color::new([1.0, 1.0, 1.0, 1.0]));
    assert_eq!(white, policy);
    let black = policy.resolve_for_color(Color::new([0.0, 0.0, 0.0, 1.0]));
    assert_eq!(black, CoverageContrast::from_bits(55, 0));

    // Gamma-2: mid-gray has luminance 0.25.
    let gray = policy.resolve_for_color(Color::new([0.5, 0.5, 0.5, 1.0]));
    assert_eq!(gray.contrast_bits(), 55, "contrast passes through");
    assert_eq!(gray.weight_bits(), 50, "200 * 0.25 = 50");

    // Primaries order by the Rec. 709 luma coefficients.
    let lum = |r: f32, g: f32, b: f32| {
        policy
            .resolve_for_color(Color::new([r, g, b, 1.0]))
            .weight_bits()
    };
    let (red, green, blue) = (lum(1.0, 0.0, 0.0), lum(0.0, 1.0, 0.0), lum(0.0, 0.0, 1.0));
    assert!(green > red && red > blue);
    assert_eq!(green, 143, "200 * 0.7152");
    assert_eq!(red, 43, "200 * 0.2126");
    assert_eq!(blue, 14, "200 * 0.0722");

    // Alpha is ignored: opacity does not change polarity.
    assert_eq!(
        policy.resolve_for_color(Color::new([1.0, 1.0, 1.0, 0.0])),
        policy
    );

    // Out-of-range channels clamp; NaN disables rather than poisoning.
    assert_eq!(
        policy.resolve_for_color(Color::new([2.0, 5.0, 1.5, 1.0])),
        policy
    );
    assert_eq!(
        policy.resolve_for_color(Color::new([-1.0, -0.5, 0.0, 1.0])),
        CoverageContrast::from_bits(55, 0)
    );
    assert_eq!(
        policy
            .resolve_for_color(Color::new([f32::NAN, 0.5, 0.5, 1.0]))
            .weight_bits(),
        0
    );

    // A zero stored weight is returned untouched for any color, keeping the
    // disabled path branch-only.
    let contrast_only = CoverageContrast::from_bits(153, 0);
    for color in [
        Color::new([1.0, 1.0, 1.0, 1.0]),
        Color::new([0.0, 0.0, 0.0, 1.0]),
        Color::new([f32::NAN, 0.0, 0.0, 1.0]),
    ] {
        assert_eq!(contrast_only.resolve_for_color(color), contrast_only);
    }
}

/// `apply_u8` must be the `apply` curve plus the rounding the 8-bit pipeline
/// already imposes — not an independent approximation — so the u8 kernel, the
/// f32 kernel and the shader cannot drift apart.
#[test]
fn u8_path_tracks_the_f32_path() {
    for contrast in [0_u8, 1, 64, 128, 255] {
        for weight in WEIGHT_SAMPLES {
            if contrast == 0 && weight == 0 {
                continue; // the identity path is proven bit-exact above
            }
            let c = CoverageContrast::from_bits(contrast, weight);
            for byte in 0..=u8::MAX {
                // Same reciprocal form `apply_u8` uses, so this isolates the
                // rounding rather than a one-ulp conversion difference.
                let via_f32 = c.apply(f32::from(byte) * (1.0 / 255.0)) * 255.0;
                let via_u8 = f32::from(c.apply_u8(byte));
                assert!(
                    (via_f32 - via_u8).abs() <= 0.5 + 1e-3,
                    "u8 vs f32 at byte={byte}, ({contrast}, {weight}): {via_u8} vs {via_f32}"
                );
            }
        }
    }
}
