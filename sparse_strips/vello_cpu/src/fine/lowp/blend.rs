// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::{Splat4thExt, highp, u8_to_f32};
use crate::peniko::{BlendMode, Mix};
use vello_common::fearless_simd::*;
use vello_common::util::{Div255Ext, f32_to_u8, normalized_mul_u8x32};

// TODO: Make sure this vectorizes properly (also the f32 pipeline) by inlining if needed.
pub(crate) fn mix<S: Simd>(src_c: u8x32<S>, bg_c: u8x32<S>, blend_mode: BlendMode) -> u8x32<S> {
    if let Some(res) = try_u8_mix(blend_mode, src_c, bg_c) {
        return res;
    }

    // Fallback for blend modes that aren't supported in u8.

    let to_f32 = |val: u8x32<S>| {
        let (a, b) = src_c.simd.split_u8x32(val);
        let mut a = u8_to_f32(a);
        let mut b = u8_to_f32(b);
        a *= f32x16::splat(src_c.simd, 1.0 / 255.0);
        b *= f32x16::splat(src_c.simd, 1.0 / 255.0);
        (a, b)
    };

    let to_u8 = |val1: f32x16<S>, val2: f32x16<S>| {
        let val1 =
            f32_to_u8(f32x16::splat(val1.simd, 255.0).mul_add(val1, f32x16::splat(val1.simd, 0.5)));
        let val2 =
            f32_to_u8(f32x16::splat(val2.simd, 255.0).mul_add(val2, f32x16::splat(val2.simd, 0.5)));

        val1.simd.combine_u8x16(val1, val2)
    };

    let (mut src_1, mut src_2) = to_f32(src_c);
    let (bg_1, bg_2) = to_f32(bg_c);

    src_1 = highp::blend::mix(src_1, bg_1, blend_mode);
    src_2 = highp::blend::mix(src_2, bg_2, blend_mode);

    to_u8(src_1, src_2)
}

fn try_u8_mix<S: Simd>(blend_mode: BlendMode, src_c: u8x32<S>, bg_c: u8x32<S>) -> Option<u8x32<S>> {
    // We implement the u8 fast path for blend modes that
    // 1) are separable.
    // 2) don't have too many divisions, since integer normalization is
    // relatively expensive.
    // In the future, it's possible to do further experimentation to see whether
    // some more blend modes are worth doing in integer space.
    Some(match blend_mode.mix {
        Mix::Normal => src_c,
        Mix::Multiply => Multiply::mix(src_c, bg_c),
        Mix::Screen => Screen::mix(src_c, bg_c),
        Mix::Overlay => Overlay::mix(src_c, bg_c),
        Mix::Darken => Darken::mix(src_c, bg_c),
        Mix::Lighten => Lighten::mix(src_c, bg_c),
        Mix::HardLight => HardLight::mix(src_c, bg_c),
        Mix::Difference => Difference::mix(src_c, bg_c),
        Mix::Exclusion => Exclusion::mix(src_c, bg_c),
        Mix::ColorDodge
        | Mix::ColorBurn
        | Mix::SoftLight
        | Mix::Luminosity
        | Mix::Color
        | Mix::Hue
        | Mix::Saturation => return None,
    })
}

#[inline(always)]
fn narrow_saturating_u16x32<S: Simd>(simd: S, val: u16x32<S>) -> u8x32<S> {
    // In case we had an overflow, make sure to clamp back to `u8::MAX`.
    simd.narrow_u16x32(val.min(u16x32::splat(simd, 255)))
}

macro_rules! u8_mix {
    ($name:ident, $calc:expr) => {
        struct $name;

        impl $name {
            #[inline(always)]
            fn mix<S: Simd>(src_c: u8x32<S>, bg_c: u8x32<S>) -> u8x32<S> {
                let simd = src_c.simd;
                let res = $calc(src_c, bg_c);

                with_src_alpha(simd, res, src_c)
            }
        }
    };
}

// Formula for blending is (see https://www.w3.org/TR/compositing-1/#generalformula):
//   Cs' = (1 - Ab) * Cs + Ab * B(Cb, Cs)
// Since vello_cpu expects premultiplied colors, we need to return:
//   M = As * Cs'
//     = As * ((1 - Ab) * Cs + Ab * B(Cb, Cs))
//     = As * (1 - Ab) * Cs + As * Ab * B(Cb, Cs)
//     = S * (1 - Ab) + As * Ab * B(Cb, Cs)
// where S = As * Cs and D = Ab * Cb (so just the premultiplied color).

// Multiply:
//   B(Cb, Cs) = Cb * Cs
//   M = S * (1 - Ab) + As * Ab * Cb * Cs
//     = S * (1 - Ab) + S * D
u8_mix!(Multiply, |src_c: u8x32<S>, bg_c: u8x32<S>| {
    let simd = src_c.simd;
    let one_minus_bg_a = 255 - bg_c.splat_4th();
    let p1 = normalized_mul_u8x32(src_c, one_minus_bg_a);
    let p2 = normalized_mul_u8x32(src_c, bg_c);

    narrow_saturating_u16x32(simd, p1 + p2)
});

// Screen:
//   B(Cb, Cs) = Cb + Cs - Cb * Cs
//   M = S * (1 - Ab) + As * D + S * Ab - S * D
//     = S + As * D - S * D
u8_mix!(Screen, |src_c: u8x32<S>, bg_c: u8x32<S>| {
    let simd = src_c.simd;
    let p1 = normalized_mul_u8x32(src_c.splat_4th(), bg_c);
    let p2 = normalized_mul_u8x32(src_c, bg_c);
    let res = simd.widen_u8x32(src_c) + p1 - p2;

    narrow_saturating_u16x32(simd, res)
});

// Overlay is hard-light with source and backdrop swapped.
u8_mix!(Overlay, |src_c: u8x32<S>, bg_c: u8x32<S>| {
    hard_light_inner(src_c, bg_c, bg_c)
});

// Darken:
//   B(Cb, Cs) = min(Cb, Cs)
//   M = S * (1 - Ab) + min(S * Ab, D * As)
u8_mix!(Darken, |src_c: u8x32<S>, bg_c: u8x32<S>| {
    let simd = src_c.simd;
    let src_a = src_c.splat_4th();
    let bg_a = bg_c.splat_4th();
    let p1 = normalized_mul_u8x32(src_c, 255 - bg_a);
    let p2 = normalized_mul_u8x32(src_c, bg_a).min(normalized_mul_u8x32(bg_c, src_a));

    narrow_saturating_u16x32(simd, p1 + p2)
});

// Lighten:
//   B(Cb, Cs) = max(Cb, Cs)
//   M = S * (1 - Ab) + max(S * Ab, D * As)
u8_mix!(Lighten, |src_c: u8x32<S>, bg_c: u8x32<S>| {
    let simd = src_c.simd;
    let src_a = src_c.splat_4th();
    let bg_a = bg_c.splat_4th();
    let p1 = normalized_mul_u8x32(src_c, 255 - bg_a);
    let p2 = normalized_mul_u8x32(src_c, bg_a).max(normalized_mul_u8x32(bg_c, src_a));

    narrow_saturating_u16x32(simd, p1 + p2)
});

// Hard-light:
//   if Cs <= 0.5: B(Cb, Cs) = 2 * Cb * Cs
//   otherwise:    B(Cb, Cs) = 1 - 2 * (1 - Cb) * (1 - Cs)
u8_mix!(HardLight, |src_c: u8x32<S>, bg_c: u8x32<S>| {
    hard_light_inner(src_c, bg_c, src_c)
});

// Difference:
//   B(Cb, Cs) = abs(Cb - Cs)
//   M = S * (1 - Ab) + abs(S * Ab - D * As)
u8_mix!(Difference, |src_c: u8x32<S>, bg_c: u8x32<S>| {
    let simd = src_c.simd;
    let src_a = src_c.splat_4th();
    let bg_a = bg_c.splat_4th();
    let p1 = normalized_mul_u8x32(src_c, 255 - bg_a);
    let p2 = normalized_mul_u8x32(src_c, bg_a);
    let p3 = normalized_mul_u8x32(bg_c, src_a);
    let diff = p2.max(p3) - p2.min(p3);

    narrow_saturating_u16x32(simd, p1 + diff)
});

// Exclusion:
//   B(Cb, Cs) = Cb + Cs - 2 * Cb * Cs
//   M = S * (1 - Ab) + As * D + S * Ab - 2 * S * D
//     = S + As * D - 2 * S * D
u8_mix!(Exclusion, |src_c: u8x32<S>, bg_c: u8x32<S>| {
    let simd = src_c.simd;
    let p1 = normalized_mul_u8x32(src_c.splat_4th(), bg_c);
    let p2 = normalized_mul_u8x32(src_c, bg_c);
    let res = simd.widen_u8x32(src_c) + p1;
    let sub = p2 + p2;
    let res = simd.select_u16x32(res.simd_ge(sub), res - sub, u16x32::splat(simd, 0));

    narrow_saturating_u16x32(simd, res)
});

#[inline(always)]
fn hard_light_inner<S: Simd>(src_c: u8x32<S>, bg_c: u8x32<S>, condition: u8x32<S>) -> u8x32<S> {
    let simd = src_c.simd;
    let src = simd.widen_u8x32(src_c);
    let bg = simd.widen_u8x32(bg_c);
    let src_a = simd.widen_u8x32(src_c.splat_4th());
    let bg_a = simd.widen_u8x32(bg_c.splat_4th());
    let condition_a = simd.widen_u8x32(condition.splat_4th());
    let condition = simd.widen_u8x32(condition);

    let base = src * (255 - bg_a);
    // Multiply branch: As * Ab * 2 * Cb * Cs = 2 * S * D.
    let multiply = 2 * src * bg;
    // Screen branch: As * Ab * (1 - 2 * (1 - Cb) * (1 - Cs))
    //              = As * Ab - 2 * (As - S) * (Ab - D).
    let screen = src_a * bg_a - 2 * (src_a - src) * (bg_a - bg);
    let blended = simd.select_u16x32(
        // The spec condition is `Cs <= 0.5` but on unpremultiplied color.
        // Since `Cs = S / As`, we avoid division by multiplying both sides
        // by alpha: `Cs <= 0.5` => `S <= 0.5 * As` => `2 * S <= As`.
        (condition + condition).simd_le(condition_a),
        multiply,
        screen,
    );
    let res = (base + blended).div_255();

    narrow_saturating_u16x32(simd, res)
}

#[inline(always)]
fn with_src_alpha<S: Simd>(simd: S, rgb: u8x32<S>, src_c: u8x32<S>) -> u8x32<S> {
    let alpha_mask = u32x8::splat(simd, u32::from_ne_bytes([0, 0, 0, 255])).to_bytes();
    // It can happen that we end up with an R/G/B larger than the alpha value due to
    // arithmetic errors. We need to clamp to the alpha to ensure the color is still a valid
    // premultiplied color.
    let rgb = rgb.min(src_c.splat_4th());

    (rgb & !alpha_mask) | (src_c & alpha_mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peniko::Compose;
    use vello_common::fearless_simd::Fallback;

    fn lowp_first_pixel_for_mix(blend: Mix, src: [u8; 4], bg: [u8; 4]) -> [u8; 4] {
        let simd = Fallback::new();
        let src = u8x32::from_slice(simd, &src.repeat(8));
        let bg = u8x32::from_slice(simd, &bg.repeat(8));
        let blend_mode = BlendMode::new(blend, Compose::SrcOver);
        let res = mix(src, bg, blend_mode);
        let mut out = [0; 32];
        res.store_slice(&mut out);
        out[..4].try_into().unwrap()
    }

    fn highp_first_pixel_for_mix(mix: Mix, src: [u8; 4], bg: [u8; 4]) -> [u8; 4] {
        let simd = Fallback::new();
        let to_f32 = |pixel: [u8; 4]| {
            let pixel = pixel.map(|component| component as f32 * (1.0 / 255.0));
            f32x16::from_slice(simd, &pixel.repeat(4))
        };
        let src = to_f32(src);
        let bg = to_f32(bg);
        let blend_mode = BlendMode::new(mix, Compose::SrcOver);

        let res = highp::blend::mix(src, bg, blend_mode);
        let res = f32_to_u8(f32x16::splat(simd, 255.0).mul_add(res, f32x16::splat(simd, 0.5)));
        let mut out = [0; 16];
        res.store_slice(&mut out);
        out[..4].try_into().unwrap()
    }

    fn assert_lowp_matches_highp(mix: Mix, src: [u8; 4], bg: [u8; 4]) {
        const MAX_DELTA: u8 = 2;

        let lowp = lowp_first_pixel_for_mix(mix, src, bg);
        let highp = highp_first_pixel_for_mix(mix, src, bg);

        let lowp_alpha = lowp[3];
        assert_eq!(lowp_alpha, src[3]);
        for (component, &component_value) in lowp[..3].iter().enumerate() {
            assert!(
                component_value <= lowp_alpha,
                "{mix:?} component {component} exceeded alpha: lowp={lowp:?}, src={src:?}, bg={bg:?}"
            );
        }

        for (component, (&lowp, &highp)) in lowp.iter().zip(highp.iter()).enumerate() {
            let delta = lowp.abs_diff(highp);
            assert!(
                delta <= MAX_DELTA,
                "{mix:?} component {component} differed by {delta}: lowp={lowp}, highp={highp}, src={src:?}, bg={bg:?}"
            );
        }
    }

    #[test]
    fn multiply_does_not_wrap() {
        assert_lowp_matches_highp(Mix::Multiply, [1, 1, 1, 1], [1, 1, 1, 129]);
    }

    #[test]
    fn screen_does_not_wrap() {
        assert_lowp_matches_highp(Mix::Screen, [255, 255, 255, 255], [255, 255, 255, 255]);
    }

    #[test]
    fn darken_does_not_wrap() {
        assert_lowp_matches_highp(Mix::Darken, [20, 20, 20, 20], [92, 92, 92, 92]);
    }

    #[test]
    fn lighten_does_not_wrap() {
        assert_lowp_matches_highp(Mix::Lighten, [1, 1, 1, 2], [129, 129, 129, 131]);
    }

    #[test]
    fn difference_does_not_wrap() {
        assert_lowp_matches_highp(Mix::Difference, [1, 1, 1, 2], [129, 129, 129, 193]);
    }

    #[test]
    fn exclusion_does_not_wrap() {
        assert_lowp_matches_highp(Mix::Exclusion, [128, 128, 128, 255], [128, 128, 128, 255]);
    }

    #[test]
    fn hard_light_does_not_wrap() {
        assert_lowp_matches_highp(Mix::HardLight, [1, 1, 1, 2], [2, 2, 2, 2]);
    }

    #[test]
    fn overlay_does_not_wrap() {
        assert_lowp_matches_highp(Mix::Overlay, [0, 0, 0, 1], [1, 1, 1, 1]);
    }
}
