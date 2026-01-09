// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko::{BlendMode, Mix};
use crate::util::Premultiply;
use vello_common::fearless_simd::*;

#[derive(Copy, Clone)]
struct Channels<S: Simd> {
    r: f32x4<S>,
    g: f32x4<S>,
    b: f32x4<S>,
}

impl<S: Simd> Channels<S> {
    #[inline(always)]
    fn unpremultiply(mut self, a: f32x4<S>) -> Self {
        self.r = self.r.unpremultiply(a);
        self.g = self.g.unpremultiply(a);
        self.b = self.b.unpremultiply(a);

        self
    }
}

// TODO: blending is still extremely slow, investigate whether there is something obvious we are
// missing that other renderers do.
pub(crate) fn mix<S: Simd>(src_c: f32x16<S>, bg: f32x16<S>, blend_mode: BlendMode) -> f32x16<S> {
    if matches!(blend_mode.mix, Mix::Normal) {
        return src_c;
    }
    // See https://www.w3.org/TR/compositing-1/#blending
    let simd = src_c.simd;

    let split = |input: f32x16<S>| {
        let mut storage = [0.0; 16];
        simd.store_interleaved_128_f32x16(input, &mut storage);
        let input_v = f32x16::from_slice(simd, &storage);

        let p1 = simd.split_f32x16(input_v);
        let (r, g) = simd.split_f32x8(p1.0);
        let (b, a) = simd.split_f32x8(p1.1);

        (Channels { r, g, b }, a)
    };

    let (bg_channels, bg_a) = split(bg);
    let (src_channels, src_a) = split(src_c);

    let unpremultiplied_bg = bg_channels.unpremultiply(bg_a);
    let unpremultiplied_src = src_channels.unpremultiply(src_a);

    let mut res_bg = unpremultiplied_bg;
    let mix_src = blend_mode.mix(unpremultiplied_src, unpremultiplied_bg);

    let apply_alpha = |unpremultiplied_src_channel: f32x4<S>,
                       mix_src_channel: f32x4<S>,
                       dest_channel: &mut f32x4<S>| {
        let p1 = (1.0 - bg_a) * unpremultiplied_src_channel;
        let p2 = bg_a * mix_src_channel;

        *dest_channel = (p1 + p2).premultiply(src_a);
    };

    apply_alpha(unpremultiplied_src.r, mix_src.r, &mut res_bg.r);
    apply_alpha(unpremultiplied_src.g, mix_src.g, &mut res_bg.g);
    apply_alpha(unpremultiplied_src.b, mix_src.b, &mut res_bg.b);

    let combined = simd.combine_f32x8(
        simd.combine_f32x4(res_bg.r, res_bg.g),
        simd.combine_f32x4(res_bg.b, src_a),
    );

    let mut storage = [0.0; 16];
    simd.store_interleaved_128_f32x16(combined, &mut storage);
    f32x16::from_slice(simd, &storage)
}

trait MixExt {
    fn mix<S: Simd>(&self, src: Channels<S>, bg: Channels<S>) -> Channels<S>;
}

impl MixExt for BlendMode {
    fn mix<S: Simd>(&self, src: Channels<S>, bg: Channels<S>) -> Channels<S> {
        match self.mix {
            Mix::Normal => src,
            Mix::Multiply => Multiply::mix(src, bg),
            Mix::Screen => Screen::mix(src, bg),
            Mix::Overlay => Overlay::mix(src, bg),
            Mix::Darken => Darken::mix(src, bg),
            Mix::Lighten => Lighten::mix(src, bg),
            Mix::ColorDodge => ColorDodge::mix(src, bg),
            Mix::ColorBurn => ColorBurn::mix(src, bg),
            Mix::HardLight => HardLight::mix(src, bg),
            Mix::SoftLight => SoftLight::mix(src, bg),
            Mix::Difference => Difference::mix(src, bg),
            Mix::Exclusion => Exclusion::mix(src, bg),
            Mix::Luminosity => Luminosity::mix(src, bg),
            Mix::Color => Color::mix(src, bg),
            Mix::Hue => Hue::mix(src, bg),
            Mix::Saturation => Saturation::mix(src, bg),
        }
    }
}

impl Multiply {
    #[inline(always)]
    fn single<S: Simd>(src: f32x4<S>, bg: f32x4<S>) -> f32x4<S> {
        src * bg
    }
}

impl Screen {
    #[inline(always)]
    fn single<S: Simd>(src: f32x4<S>, bg: f32x4<S>) -> f32x4<S> {
        bg + src - src * bg
    }
}

impl HardLight {
    fn single<S: Simd>(src: f32x4<S>, bg: f32x4<S>) -> f32x4<S> {
        let two = f32x4::splat(src.simd, 2.0);

        let mask = src.simd.simd_le_f32x4(src, f32x4::splat(src.simd, 0.5));
        let opt1 = Multiply::single(bg, src * two);
        let opt2 = Screen::single(bg, two * src - 1.0);

        src.simd.select_f32x4(mask, opt1, opt2)
    }
}

macro_rules! separable_mix {
    ($name:ident, $calc:expr) => {
        pub(crate) struct $name;

        impl $name {
            #[inline(always)]
            fn mix<S: Simd>(mut src: Channels<S>, bg: Channels<S>) -> Channels<S> {
                src.r = $calc(src.r, bg.r);
                src.g = $calc(src.g, bg.g);
                src.b = $calc(src.b, bg.b);

                src
            }
        }
    };
}

separable_mix!(Multiply, |cs: f32x4<S>, cb: f32x4<S>| Multiply::single(
    cs, cb
));
separable_mix!(Screen, |cs: f32x4<S>, cb: f32x4<S>| Screen::single(cs, cb));
separable_mix!(Overlay, |cs: f32x4<S>, cb: f32x4<S>| HardLight::single(
    cb, cs
));
separable_mix!(Darken, |cs: f32x4<S>, cb: f32x4<S>| cs.min(cb));
separable_mix!(Lighten, |cs: f32x4<S>, cb: f32x4<S>| cs.max(cb));
separable_mix!(Difference, |cs: f32x4<S>, cb: f32x4<S>| {
    cs.simd
        .select_f32x4(cs.simd.simd_le_f32x4(cs, cb), cb - cs, cs - cb)
});
separable_mix!(HardLight, |cs: f32x4<S>, cb: f32x4<S>| HardLight::single(
    cs, cb
));
separable_mix!(Exclusion, |cs: f32x4<S>, cb: f32x4<S>| {
    (cs + cb) - 2.0 * (cs * cb)
});
separable_mix!(SoftLight, |cs: f32x4<S>, cb: f32x4<S>| {
    let mask_1 = cs.simd.simd_le_f32x4(cb, f32x4::splat(cs.simd, 0.25));

    let d = cs
        .simd
        .select_f32x4(mask_1, ((16.0 * cb - 12.0) * cb + 4.0) * cb, cb.sqrt());

    let mask_2 = cs.simd.simd_le_f32x4(cs, f32x4::splat(cs.simd, 0.5));

    cs.simd.select_f32x4(
        mask_2,
        cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb),
        cb + (2.0 * cs - 1.0) * (d - cb),
    )
});
separable_mix!(ColorDodge, |cs: f32x4<S>, cb: f32x4<S>| {
    let mask_1 = cb.simd.simd_eq_f32x4(cb, f32x4::splat(cb.simd, 0.0));
    let mask_2 = cs.simd.simd_eq_f32x4(cs, f32x4::splat(cs.simd, 1.0));

    cs.simd.select_f32x4(
        // if cb == 0
        mask_1,
        f32x4::splat(cs.simd, 0.0),
        // else if cs == 1
        cs.simd.select_f32x4(
            mask_2,
            f32x4::splat(cs.simd, 1.0),
            // else
            f32x4::splat(cs.simd, 1.0).min(cb / (1.0 - cs)),
        ),
    )
});
separable_mix!(ColorBurn, |cs: f32x4<S>, cb: f32x4<S>| {
    let mask_1 = cb.simd.simd_eq_f32x4(cb, f32x4::splat(cb.simd, 1.0));
    let mask_2 = cs.simd.simd_eq_f32x4(cs, f32x4::splat(cs.simd, 0.0));

    cs.simd.select_f32x4(
        // if cb == 1
        mask_1,
        f32x4::splat(cs.simd, 1.0),
        // else if cs == 0
        cs.simd.select_f32x4(
            mask_2,
            f32x4::splat(cs.simd, 0.0),
            // else
            1.0 - f32x4::splat(cs.simd, 1.0).min((1.0 - cb) / cs),
        ),
    )
});

macro_rules! non_separable_mix {
    ($name:ident, $calc:expr) => {
        pub(crate) struct $name;

        impl $name {
            #[inline(always)]
            fn mix<S: Simd>(mut src: Channels<S>, mut bg: Channels<S>) -> Channels<S> {
                $calc(&mut src, &mut bg)
            }
        }
    };
}

non_separable_mix!(Hue, |cs: &mut Channels<S>, cb: &mut Channels<S>| {
    set_sat(&mut cs.r, &mut cs.g, &mut cs.b, sat(cb.r, cb.g, cb.b));
    set_lum(&mut cs.r, &mut cs.g, &mut cs.b, lum(cb.r, cb.g, cb.b));

    *cs
});

non_separable_mix!(Saturation, |cs: &mut Channels<S>, cb: &mut Channels<S>| {
    let lum = lum(cb.r, cb.g, cb.b);
    set_sat(&mut cb.r, &mut cb.g, &mut cb.b, sat(cs.r, cs.g, cs.b));
    set_lum(&mut cb.r, &mut cb.g, &mut cb.b, lum);

    *cb
});

non_separable_mix!(Color, |cs: &mut Channels<S>, cb: &mut Channels<S>| {
    set_lum(&mut cs.r, &mut cs.g, &mut cs.b, lum(cb.r, cb.g, cb.b));

    *cs
});
non_separable_mix!(Luminosity, |cs: &mut Channels<S>, cb: &mut Channels<S>| {
    set_lum(&mut cb.r, &mut cb.g, &mut cb.b, lum(cs.r, cs.g, cs.b));

    *cb
});

fn lum<S: Simd>(r: f32x4<S>, g: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
    0.3 * r + 0.59 * g + 0.11 * b
}

fn sat<S: Simd>(r: f32x4<S>, g: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
    r.max(g).max(b) - r.min(g).min(b)
}

fn clip_color<S: Simd>(r: &mut f32x4<S>, g: &mut f32x4<S>, b: &mut f32x4<S>) {
    let simd = r.simd;

    let l = lum(*r, *g, *b);
    let n = r.min(g.min(*b));
    let x = r.max(g.max(*b));

    for c in [r, g, b] {
        *c = simd.select_f32x4(
            simd.simd_lt_f32x4(n, f32x4::splat(simd, 0.0)),
            l + (((*c - l) * l) / (l - n)),
            *c,
        );

        *c = simd.select_f32x4(
            simd.simd_gt_f32x4(x, f32x4::splat(simd, 1.0)),
            l + (((*c - l) * (1.0 - l)) / (x - l)),
            *c,
        );
    }
}

fn set_lum<S: Simd>(r: &mut f32x4<S>, g: &mut f32x4<S>, b: &mut f32x4<S>, l: f32x4<S>) {
    let d = l - lum(*r, *g, *b);
    *r += d;
    *g += d;
    *b += d;

    clip_color(r, g, b);
}

// Adapted from tiny-skia
fn set_sat<S: Simd>(r: &mut f32x4<S>, g: &mut f32x4<S>, b: &mut f32x4<S>, s: f32x4<S>) {
    let simd = r.simd;
    let zero = f32x4::splat(simd, 0.0);
    let mn = r.min(g.min(*b));
    let mx = r.max(g.max(*b));
    let sat = mx - mn;

    // Map min channel to 0, max channel to s, and scale the middle proportionally.
    let scale = |c| simd.select_f32x4(simd.simd_eq_f32x4(sat, zero), zero, (c - mn) * s / sat);

    *r = scale(*r);
    *g = scale(*g);
    *b = scale(*b);
}
