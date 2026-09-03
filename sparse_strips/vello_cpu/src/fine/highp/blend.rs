// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::peniko::{BlendMode, Mix};
use crate::util::Premultiply;
use fearless_simd_macros::simd;
use vello_common::fearless_simd::*;

#[derive(Copy, Clone)]
struct Channels<S: Simd> {
    r: f32x4<S>,
    g: f32x4<S>,
    b: f32x4<S>,
}

impl<S: Simd> Channels<S> {
    #[simd]
    fn unpremultiply(mut self, simd: S, a: f32x4<S>) -> Self {
        self.r = self.r.unpremultiply(simd, a);
        self.g = self.g.unpremultiply(simd, a);
        self.b = self.b.unpremultiply(simd, a);

        self
    }
}

#[simd]
pub(crate) fn mix<S: Simd>(
    simd: S,
    src_c: f32x16<S>,
    bg: f32x16<S>,
    blend_mode: BlendMode,
) -> f32x16<S> {
    if matches!(blend_mode.mix, Mix::Normal) {
        return src_c;
    }
    // See https://www.w3.org/TR/compositing-1/#blending

    let (bg_channels, bg_a) = split(simd, bg);
    let (src_channels, src_a) = split(simd, src_c);

    let unpremultiplied_bg = bg_channels.unpremultiply(simd, bg_a);
    let unpremultiplied_src = src_channels.unpremultiply(simd, src_a);

    let mut res_bg = unpremultiplied_bg;
    let mix_src = blend_mode.mix(simd, unpremultiplied_src, unpremultiplied_bg);

    res_bg.r = apply_alpha(simd, bg_a, src_a, unpremultiplied_src.r, mix_src.r);
    res_bg.g = apply_alpha(simd, bg_a, src_a, unpremultiplied_src.g, mix_src.g);
    res_bg.b = apply_alpha(simd, bg_a, src_a, unpremultiplied_src.b, mix_src.b);

    let mut storage = [0.0; 16];
    simd.store_four_interleaved_f32x4([res_bg.r, res_bg.g, res_bg.b, src_a], &mut storage);
    f32x16::from_slice(simd, &storage)
}

#[simd]
fn split<S: Simd>(simd: S, input: f32x16<S>) -> (Channels<S>, f32x4<S>) {
    let mut storage = [0.0; 16];
    input.store_slice(&mut storage);
    let [r, g, b, a] = simd.load_four_interleaved_f32x4(&storage);

    (Channels { r, g, b }, a)
}

#[simd]
fn apply_alpha<S: Simd>(
    simd: S,
    bg_a: f32x4<S>,
    src_a: f32x4<S>,
    unpremultiplied_src_channel: f32x4<S>,
    mix_src_channel: f32x4<S>,
) -> f32x4<S> {
    let p1 = (1.0 - bg_a) * unpremultiplied_src_channel;
    let p2 = bg_a * mix_src_channel;

    (p1 + p2).premultiply(simd, src_a)
}

trait MixExt {
    fn mix<S: Simd>(&self, simd: S, src: Channels<S>, bg: Channels<S>) -> Channels<S>;
}

impl MixExt for BlendMode {
    #[simd]
    fn mix<S: Simd>(&self, simd: S, src: Channels<S>, bg: Channels<S>) -> Channels<S> {
        match self.mix {
            Mix::Normal => src,
            Mix::Multiply => Multiply::mix(simd, src, bg),
            Mix::Screen => Screen::mix(simd, src, bg),
            Mix::Overlay => Overlay::mix(simd, src, bg),
            Mix::Darken => Darken::mix(simd, src, bg),
            Mix::Lighten => Lighten::mix(simd, src, bg),
            Mix::ColorDodge => ColorDodge::mix(simd, src, bg),
            Mix::ColorBurn => ColorBurn::mix(simd, src, bg),
            Mix::HardLight => HardLight::mix(simd, src, bg),
            Mix::SoftLight => SoftLight::mix(simd, src, bg),
            Mix::Difference => Difference::mix(simd, src, bg),
            Mix::Exclusion => Exclusion::mix(simd, src, bg),
            Mix::Luminosity => Luminosity::mix(simd, src, bg),
            Mix::Color => Color::mix(simd, src, bg),
            Mix::Hue => Hue::mix(simd, src, bg),
            Mix::Saturation => Saturation::mix(simd, src, bg),
        }
    }
}

impl Multiply {
    #[simd]
    fn single<S: Simd>(simd: S, src: f32x4<S>, bg: f32x4<S>) -> f32x4<S> {
        src * bg
    }
}

impl Screen {
    #[simd]
    fn single<S: Simd>(simd: S, src: f32x4<S>, bg: f32x4<S>) -> f32x4<S> {
        bg + src - src * bg
    }
}

impl HardLight {
    #[simd]
    fn single<S: Simd>(simd: S, src: f32x4<S>, bg: f32x4<S>) -> f32x4<S> {
        let two = f32x4::splat(simd, 2.0);

        let mask = simd.simd_le_f32x4(src, f32x4::splat(simd, 0.5));
        let opt1 = Multiply::single(simd, bg, src * two);
        let opt2 = Screen::single(simd, bg, two * src - 1.0);

        simd.select_f32x4(mask, opt1, opt2)
    }
}

macro_rules! separable_mix {
    ($name:ident, $calc:expr) => {
        pub(crate) struct $name;

        impl $name {
            #[simd]
            fn mix<S: Simd>(simd: S, mut src: Channels<S>, bg: Channels<S>) -> Channels<S> {
                src.r = ($calc)(simd, src.r, bg.r);
                src.g = ($calc)(simd, src.g, bg.g);
                src.b = ($calc)(simd, src.b, bg.b);

                src
            }
        }
    };
}

separable_mix!(Multiply, |simd: S, cs: f32x4<S>, cb: f32x4<S>| {
    Multiply::single(simd, cs, cb)
});
separable_mix!(Screen, |simd: S, cs: f32x4<S>, cb: f32x4<S>| {
    Screen::single(simd, cs, cb)
});
separable_mix!(Overlay, |simd: S, cs: f32x4<S>, cb: f32x4<S>| {
    HardLight::single(simd, cb, cs)
});
separable_mix!(Darken, |_: S, cs: f32x4<S>, cb: f32x4<S>| cs.min(cb));
separable_mix!(Lighten, |_: S, cs: f32x4<S>, cb: f32x4<S>| cs.max(cb));
separable_mix!(Difference, |simd: S, cs: f32x4<S>, cb: f32x4<S>| {
    simd.select_f32x4(simd.simd_le_f32x4(cs, cb), cb - cs, cs - cb)
});
separable_mix!(HardLight, |simd: S, cs: f32x4<S>, cb: f32x4<S>| {
    HardLight::single(simd, cs, cb)
});
separable_mix!(Exclusion, |_: S, cs: f32x4<S>, cb: f32x4<S>| {
    (cs + cb) - 2.0 * (cs * cb)
});
separable_mix!(SoftLight, |simd: S, cs: f32x4<S>, cb: f32x4<S>| {
    let mask_1 = simd.simd_le_f32x4(cb, f32x4::splat(simd, 0.25));

    let d = simd.select_f32x4(mask_1, ((16.0 * cb - 12.0) * cb + 4.0) * cb, cb.sqrt());

    let mask_2 = simd.simd_le_f32x4(cs, f32x4::splat(simd, 0.5));

    simd.select_f32x4(
        mask_2,
        cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb),
        cb + (2.0 * cs - 1.0) * (d - cb),
    )
});
separable_mix!(ColorDodge, |simd: S, cs: f32x4<S>, cb: f32x4<S>| {
    let mask_1 = simd.simd_eq_f32x4(cb, f32x4::splat(simd, 0.0));
    let mask_2 = simd.simd_eq_f32x4(cs, f32x4::splat(simd, 1.0));

    simd.select_f32x4(
        // if cb == 0
        mask_1,
        f32x4::splat(simd, 0.0),
        // else if cs == 1
        simd.select_f32x4(
            mask_2,
            f32x4::splat(simd, 1.0),
            // else
            f32x4::splat(simd, 1.0).min(cb / (1.0 - cs)),
        ),
    )
});
separable_mix!(ColorBurn, |simd: S, cs: f32x4<S>, cb: f32x4<S>| {
    let mask_1 = simd.simd_eq_f32x4(cb, f32x4::splat(simd, 1.0));
    let mask_2 = simd.simd_eq_f32x4(cs, f32x4::splat(simd, 0.0));

    simd.select_f32x4(
        // if cb == 1
        mask_1,
        f32x4::splat(simd, 1.0),
        // else if cs == 0
        simd.select_f32x4(
            mask_2,
            f32x4::splat(simd, 0.0),
            // else
            1.0 - f32x4::splat(simd, 1.0).min((1.0 - cb) / cs),
        ),
    )
});

macro_rules! non_separable_mix {
    ($name:ident, $calc:expr) => {
        pub(crate) struct $name;

        impl $name {
            #[simd]
            fn mix<S: Simd>(simd: S, mut src: Channels<S>, mut bg: Channels<S>) -> Channels<S> {
                ($calc)(simd, &mut src, &mut bg)
            }
        }
    };
}

non_separable_mix!(Hue, |simd: S,
                         cs: &mut Channels<S>,
                         cb: &mut Channels<S>| {
    set_sat(
        simd,
        &mut cs.r,
        &mut cs.g,
        &mut cs.b,
        sat(simd, cb.r, cb.g, cb.b),
    );
    set_lum(
        simd,
        &mut cs.r,
        &mut cs.g,
        &mut cs.b,
        lum(simd, cb.r, cb.g, cb.b),
    );

    *cs
});

non_separable_mix!(
    Saturation,
    |simd: S, cs: &mut Channels<S>, cb: &mut Channels<S>| {
        let lum = lum(simd, cb.r, cb.g, cb.b);
        set_sat(
            simd,
            &mut cb.r,
            &mut cb.g,
            &mut cb.b,
            sat(simd, cs.r, cs.g, cs.b),
        );
        set_lum(simd, &mut cb.r, &mut cb.g, &mut cb.b, lum);

        *cb
    }
);

non_separable_mix!(Color, |simd: S,
                           cs: &mut Channels<S>,
                           cb: &mut Channels<S>| {
    set_lum(
        simd,
        &mut cs.r,
        &mut cs.g,
        &mut cs.b,
        lum(simd, cb.r, cb.g, cb.b),
    );

    *cs
});
non_separable_mix!(
    Luminosity,
    |simd: S, cs: &mut Channels<S>, cb: &mut Channels<S>| {
        set_lum(
            simd,
            &mut cb.r,
            &mut cb.g,
            &mut cb.b,
            lum(simd, cs.r, cs.g, cs.b),
        );

        *cb
    }
);

#[simd]
fn lum<S: Simd>(simd: S, r: f32x4<S>, g: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
    0.3 * r + 0.59 * g + 0.11 * b
}

#[simd]
fn sat<S: Simd>(simd: S, r: f32x4<S>, g: f32x4<S>, b: f32x4<S>) -> f32x4<S> {
    r.max(g).max(b) - r.min(g).min(b)
}

#[simd]
fn clip_color<S: Simd>(simd: S, r: &mut f32x4<S>, g: &mut f32x4<S>, b: &mut f32x4<S>) {
    let l = lum(simd, *r, *g, *b);
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

#[simd]
fn set_lum<S: Simd>(simd: S, r: &mut f32x4<S>, g: &mut f32x4<S>, b: &mut f32x4<S>, l: f32x4<S>) {
    let d = l - lum(simd, *r, *g, *b);
    *r += d;
    *g += d;
    *b += d;

    clip_color(simd, r, g, b);
}

// Adapted from tiny-skia
#[simd]
fn set_sat<S: Simd>(simd: S, r: &mut f32x4<S>, g: &mut f32x4<S>, b: &mut f32x4<S>, s: f32x4<S>) {
    let mn = r.min(g.min(*b));
    let mx = r.max(g.max(*b));
    let sat = mx - mn;

    // Map min channel to 0, max channel to s, and scale the middle proportionally.
    *r = scale_sat_channel(simd, *r, mn, sat, s);
    *g = scale_sat_channel(simd, *g, mn, sat, s);
    *b = scale_sat_channel(simd, *b, mn, sat, s);
}

#[simd]
fn scale_sat_channel<S: Simd>(
    simd: S,
    c: f32x4<S>,
    mn: f32x4<S>,
    sat: f32x4<S>,
    s: f32x4<S>,
) -> f32x4<S> {
    simd.select_f32x4(
        simd.simd_eq_f32x4(sat, f32x4::splat(simd, 0.0)),
        f32x4::splat(simd, 0.0),
        (c - mn) * s / sat,
    )
}
