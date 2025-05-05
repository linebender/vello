// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for blending. See <https://www.w3.org/TR/compositing-1/#introduction> for
//! an introduction as well as the corresponding formulas.

use crate::fine::{COLOR_COMPONENTS, FineType};
use vello_common::peniko::{BlendMode, Compose, Mix};

pub(crate) mod fill {
    use crate::fine::blend::{BlendModeExt, mix};
    use crate::fine::{COLOR_COMPONENTS, FineType, TILE_HEIGHT_COMPONENTS};
    use vello_common::peniko::BlendMode;

    pub(crate) fn blend<F: FineType, T: Iterator<Item = [F; COLOR_COMPONENTS]>>(
        target: &mut [F],
        mut color_iter: T,
        blend_mode: BlendMode,
    ) {
        for strip in target.chunks_exact_mut(TILE_HEIGHT_COMPONENTS) {
            for bg_c in strip.chunks_exact_mut(COLOR_COMPONENTS) {
                let mixed_src_color = mix(color_iter.next().unwrap(), bg_c, blend_mode);

                blend_mode.compose(&mixed_src_color, bg_c, F::from_u8(255));
            }
        }
    }
}

pub(crate) mod strip {
    use crate::fine::blend::{BlendModeExt, mix};
    use crate::fine::{COLOR_COMPONENTS, FineType, TILE_HEIGHT_COMPONENTS};
    use vello_common::peniko::BlendMode;
    use vello_common::tile::Tile;

    pub(crate) fn blend<
        F: FineType,
        T: Iterator<Item = [F; COLOR_COMPONENTS]>,
        A: Iterator<Item = [u8; Tile::HEIGHT as usize]>,
    >(
        target: &mut [F],
        mut color_iter: T,
        blend_mode: BlendMode,
        mut alphas: A,
    ) {
        for bg_col in target.chunks_exact_mut(TILE_HEIGHT_COMPONENTS) {
            let masks = alphas.next().unwrap();

            for (bg_pix, mask) in bg_col.chunks_exact_mut(Tile::HEIGHT as usize).zip(masks) {
                let mixed_src_color = mix(color_iter.next().unwrap(), bg_pix, blend_mode);

                blend_mode.compose(&mixed_src_color, bg_pix, F::from_u8(mask));
            }
        }
    }
}

fn mix<F: FineType>(mut src_c: [F; 4], bg_c: &[F], blend_mode: BlendMode) -> [F; 4] {
    // See https://www.w3.org/TR/compositing-1/#blending

    let mut mix_bg = [bg_c[0], bg_c[1], bg_c[2], bg_c[3]];
    let bg_alpha = mix_bg[3];

    // For blending, we need to first unpremultiply everything.
    unpremultiply(&mut mix_bg);
    unpremultiply(&mut src_c);

    let mut mixed = src_c;

    // Mix the source and background color. This will then be our
    // new source color.
    blend_mode.mix(&mut mixed[0..3], &mix_bg[0..3]);

    // Account for alpha.
    for i in 0..3 {
        src_c[i] = bg_alpha.inv().norm_mul_add(src_c[i], bg_alpha, mixed[i]);
    }

    // Premultiply again.
    premultiply(&mut src_c);

    src_c
}

pub(crate) trait BlendModeExt {
    fn mix<F: FineType>(&self, src: &mut [F], bg: &[F]);
    fn compose<F: FineType>(&self, src_c: &[F; 4], bg_c: &mut [F], mask: F);
}

impl BlendModeExt for BlendMode {
    fn mix<F: FineType>(&self, src: &mut [F], bg: &[F]) {
        match self.mix {
            Mix::Normal => {}
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
            Mix::Hue => Hue::mix(src, bg),
            Mix::Saturation => Saturation::mix(src, bg),
            Mix::Color => Color::mix(src, bg),
            Mix::Luminosity => Luminosity::mix(src, bg),
            // Same as `Normal`.
            Mix::Clip => {}
        }
    }

    fn compose<F: FineType>(&self, src_c: &[F; 4], bg_c: &mut [F], alpha_mask: F) {
        match self.compose {
            Compose::SrcOver => SrcOver::compose(src_c, bg_c, alpha_mask),
            Compose::Clear => Clear::compose(src_c, bg_c, alpha_mask),
            Compose::Copy => Copy::compose(src_c, bg_c, alpha_mask),
            Compose::DestOver => DestOver::compose(src_c, bg_c, alpha_mask),
            Compose::Dest => Dest::compose(src_c, bg_c, alpha_mask),
            Compose::SrcIn => SrcIn::compose(src_c, bg_c, alpha_mask),
            Compose::DestIn => DestIn::compose(src_c, bg_c, alpha_mask),
            Compose::SrcOut => SrcOut::compose(src_c, bg_c, alpha_mask),
            Compose::DestOut => DestOut::compose(src_c, bg_c, alpha_mask),
            Compose::SrcAtop => SrcAtop::compose(src_c, bg_c, alpha_mask),
            Compose::DestAtop => DestAtop::compose(src_c, bg_c, alpha_mask),
            Compose::Xor => Xor::compose(src_c, bg_c, alpha_mask),
            Compose::Plus => Plus::compose(src_c, bg_c, alpha_mask),
            // Have not been able to find a formula for this, so just fallback to Plus.
            Compose::PlusLighter => SrcOver::compose(src_c, bg_c, alpha_mask),
        }
    }
}

macro_rules! separable_mix {
    ($name:ident, $calc:expr) => {
        pub(crate) struct $name;
        impl $name {
            fn mix<F: FineType>(source: &mut [F], background: &[F]) {
                for i in 0..(COLOR_COMPONENTS - 1) {
                    source[i] = $calc(source[i], background[i]);
                }
            }
        }
    };
}

macro_rules! non_separable_mix {
    ($name:ident, $calc:expr) => {
        pub(crate) struct $name;
        impl $name {
            fn mix<F: FineType>(source: &mut [F], background: &[F]) {
                let cs = to_f32(source);
                let cb = to_f32(background);

                let res = $calc(cs, cb);
                source[..3].copy_from_slice(&from_f32(&res));
            }
        }
    };
}

impl Multiply {
    fn single<F: FineType>(src: F, bg: F) -> F {
        src.norm_mul(bg)
    }
}

impl Screen {
    fn single<F: FineType>(src: F, bg: F) -> F {
        bg.add(src).add(src.norm_mul(bg))
    }
}

impl HardLight {
    fn single<F: FineType>(src: F, bg: F) -> F {
        let two = F::from_u8(2);

        if src <= F::from_f32(0.5) {
            Multiply::single(bg, src.mul(two))
        } else {
            Screen::single(bg, two.norm_mul(src).minus(F::ONE))
        }
    }
}

separable_mix!(Multiply, |cs: F, cb| cs.norm_mul(cb));
separable_mix!(Screen, |cs: F, cb: F| cb.add_minus(cs, cs.norm_mul(cb)));
separable_mix!(Overlay, |cs: F, cb: F| HardLight::single(cb, cs));
separable_mix!(Darken, |cs: F, cb: F| cs.min(cb));
separable_mix!(Lighten, |cs: F, cb: F| cs.max(cb));
separable_mix!(ColorDodge, |cs: F, cb: F| {
    if cb == F::ZERO {
        F::ZERO
    } else if cs == F::ONE {
        F::ONE
    } else {
        F::ONE.min(cb.mul_div(F::ONE, cs.inv()))
    }
});
separable_mix!(ColorBurn, |cs: F, cb: F| {
    if cb == F::ONE {
        F::ONE
    } else if cs == F::ZERO {
        F::ZERO
    } else {
        F::ONE.minus(F::ONE.min(cb.inv().mul_div(F::ONE, cs)))
    }
});
separable_mix!(HardLight, |cs: F, cb: F| {
    if cs <= F::from_f32(0.5) {
        Multiply::single(cb, cs.mul(F::from_u8(2)))
    } else {
        Screen::single(cb, cs.mul_minus(F::from_u8(2), F::ONE))
    }
});
separable_mix!(SoftLight, |cs: F, cb: F| {
    let new_src = cs.to_f32();
    let cb = cb.to_f32();

    let d = if cb <= 0.25 {
        ((16.0 * cb - 12.0) * cb + 4.0) * cb
    } else {
        cb.sqrt()
    };

    let res = if new_src <= 0.5 {
        cb - (1.0 - 2.0 * new_src) * cb * (1.0 - cb)
    } else {
        cb + (2.0 * new_src - 1.0) * (d - cb)
    };

    F::from_f32(res)
});
separable_mix!(Difference, |cs: F, cb: F| {
    if cs <= cb { cb.minus(cs) } else { cs.minus(cb) }
});
separable_mix!(Exclusion, |cs: F, cb: F| {
    let new_src = cs.to_f32();
    let cb = cb.to_f32();

    F::from_f32((new_src + cb) - 2.0 * (new_src * cb))
});

non_separable_mix!(Hue, |cs, cb| set_lum(&set_sat(&cs, sat(&cb)), lum(&cb)));
non_separable_mix!(Saturation, |cs, cb| set_lum(
    &set_sat(&cb, sat(&cs)),
    lum(&cb)
));
non_separable_mix!(Color, |cs, cb| set_lum(&cs, lum(&cb)));
non_separable_mix!(Luminosity, |cs, cb| set_lum(&cb, lum(&cs)));

fn to_f32<F: FineType>(c: &[F]) -> [f32; 3] {
    let mut nums = [0.0; 3];

    for i in 0..3 {
        nums[i] = c[i].to_f32();
    }

    nums
}

fn from_f32<F: FineType>(c: &[f32; 3]) -> [F; 3] {
    let mut nums = [F::ZERO; 3];

    for i in 0..3 {
        nums[i] = F::from_f32(c[i]);
    }

    nums
}

fn lum(c: &[f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn sat(c: &[f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn clip_color(color: &[f32; 3]) -> [f32; 3] {
    let mut c_new = *color;

    let l = lum(&c_new);
    let n = c_new[0].min(c_new[1].min(c_new[2]));
    let x = c_new[0].max(c_new[1].max(c_new[2]));

    for c in &mut c_new {
        if n < 0.0 {
            *c = l + (((*c - l) * l) / (l - n));
        }

        if x > 1.0 {
            *c = l + (((*c - l) * (1.0 - l)) / (x - l));
        }
    }

    c_new
}

fn set_lum(c: &[f32; 3], l: f32) -> [f32; 3] {
    let mut c = *c;

    let d = l - lum(&c);
    c[0] += d;
    c[1] += d;
    c[2] += d;

    clip_color(&c)
}

fn set_sat(c: &[f32; 3], s: f32) -> [f32; 3] {
    let mut c = *c;
    let (min, tail) = c.split_at_mut(1);
    let (mid, max) = tail.split_at_mut(1);

    let mut min = &mut min[0];
    let mut mid = &mut mid[0];
    let mut max = &mut max[0];

    if *min > *mid {
        core::mem::swap(&mut min, &mut mid);
    }

    if *min > *max {
        core::mem::swap(&mut min, &mut max);
    }

    if *mid > *max {
        core::mem::swap(&mut mid, &mut max);
    }

    if *max > *min {
        *mid = ((*mid - *min) * s) / (*max - *min);
        *max = s;
    } else {
        *mid = 0.0;
        *max = 0.0;
    }

    *min = 0.0;

    c
}

fn unpremultiply<F: FineType>(color: &mut [F; 4]) {
    let alpha = color[3];

    if alpha != F::ZERO {
        for c in &mut color[0..3] {
            *c = c.mul_div(F::ONE, alpha);
        }
    }
}

fn premultiply<F: FineType>(color: &mut [F; 4]) {
    let alpha = color[3];

    for c in &mut color[0..3] {
        *c = c.norm_mul(alpha);
    }
}

macro_rules! compose {
    ($name:ident, $fa:expr, $fb:expr, $sat:expr) => {
        struct $name;

        impl $name {
            fn compose<F: FineType>(src_c: &[F; 4], bg_c: &mut [F], mask: F) {
                let al_b = bg_c[3];
                let al_s = src_c[3].norm_mul(mask);

                for i in 0..4 {
                    let fa = $fa(al_s, al_b);
                    let fb = $fb(al_s, al_b);

                    let src_c = src_c[i].norm_mul(mask);

                    if $sat {
                        bg_c[i] = src_c.norm_mul(fa).saturating_add(fb.norm_mul(bg_c[i]));
                    } else {
                        bg_c[i] = src_c.norm_mul(fa).add(fb.norm_mul(bg_c[i]));
                    }
                }
            }
        }
    };
}

compose!(Clear, |_, _| F::ZERO, |_, _| F::ZERO, false);
compose!(Copy, |_, _| F::ONE, |_, _| F::ZERO, false);
compose!(SrcOver, |_, _| F::ONE, |al_s: F, _| al_s.inv(), false);
compose!(DestOver, |_, al_b: F| al_b.inv(), |_, _| F::ONE, false);
compose!(Dest, |_, _| F::ZERO, |_, _| F::ONE, false);
compose!(Xor, |_, al_b: F| al_b.inv(), |al_s: F, _| al_s.inv(), false);
compose!(SrcIn, |_, al_b: F| al_b, |_, _| F::ZERO, false);
compose!(DestIn, |_, _| F::ZERO, |al_s: F, _| al_s, false);
compose!(SrcOut, |_, al_b: F| al_b.inv(), |_, _| F::ZERO, false);
compose!(DestOut, |_, _| F::ZERO, |al_s: F, _| al_s.inv(), false);
compose!(SrcAtop, |_, al_b: F| al_b, |al_s: F, _| al_s.inv(), false);
compose!(DestAtop, |_, al_b: F| al_b.inv(), |al_s: F, _| al_s, false);
compose!(Plus, |_, _| F::ONE, |_, _| F::ONE, true);
