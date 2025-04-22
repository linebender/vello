// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Support for blending. See <https://www.w3.org/TR/compositing-1/#introduction> for
//! an introduction as well as the corresponding formulas.

use crate::fine::COLOR_COMPONENTS;
use crate::util::scalar::div_255;
use vello_common::peniko::{BlendMode, Compose, Mix};

pub(crate) mod fill {
    use crate::fine::blend::{BlendModeExt, mix};
    use crate::fine::{COLOR_COMPONENTS, TILE_HEIGHT_COMPONENTS};
    use vello_common::peniko::BlendMode;

    pub(crate) fn blend<T: Iterator<Item = [u8; COLOR_COMPONENTS]>>(
        target: &mut [u8],
        mut color_iter: T,
        blend_mode: BlendMode,
    ) {
        for strip in target.chunks_exact_mut(TILE_HEIGHT_COMPONENTS) {
            for bg_c in strip.chunks_exact_mut(COLOR_COMPONENTS) {
                let mixed_src_color = mix(color_iter.next().unwrap(), bg_c, blend_mode);

                blend_mode.compose(&mixed_src_color, bg_c, 255);
            }
        }
    }
}

pub(crate) mod strip {
    use crate::fine::blend::{BlendModeExt, mix};
    use crate::fine::{COLOR_COMPONENTS, TILE_HEIGHT_COMPONENTS};
    use vello_common::peniko::BlendMode;
    use vello_common::tile::Tile;

    pub(crate) fn blend<
        T: Iterator<Item = [u8; COLOR_COMPONENTS]>,
        A: Iterator<Item = [u8; Tile::HEIGHT as usize]>,
    >(
        target: &mut [u8],
        mut color_iter: T,
        blend_mode: BlendMode,
        mut alphas: A,
    ) {
        for bg_col in target.chunks_exact_mut(TILE_HEIGHT_COMPONENTS) {
            let masks = alphas.next().unwrap();

            for (bg_pix, mask) in bg_col.chunks_exact_mut(Tile::HEIGHT as usize).zip(masks) {
                let mixed_src_color = mix(color_iter.next().unwrap(), bg_pix, blend_mode);

                blend_mode.compose(&mixed_src_color, bg_pix, mask);
            }
        }
    }
}

fn mix(mut src_c: [u8; 4], bg_c: &[u8], blend_mode: BlendMode) -> [u8; 4] {
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
        src_c[i] =
            div_255((255 - bg_alpha) as u16 * src_c[i] as u16 + bg_alpha as u16 * mixed[i] as u16)
                as u8;
    }

    // Premultiply again.
    premultiply(&mut src_c);

    src_c
}

pub(crate) trait BlendModeExt {
    fn mix(&self, src: &mut [u8], bg: &[u8]);
    fn compose(&self, src_c: &[u8; 4], bg_c: &mut [u8], mask: u8);
}

impl BlendModeExt for BlendMode {
    fn mix(&self, src: &mut [u8], bg: &[u8]) {
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

    fn compose(&self, src_c: &[u8; 4], bg_c: &mut [u8], alpha_mask: u8) {
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
            fn mix(source: &mut [u8], background: &[u8]) {
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
            fn mix(source: &mut [u8], background: &[u8]) {
                let cs = to_f32(source);
                let cb = to_f32(background);

                let res = $calc(cs, cb);
                source[..3].copy_from_slice(&from_f32(&res));
            }
        }
    };
}

impl Multiply {
    fn single(src: u8, bg: u8) -> u8 {
        div_255(src as u16 * bg as u16) as u8
    }
}

impl Screen {
    fn single(src: u8, bg: u8) -> u8 {
        (bg as u16 + src as u16 - div_255(src as u16 * bg as u16)) as u8
    }
}

impl HardLight {
    fn single(src: u8, bg: u8) -> u8 {
        if src <= 127 {
            Multiply::single(bg, 2 * src)
        } else {
            Screen::single(bg, ((2 * src as u16) - 255) as u8)
        }
    }
}

separable_mix!(Multiply, |cs, cb| div_255(cs as u16 * cb as u16) as u8);
separable_mix!(
    Screen,
    |cs, cb| (cb as u16 + cs as u16 - div_255(cs as u16 * cb as u16)) as u8
);
separable_mix!(Overlay, |cs, cb| HardLight::single(cb, cs));
separable_mix!(Darken, |cs: u8, cb| cs.min(cb));
separable_mix!(Lighten, |cs: u8, cb| cs.max(cb));
separable_mix!(ColorDodge, |cs: u8, cb| {
    if cb == 0 {
        0
    } else if cs == 255 {
        255
    } else {
        255.min((cb as u16 * 255) / (255 - cs) as u16) as u8
    }
});
separable_mix!(ColorBurn, |cs: u8, cb| {
    if cb == 255 {
        255
    } else if cs == 0 {
        0
    } else {
        255 - 255.min((255 - cb) as u16 * 255 / cs as u16) as u8
    }
});
separable_mix!(HardLight, |cs: u8, cb| {
    if cs <= 127 {
        Multiply::single(cb, 2 * cs)
    } else {
        Screen::single(cb, ((2 * cs as u16) - 255) as u8)
    }
});
separable_mix!(SoftLight, |cs: u8, cb| {
    let new_src = cs as f32 / 255.0;
    let cb = cb as f32 / 255.0;

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

    (res * 255.0 + 0.5) as u8
});
separable_mix!(Difference, |cs: u8, cb| {
    if cs <= cb { cb - cs } else { cs - cb }
});
separable_mix!(Exclusion, |cs: u8, cb| ((cs as u16 + cb as u16)
    - 2 * div_255(cs as u16 * cb as u16))
    as u8);

non_separable_mix!(Hue, |cs, cb| set_lum(&set_sat(&cs, sat(&cb)), lum(&cb)));
non_separable_mix!(Saturation, |cs, cb| set_lum(
    &set_sat(&cb, sat(&cs)),
    lum(&cb)
));
non_separable_mix!(Color, |cs, cb| set_lum(&cs, lum(&cb)));
non_separable_mix!(Luminosity, |cs, cb| set_lum(&cb, lum(&cs)));

fn to_f32(c: &[u8]) -> [f32; 3] {
    let mut nums = [0.0; 3];

    for i in 0..3 {
        nums[i] = c[i] as f32 / 255.0;
    }

    nums
}

fn from_f32(c: &[f32; 3]) -> [u8; 3] {
    let mut nums = [0; 3];

    for i in 0..3 {
        nums[i] = (c[i] * 255.0 + 0.5) as u8;
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

fn unpremultiply(color: &mut [u8; 4]) {
    let alpha = color[3] as u16;

    if alpha != 0 {
        for c in &mut color[0..3] {
            *c = ((*c as u16 * 255) / alpha) as u8;
        }
    }
}

fn premultiply(color: &mut [u8; 4]) {
    let alpha = color[3] as u16;

    for c in &mut color[0..3] {
        *c = div_255(*c as u16 * alpha) as u8;
    }
}

macro_rules! compose {
    ($name:ident, $fa:expr, $fb:expr, $sat:expr) => {
        struct $name;

        impl $name {
            fn compose(src_c: &[u8; 4], bg_c: &mut [u8], mask: u8) {
                let al_b = bg_c[3] as u16;
                let al_s = div_255(src_c[3] as u16 * mask as u16);

                for i in 0..4 {
                    let fa = $fa(al_s, al_b);
                    let fb = $fb(al_s, al_b);

                    let src_c = div_255(src_c[i] as u16 * mask as u16);

                    if $sat {
                        bg_c[i] = (div_255(src_c * fa) as u8)
                            .saturating_add(div_255(fb * bg_c[i] as u16) as u8);
                    } else {
                        bg_c[i] = (div_255(src_c * fa) + div_255(fb * bg_c[i] as u16)) as u8;
                    }
                }
            }
        }
    };
}

compose!(Clear, |_, _| 0, |_, _| 0, false);
compose!(Copy, |_, _| 255, |_, _| 0, false);
compose!(SrcOver, |_, _| 255, |al_s, _| 255 - al_s, false);
compose!(DestOver, |_, al_b| 255 - al_b, |_, _| 255, false);
compose!(Dest, |_, _| 0, |_, _| 255, false);
compose!(Xor, |_, al_b| 255 - al_b, |al_s, _| 255 - al_s, false);
compose!(SrcIn, |_, al_b| al_b, |_, _| 0, false);
compose!(DestIn, |_, _| 0, |al_s, _| al_s, false);
compose!(SrcOut, |_, al_b| 255 - al_b, |_, _| 0, false);
compose!(DestOut, |_, _| 0, |al_s, _| 255 - al_s, false);
compose!(SrcAtop, |_, al_b| al_b, |al_s, _| 255 - al_s, false);
compose!(DestAtop, |_, al_b| 255 - al_b, |al_s, _| al_s, false);
compose!(Plus, |_, _| 255, |_, _| 255, true);
