// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::Splat4thExt;
use crate::peniko::{BlendMode, Compose};
use crate::util::NormalizedMulExt;
use vello_common::fearless_simd::*;

pub(crate) trait ComposeExt {
    fn compose<S: Simd>(
        &self,
        simd: S,
        src_c: u8x32<S>,
        bg_c: u8x32<S>,
        alpha_mask: u8x32<S>,
    ) -> u8x32<S>;
}

impl ComposeExt for BlendMode {
    fn compose<S: Simd>(
        &self,
        simd: S,
        src_c: u8x32<S>,
        bg_c: u8x32<S>,
        alpha_mask: u8x32<S>,
    ) -> u8x32<S> {
        match self.compose {
            Compose::SrcOver => SrcOver::compose(simd, src_c, bg_c, alpha_mask),
            Compose::Clear => Clear::compose(simd, src_c, bg_c, alpha_mask),
            Compose::Copy => Copy::compose(simd, src_c, bg_c, alpha_mask),
            Compose::DestOver => DestOver::compose(simd, src_c, bg_c, alpha_mask),
            Compose::Dest => Dest::compose(simd, src_c, bg_c, alpha_mask),
            Compose::SrcIn => SrcIn::compose(simd, src_c, bg_c, alpha_mask),
            Compose::DestIn => DestIn::compose(simd, src_c, bg_c, alpha_mask),
            Compose::SrcOut => SrcOut::compose(simd, src_c, bg_c, alpha_mask),
            Compose::DestOut => DestOut::compose(simd, src_c, bg_c, alpha_mask),
            Compose::SrcAtop => SrcAtop::compose(simd, src_c, bg_c, alpha_mask),
            Compose::DestAtop => DestAtop::compose(simd, src_c, bg_c, alpha_mask),
            Compose::Xor => Xor::compose(simd, src_c, bg_c, alpha_mask),
            Compose::Plus => Plus::compose(simd, src_c, bg_c, alpha_mask),
            // Have not been able to find a formula for this, so just fallback to Plus.
            Compose::PlusLighter => Plus::compose(simd, src_c, bg_c, alpha_mask),
        }
    }
}

macro_rules! compose {
    ($name:ident, $fa:expr, $fb:expr, $sat:expr) => {
        struct $name;

        impl $name {
            fn compose<S: Simd>(
                simd: S,
                src_c: u8x32<S>,
                bg_c: u8x32<S>,
                mask: u8x32<S>,
            ) -> u8x32<S> {
                let al_b = bg_c.splat_4th();
                let al_s = src_c.splat_4th().normalized_mul(mask);

                let fa = $fa(simd, al_s, al_b);
                let fb = $fb(simd, al_s, al_b);

                let src_c = src_c.normalized_mul(mask);

                if $sat {
                    simd.narrow_u16x32(
                        (simd.widen_u8x32(src_c.normalized_mul(fa))
                            + simd.widen_u8x32(fb.normalized_mul(bg_c)))
                        .min(u16x32::splat(simd, 255))
                        .max(u16x32::splat(simd, 0)),
                    )
                } else {
                    src_c.normalized_mul(fa) + fb.normalized_mul(bg_c)
                }
            }
        }
    };
}

compose!(
    Clear,
    |simd, _, _| u8x32::splat(simd, 0),
    |simd, _, _| u8x32::splat(simd, 0),
    false
);
compose!(
    Copy,
    |simd, _, _| u8x32::splat(simd, 255),
    |simd, _, _| u8x32::splat(simd, 0),
    false
);
compose!(
    SrcOver,
    |simd, _, _| u8x32::splat(simd, 255),
    |_, al_s: u8x32<S>, _| 255 - al_s,
    false
);
compose!(
    DestOver,
    |_, _, al_b: u8x32<S>| 255 - al_b,
    |simd, _, _| u8x32::splat(simd, 255),
    false
);
compose!(
    Dest,
    |simd, _, _| u8x32::splat(simd, 0),
    |simd, _, _| u8x32::splat(simd, 255),
    false
);
compose!(
    Xor,
    |_, _, al_b: u8x32<S>| 255 - al_b,
    |_, al_s: u8x32<S>, _| 255 - al_s,
    false
);
compose!(
    SrcIn,
    |_, _, al_b: u8x32<S>| al_b,
    |simd, _, _| u8x32::splat(simd, 0),
    false
);
compose!(
    DestIn,
    |simd, _, _| u8x32::splat(simd, 0),
    |_, al_s: u8x32<S>, _| al_s,
    false
);
compose!(
    SrcOut,
    |_, _, al_b: u8x32<S>| 255 - al_b,
    |simd, _, _| u8x32::splat(simd, 0),
    false
);
compose!(
    DestOut,
    |simd, _, _| u8x32::splat(simd, 0),
    |_, al_s: u8x32<S>, _| 255 - al_s,
    false
);
compose!(
    SrcAtop,
    |_, _, al_b: u8x32<S>| al_b,
    |_, al_s: u8x32<S>, _| 255 - al_s,
    false
);
compose!(
    DestAtop,
    |_, _, al_b: u8x32<S>| 255 - al_b,
    |_, al_s: u8x32<S>, _| al_s,
    false
);
compose!(
    Plus,
    |simd, _, _| u8x32::splat(simd, 255),
    |simd, _, _| u8x32::splat(simd, 255),
    true
);
