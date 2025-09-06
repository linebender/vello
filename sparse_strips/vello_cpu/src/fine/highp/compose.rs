// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::Splat4thExt;
use crate::peniko::{BlendMode, Compose};
use vello_common::fearless_simd::*;

pub(crate) trait ComposeExt {
    fn compose<S: Simd>(
        &self,
        simd: S,
        src_c: f32x16<S>,
        bg_c: f32x16<S>,
        alpha_mask: Option<f32x16<S>>,
    ) -> f32x16<S>;
}

impl ComposeExt for BlendMode {
    fn compose<S: Simd>(
        &self,
        simd: S,
        src_c: f32x16<S>,
        bg_c: f32x16<S>,
        alpha_mask: Option<f32x16<S>>,
    ) -> f32x16<S> {
        // There some non-obvious subtleties worth highlighting here.
        // We support two kinds of blending (in this case, we focus on compositing specifically):
        // - Isolated blending, where layers as a whole are blended together with their backdrop.
        //   If we are currently performing this kind of blending, `alpha_mask` will always be `None`.
        //   After all, there is no concrete shape opacity associated with a layer. Instead, we are
        //   just compositing the RGBA values at _all_ positions of the source layer with the backdrop
        //   layer. For example, if the backdrop contains a green rectangle and source layer is just
        //   empty, if we perform blending with `Compose::Clear`, then _everything_ will be cleared,
        //   because we are compositing the whole source layer with the whole backdrop, and not
        //   just the parts of the source layer that have actually be drawn on.
        // - Non-isolated blending, where a single path is blended with the backdrop. In this case,
        //   `alpha_mask` _might_ be `Some` and contain the alpha values of the strips we are currently
        //   compositing. Remember that strips always have a fixed height of 4, because of this, the
        //   strips might cover areas that aren't actually covered by the path (and just have an alpha
        //   value of 0, or a value between 0-254 for anti-aliased parts). Because of this, for
        //   non-isolated blending, we need to lerp the result with the backdrop using `alpha_mask`.

        let mut res = match self.compose {
            Compose::SrcOver => SrcOver::compose(simd, src_c, bg_c),
            Compose::Clear => Clear::compose(simd, src_c, bg_c),
            Compose::Copy => Copy::compose(simd, src_c, bg_c),
            Compose::DestOver => DestOver::compose(simd, src_c, bg_c),
            Compose::Dest => Dest::compose(simd, src_c, bg_c),
            Compose::SrcIn => SrcIn::compose(simd, src_c, bg_c),
            Compose::DestIn => DestIn::compose(simd, src_c, bg_c),
            Compose::SrcOut => SrcOut::compose(simd, src_c, bg_c),
            Compose::DestOut => DestOut::compose(simd, src_c, bg_c),
            Compose::SrcAtop => SrcAtop::compose(simd, src_c, bg_c),
            Compose::DestAtop => DestAtop::compose(simd, src_c, bg_c),
            Compose::Xor => Xor::compose(simd, src_c, bg_c),
            Compose::Plus => Plus::compose(simd, src_c, bg_c),
            // Have not been able to find a formula for this, so just fallback to Plus.
            Compose::PlusLighter => Plus::compose(simd, src_c, bg_c),
        };

        if let Some(alpha_mask) = alpha_mask {
            let alpha_mask_inv = 1.0 - alpha_mask;
            res = alpha_mask * res + alpha_mask_inv * bg_c;
        }

        res
    }
}

macro_rules! compose {
    ($name:ident, $fa:expr, $fb:expr, $sat:expr) => {
        struct $name;

        impl $name {
            fn compose<S: Simd>(simd: S, src_c: f32x16<S>, bg_c: f32x16<S>) -> f32x16<S> {
                let al_b = bg_c.splat_4th();
                let al_s = src_c.splat_4th();

                let fa = $fa(simd, al_s, al_b);
                let fb = $fb(simd, al_s, al_b);

                if $sat {
                    (src_c * fa + fb * bg_c)
                        .min(f32x16::splat(simd, 1.0))
                        .max(f32x16::splat(simd, 0.0))
                } else {
                    src_c * fa + fb * bg_c
                }
            }
        }
    };
}

compose!(
    Clear,
    |simd, _, _| f32x16::splat(simd, 0.0),
    |simd, _, _| f32x16::splat(simd, 0.0),
    false
);
compose!(
    Copy,
    |simd, _, _| f32x16::splat(simd, 1.0),
    |simd, _, _| f32x16::splat(simd, 0.0),
    false
);
compose!(
    SrcOver,
    |simd, _, _| f32x16::splat(simd, 1.0),
    |_, al_s: f32x16<S>, _| 1.0 - al_s,
    false
);
compose!(
    DestOver,
    |_, _, al_b: f32x16<S>| 1.0 - al_b,
    |simd, _, _| f32x16::splat(simd, 1.0),
    false
);
compose!(
    Dest,
    |simd, _, _| f32x16::splat(simd, 0.0),
    |simd, _, _| f32x16::splat(simd, 1.0),
    false
);
compose!(
    Xor,
    |_, _, al_b: f32x16<S>| 1.0 - al_b,
    |_, al_s: f32x16<S>, _| 1.0 - al_s,
    false
);
compose!(
    SrcIn,
    |_, _, al_b: f32x16<S>| al_b,
    |simd, _, _| f32x16::splat(simd, 0.0),
    false
);
compose!(
    DestIn,
    |simd, _, _| f32x16::splat(simd, 0.0),
    |_, al_s: f32x16<S>, _| al_s,
    false
);
compose!(
    SrcOut,
    |_, _, al_b: f32x16<S>| 1.0 - al_b,
    |simd, _, _| f32x16::splat(simd, 0.0),
    false
);
compose!(
    DestOut,
    |simd, _, _| f32x16::splat(simd, 0.0),
    |_, al_s: f32x16<S>, _| 1.0 - al_s,
    false
);
compose!(
    SrcAtop,
    |_, _, al_b: f32x16<S>| al_b,
    |_, al_s: f32x16<S>, _| 1.0 - al_s,
    false
);
compose!(
    DestAtop,
    |_, _, al_b: f32x16<S>| 1.0 - al_b,
    |_, al_s: f32x16<S>, _| al_s,
    false
);
compose!(
    Plus,
    |simd, _, _| f32x16::splat(simd, 1.0),
    |simd, _, _| f32x16::splat(simd, 1.0),
    true
);
