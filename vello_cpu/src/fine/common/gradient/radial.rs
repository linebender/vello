// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::common::gradient::SimdGradientKind;
use fearless_simd_macros::simd;
use vello_common::encode::{FocalKind, RadialKind};
use vello_common::fearless_simd::{Simd, SimdBase, SimdFloat, f32x8};

pub(crate) enum SimdRadialKindInner<S: Simd> {
    Radial {
        bias: f32x8<S>,
        scale: f32x8<S>,
    },
    Strip {
        scaled_r0_squared: f32x8<S>,
    },
    Focal {
        kind: FocalKind,
        fp0: f32x8<S>,
        scale: f32x8<S>,
        bias: f32x8<S>,
    },
}

pub(crate) struct SimdRadialKind<S: Simd> {
    inner: SimdRadialKindInner<S>,
}

impl<S: Simd> SimdRadialKind<S> {
    #[simd]
    pub(crate) fn new(simd: S, kind: RadialKind) -> Self {
        let inner = match kind {
            RadialKind::Radial { bias, scale } => SimdRadialKindInner::Radial {
                bias: f32x8::splat(simd, bias),
                scale: f32x8::splat(simd, scale),
            },
            RadialKind::Strip { scaled_r0_squared } => SimdRadialKindInner::Strip {
                scaled_r0_squared: f32x8::splat(simd, scaled_r0_squared),
            },
            RadialKind::Focal {
                kind,
                fp0,
                scale,
                bias,
            } => SimdRadialKindInner::Focal {
                kind,
                fp0: f32x8::splat(simd, fp0),
                scale: f32x8::splat(simd, scale),
                bias: f32x8::splat(simd, bias),
            },
        };

        Self { inner }
    }
}

impl<S: Simd> SimdGradientKind<S> for SimdRadialKind<S> {
    #[inline(always)]
    fn cur_pos(&self, x_pos: f32x8<S>, y_pos: f32x8<S>) -> f32x8<S> {
        let simd = x_pos.simd;

        match &self.inner {
            SimdRadialKindInner::Radial { bias, scale } => {
                let radius = x_pos.mul_add(x_pos, y_pos * y_pos).sqrt();

                radius.mul_add(*scale, *bias)
            }
            SimdRadialKindInner::Strip { scaled_r0_squared } => {
                let p1 = y_pos.mul_add(-y_pos, *scaled_r0_squared);

                let mask = simd.simd_lt_f32x8(p1, f32x8::splat(simd, 0.0));
                simd.select_f32x8(mask, f32x8::splat(simd, f32::NAN), x_pos + p1.sqrt())
            }
            SimdRadialKindInner::Focal {
                kind,
                fp0,
                scale,
                bias,
            } => {
                let mut t = match kind {
                    FocalKind::OnCircle => y_pos.mul_add(y_pos / x_pos, x_pos),
                    FocalKind::WellBehaved => {
                        let radius = x_pos.mul_add(x_pos, y_pos * y_pos).sqrt();
                        x_pos.mul_add(-*fp0, radius)
                    }
                    FocalKind::General { negate_root } => {
                        let radius = x_pos.mul_add(x_pos, -(y_pos * y_pos)).sqrt();
                        let radius = if *negate_root { -radius } else { radius };
                        x_pos.mul_add(-*fp0, radius)
                    }
                };

                if *kind != FocalKind::WellBehaved {
                    let is_degenerate = simd.simd_le_f32x8(t, f32x8::splat(simd, 0.0));

                    t = simd.select_f32x8(is_degenerate, f32x8::splat(simd, f32::NAN), t);
                }

                t.mul_add(*scale, *bias)
            }
        }
    }
}
