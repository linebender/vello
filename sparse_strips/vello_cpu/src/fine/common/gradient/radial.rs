// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::PaintPositions;
use crate::fine::common::gradient::SimdGradientKind;
use vello_common::encode::{FocalData, RadialKind};
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
        focal_data: FocalData,
        fp0: f32x8<S>,
        fp1: f32x8<S>,
    },
}

pub(crate) struct SimdRadialKind<S: Simd> {
    inner: SimdRadialKindInner<S>,
}

impl<S: Simd> SimdRadialKind<S> {
    pub(crate) fn new(simd: S, kind: &RadialKind) -> Self {
        simd.vectorize(
            #[inline(always)]
            || {
                let inner = match kind {
                    RadialKind::Radial { bias, scale } => SimdRadialKindInner::Radial {
                        bias: f32x8::splat(simd, *bias),
                        scale: f32x8::splat(simd, *scale),
                    },
                    RadialKind::Strip { scaled_r0_squared } => SimdRadialKindInner::Strip {
                        scaled_r0_squared: f32x8::splat(simd, *scaled_r0_squared),
                    },
                    RadialKind::Focal {
                        focal_data,
                        fp0,
                        fp1,
                    } => SimdRadialKindInner::Focal {
                        fp0: f32x8::splat(simd, *fp0),
                        fp1: f32x8::splat(simd, *fp1),
                        focal_data: *focal_data,
                    },
                };

                Self { inner }
            },
        )
    }
}

impl<S: Simd> SimdGradientKind<S> for SimdRadialKind<S> {
    type Positions = PaintPositions<S, f32x8<S>>;

    #[inline(always)]
    fn cur_pos(&self, positions: &Self::Positions) -> f32x8<S> {
        let (x_pos, y_pos) = positions.current();
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
                focal_data,
                fp0,
                fp1,
            } => {
                let mut t = if focal_data.is_focal_on_circle() {
                    y_pos.mul_add(y_pos / x_pos, x_pos)
                } else if focal_data.is_well_behaved() {
                    let radius = x_pos.mul_add(x_pos, y_pos * y_pos).sqrt();
                    x_pos.mul_add(-*fp0, radius)
                } else if focal_data.is_swapped() || (1.0 - focal_data.f_focal_x < 0.0) {
                    let radius = x_pos.mul_add(x_pos, -(y_pos * y_pos)).sqrt();
                    x_pos.mul_add(-*fp0, -radius)
                } else {
                    let radius = x_pos.mul_add(x_pos, -(y_pos * y_pos)).sqrt();
                    x_pos.mul_add(-*fp0, radius)
                };

                if !focal_data.is_well_behaved() {
                    // Radii < 0 should be masked out, too.
                    let is_degenerate = simd.simd_le_f32x8(t, f32x8::splat(simd, 0.0));

                    t = simd.select_f32x8(is_degenerate, f32x8::splat(simd, f32::NAN), t);
                }

                if 1.0 - focal_data.f_focal_x < 0.0 {
                    t = f32x8::splat(simd, -1.0) * t;
                }

                if !focal_data.is_natively_focal() {
                    t += *fp1;
                }

                if focal_data.is_swapped() {
                    t = f32x8::splat(simd, 1.0) - t;
                }

                t
            }
        }
    }
}
