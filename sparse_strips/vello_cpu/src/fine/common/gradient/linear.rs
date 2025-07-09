// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::common::gradient::SimdGradientKind;
use core::marker::PhantomData;
use vello_common::encode::LinearKind;
use vello_common::fearless_simd::{Simd, f32x8};

#[derive(Debug)]
pub(crate) struct SimdLinearKind<S: Simd> {
    // For consistency with the other gradient types.
    phantom_data: PhantomData<S>,
}

impl<S: Simd> SimdLinearKind<S> {
    pub(crate) fn new(_: S, _: LinearKind) -> Self {
        Self {
            phantom_data: PhantomData,
        }
    }
}

impl<S: Simd> SimdGradientKind<S> for SimdLinearKind<S> {
    #[inline(always)]
    fn cur_pos(&self, x_pos: f32x8<S>, _: f32x8<S>) -> f32x8<S> {
        x_pos
    }
}
