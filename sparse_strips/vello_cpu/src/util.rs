// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

pub(crate) mod scalar {
    #[inline(always)]
    pub(crate) const fn div_255(val: u16) -> u16 {
        (val + 1 + (val >> 8)) >> 8
    }
}
