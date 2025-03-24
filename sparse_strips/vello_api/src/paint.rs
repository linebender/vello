// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types for paints.

use peniko::color::{AlphaColor, PremulRgba8, Srgb};

pub(crate) const MAX_INDEX: usize = u32::MAX as usize;

/// A paint that needs to be resolved via its index.
// In the future, we might add additional flags, that's why we have
// this thin wrapper around u32, so we can change the underlying
// representation without breaking the API.
#[derive(Debug, Clone)]
pub struct IndexedPaint(u32);

impl IndexedPaint {
    /// Create a new indexed paint from an index.
    pub fn new(index: usize) -> Self {
        assert!(index <= MAX_INDEX, "exceeded the maximum number of paints");

        #[allow(
            clippy::cast_possible_truncation,
            reason = "Guaranteed to be in-bounds"
        )]
        Self(index as u32)
    }

    /// Return the index of the paint.
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

/// A paint used for filling or stroking paths.
#[derive(Debug, Clone)]
pub enum Paint {
    /// A premultiplied RGBA8 color.
    Solid(PremulRgba8),
    /// An paint that needs to be resolved via an index.
    Indexed(IndexedPaint),
}

impl From<AlphaColor<Srgb>> for Paint {
    fn from(value: AlphaColor<Srgb>) -> Self {
        // TODO: This might be slow on x86, see https://github.com/linebender/color/issues/142.
        // Since we only do that conversion once per path it might not be critical, but should
        // still be measured. This also applies to all other usages of `to_rgba8` in the current
        // code.
        Self::Solid(value.premultiply().to_rgba8())
    }
}
