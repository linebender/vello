// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampling helpers for image drawing.

use vello_common::geometry::RectU16;
use vello_common::kurbo::Affine;

/// A rectangular source region sampled from an image input (e.g., [`crate::TextureId`]), paired
/// with a transform of the rectangle into the destination.
#[derive(Debug, Clone, Copy)]
pub struct SampleRect {
    /// Source region in texel coordinates.
    pub source_region: RectU16,

    /// Whether the sampled source region may contain non-opaque pixels.
    ///
    /// Only set this to `false` if every pixel is guaranteed to be opaque (e.g. in a texture
    /// generated from a JPEG image). If you set this to `false` even though there are non-opaque
    /// pixels, you will get wrong rendering.
    ///
    /// If unsure, always set this to `true`.
    pub may_have_transparency: bool,

    /// Transform mapping the local source region to the destination.
    ///
    /// This maps from the *local* rectangle into the destination, ignoring the origin of
    /// [`Self::source_region`].
    pub transform: Affine,
}

impl SampleRect {
    /// Create a new [`SampleRect`].
    pub fn new(source_region: RectU16, transform: Affine) -> Self {
        Self {
            source_region,
            may_have_transparency: true,
            transform,
        }
    }

    /// Indicate that the sample rect only contains opaque pixels.
    #[must_use]
    pub fn with_opaque_hint(mut self) -> Self {
        self.may_have_transparency = false;
        self
    }
}
