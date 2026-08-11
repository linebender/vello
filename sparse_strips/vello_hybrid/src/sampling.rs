// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampling helpers for image drawing.

use vello_common::TextureId;
use vello_common::geometry::RectU16;
use vello_common::kurbo::Rect;
use vello_common::peniko::ImageSampler;

/// A rectangular region sampled from an externally bound texture and drawn into a destination
/// rectangle.
#[derive(Debug, Clone, Copy)]
pub struct ExternalTextureRect {
    /// The external texture to sample.
    pub texture_id: TextureId,

    /// Source region in texel coordinates.
    pub source_region: RectU16,

    /// Destination rectangle in local scene coordinates.
    pub destination_rect: Rect,

    /// Sampling parameters.
    ///
    /// A sampler alpha other than `1.0` is not currently supported.
    pub sampler: ImageSampler,

    /// Whether the sampled source region may contain non-opaque pixels.
    ///
    /// Only set this to `false` if every pixel is guaranteed to be opaque (e.g. in a texture
    /// generated from a JPEG image). If you set this to `false` even though there are non-opaque
    /// pixels, you will get wrong rendering.
    ///
    /// If unsure, always set this to `true`.
    pub may_have_transparency: bool,
}
