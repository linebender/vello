// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Runtime bindings for externally owned textures.

use crate::RenderError;
use crate::render::common::{GpuEncodedPaint, encode_external_texture_paint};
use hashbrown::HashMap;
use vello_common::TextureId;
use vello_common::encode::EncodedImage;
use vello_common::paint::TextureRegion;
use web_sys::WebGlTexture;

/// Runtime bindings for [externally owned textures](`TextureId`) sampled by image paints.
#[derive(Debug, Default, Clone)]
pub struct WebGlTextureBindings {
    textures: HashMap<TextureId, WebGlTextureWithDimensions>,
}

impl WebGlTextureBindings {
    /// Create an empty binding map.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a texture binding.
    ///
    /// The texture must satisfy the following requirements.
    ///
    /// - It must be complete for sampling, with a defined mip level 0 in a linearly filterable
    ///   `sampler2D` format (e.g. `RGBA8`). Integer and unfilterable float formats are unsupported.
    /// - Its contents are treated as premultiplied alpha, and are sampled in texel space with
    ///   the origin at the first uploaded texel. A texture uploaded with `UNPACK_FLIP_Y_WEBGL`
    ///   enabled therefore samples vertically flipped.
    /// - `width` and `height` must match the dimensions of its mip level 0, otherwise rendering
    ///   might be corrupted.
    #[inline]
    pub fn insert(
        &mut self,
        texture_id: TextureId,
        texture: WebGlTexture,
        width: u16,
        height: u16,
    ) {
        assert!(
            width > 0 && height > 0,
            "external texture dimensions must not be zero"
        );

        self.textures.insert(
            texture_id,
            WebGlTextureWithDimensions {
                texture,
                width,
                height,
            },
        );
    }

    /// Encodes an image paint that samples the binding of `texture_id`.
    pub(crate) fn encode_paint(
        &self,
        texture_id: TextureId,
        image: &EncodedImage,
        source_region: TextureRegion,
        render_target: Option<&WebGlTexture>,
    ) -> Result<GpuEncodedPaint, RenderError> {
        let binding = self
            .get(texture_id)
            .ok_or(RenderError::MissingTextureBinding(texture_id))?;
        if render_target.is_some_and(|target| *target == binding.texture) {
            return Err(RenderError::TextureFeedbackLoop(texture_id));
        }
        Ok(encode_external_texture_paint(
            image,
            source_region,
            binding.width,
            binding.height,
        ))
    }

    /// Get a texture binding.
    #[inline]
    pub(crate) fn get(&self, texture_id: TextureId) -> Option<&WebGlTextureWithDimensions> {
        self.textures.get(&texture_id)
    }

    /// Remove a texture binding.
    ///
    /// This returns the removed texture binding if it existed.
    #[inline]
    pub fn remove(&mut self, texture_id: TextureId) -> Option<WebGlTexture> {
        self.textures
            .remove(&texture_id)
            .map(|binding| binding.texture)
    }
}

/// Wrapper for `WebGlTexture` with known dimensions.
#[derive(Debug, Clone)]
pub(crate) struct WebGlTextureWithDimensions {
    /// The WebGL texture.
    pub(crate) texture: WebGlTexture,
    /// The width of the texture.
    pub(crate) width: u16,
    /// The height of the texture.
    pub(crate) height: u16,
}
