// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Runtime bindings for externally owned textures.

use crate::RenderError;
#[cfg(feature = "wgpu_external_texture")]
use crate::render::common::encode_wgpu_external_texture_paint;
use crate::render::common::{GpuEncodedPaint, encode_external_texture_paint};
use core::fmt::Debug;
use hashbrown::HashMap;
use vello_common::TextureId;
use vello_common::encode::EncodedImage;
use vello_common::paint::TextureRegion;
use wgpu::{Texture, TextureView};

/// A runtime binding for a [`TextureId`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TextureBinding {
    /// A premultiplied RGBA texture view; see [`TextureBindings::insert`].
    TextureView(TextureView),
    /// A wgpu external texture; see [`TextureBindings::insert_wgpu_external_texture`].
    #[cfg(feature = "wgpu_external_texture")]
    WgpuExternalTexture {
        /// The external texture.
        texture: wgpu::ExternalTexture,
        /// The external texture's width, as passed to [`wgpu::ExternalTextureDescriptor`].
        width: u32,
        /// The external texture's height, as passed to [`wgpu::ExternalTextureDescriptor`].
        height: u32,
    },
}

impl TextureBinding {
    /// Width and height of the texture space that image paints sample from.
    pub(crate) fn size(&self) -> (u32, u32) {
        match self {
            Self::TextureView(view) => {
                let texture = view.texture();
                (texture.width(), texture.height())
            }
            #[cfg(feature = "wgpu_external_texture")]
            Self::WgpuExternalTexture { width, height, .. } => (*width, *height),
        }
    }
}

/// Runtime bindings for [externally owned textures](`TextureId`) sampled by image paints.
#[derive(Debug, Default, Clone)]
pub struct TextureBindings {
    bindings: HashMap<TextureId, TextureBinding>,
}

impl TextureBindings {
    /// Create an empty binding map.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a texture binding.
    ///
    /// The [`TextureView`] must fit the following binding type.
    ///
    /// ```ignore
    /// wgpu::BindGroupLayoutEntry {
    ///     binding: 0,
    ///     visibility: wgpu::ShaderStages::FRAGMENT,
    ///     ty: wgpu::BindingType::Texture {
    ///         sample_type: wgpu::TextureSampleType::Float { filterable: true },
    ///         view_dimension: wgpu::TextureViewDimension::D2,
    ///         multisampled: false,
    ///     },
    ///     count: None,
    /// }
    /// ```
    ///
    /// This means the view must be a non-array 2D view of a filterable float texture (e.g.
    /// `Rgba8Unorm`; integer and unfilterable float formats are rejected by wgpu at bind time).
    /// The underlying texture must include [`wgpu::TextureUsages::TEXTURE_BINDING`], and only
    /// mip level 0 is read. The view must start at mip level 0 of its texture.
    #[inline]
    pub fn insert(&mut self, texture_id: TextureId, view: TextureView) {
        self.bindings
            .insert(texture_id, TextureBinding::TextureView(view));
    }

    /// Insert or replace a [`wgpu::ExternalTexture`] binding.
    ///
    /// The renderer's device must have [`wgpu::Features::EXTERNAL_TEXTURE`] enabled. When Vello
    /// samples the texture, wgpu applies its sample transform and, for multi-planar formats, the
    /// YCbCr → RGB and color space conversion. Multi-planar formats always produce opaque colors,
    /// while an `Rgba` external texture is read as-is and must contain premultiplied RGBA.
    ///
    /// Image paints must sample the full texture ([`TextureRegion::Full`]) with
    /// [`Extend::Pad`](vello_common::peniko::Extend::Pad), low or medium quality, and no tint
    /// other than opacity. Other paints fail with
    /// [`RenderError::UnsupportedExternalTextureSampling`]. To crop or rotate, use the paint
    /// transform and limit the drawn geometry.
    ///
    /// `width` and `height` are the texture's size in texels, which wgpu does not expose. Pass
    /// the [`wgpu::ExternalTextureDescriptor`]'s size, or the first plane's size if the
    /// descriptor uses `(0, 0)`.
    ///
    /// # Panics
    ///
    /// Panics if `width` or `height` is zero.
    #[cfg(feature = "wgpu_external_texture")]
    #[inline]
    pub fn insert_wgpu_external_texture(
        &mut self,
        texture_id: TextureId,
        external_texture: wgpu::ExternalTexture,
        width: u32,
        height: u32,
    ) {
        assert!(
            width > 0 && height > 0,
            "wgpu external texture size must be non-zero, got {width}x{height}"
        );
        self.bindings.insert(
            texture_id,
            TextureBinding::WgpuExternalTexture {
                texture: external_texture,
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
        render_target: &Texture,
    ) -> Result<GpuEncodedPaint, RenderError> {
        let texture_binding = self
            .get(texture_id)
            .ok_or(RenderError::MissingTextureBinding(texture_id))?;

        let (texture_width, texture_height) = texture_binding.size();
        let texture_width = u16::try_from(texture_width).unwrap();
        let texture_height = u16::try_from(texture_height).unwrap();
        match texture_binding {
            TextureBinding::TextureView(view) => {
                if view.texture() == render_target {
                    return Err(RenderError::TextureFeedbackLoop(texture_id));
                }
                Ok(encode_external_texture_paint(
                    image,
                    source_region,
                    texture_width,
                    texture_height,
                ))
            }
            #[cfg(feature = "wgpu_external_texture")]
            TextureBinding::WgpuExternalTexture { .. } => {
                encode_wgpu_external_texture_paint(image, texture_width, texture_height)
                    .ok_or(RenderError::UnsupportedExternalTextureSampling(texture_id))
            }
        }
    }

    /// Get a texture binding.
    #[inline]
    pub(crate) fn get(&self, texture_id: TextureId) -> Option<&TextureBinding> {
        self.bindings.get(&texture_id)
    }

    /// Remove a texture binding.
    ///
    /// This returns the removed binding if it existed.
    #[inline]
    pub fn remove(&mut self, texture_id: TextureId) -> Option<TextureBinding> {
        self.bindings.remove(&texture_id)
    }
}
