// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Backend agnostic renderer module.

mod paint;

#[cfg(all(feature = "wgpu", feature = "wgpu_external_texture"))]
pub(crate) use paint::encode_wgpu_external_texture_paint;
pub(crate) use paint::{
    GpuBlurredRoundedRect, GpuEncodedImage, GpuEncodedPaint, GpuLinearGradient, GpuSweepGradient,
    encode_native_image_paint, encode_radial_gradient, pack_image_offset, pack_image_params,
    pack_image_size, pack_texture_width_and_extend_mode, pack_tint,
};

use crate::IntermediateTextureError;
use crate::blend::GpuBlendInstance;
use crate::copy::GpuCopyInstance;
use crate::filter::FILTER_ATLAS_PADDING;
use crate::scene::{LayersConfig, MemorySettings, RecordedDraw};
use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::geometry::{RectU16, SizeU16};
use vello_common::record::CommandRecorder;

// TODO: If we want to use native bilinear sampling for uploaded images,
// we can pass 1 instead of 0 here.
pub(crate) const IMAGE_PADDING: u16 = 0;

/// Controls how the output target is cleared before drawing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClearSettings<'a> {
    /// Clear the complete viewport to `color`.
    Viewport {
        /// The clear color.
        color: AlphaColor<Srgb>,
    },
    /// Clear each rectangle to `color`.
    Rects {
        /// The clear color.
        color: AlphaColor<Srgb>,
        /// The rectangles to clear.
        rects: &'a [RectU16],
    },
}

impl ClearSettings<'_> {
    pub(crate) fn clear_color(self) -> AlphaColor<Srgb> {
        match self {
            Self::Viewport { color } | Self::Rects { color, .. } => color,
        }
    }
}

/// Controls whether existing target contents are preserved or cleared before drawing.
pub type TargetInit<'a> = vello_common::TargetInit<ClearSettings<'a>>;

impl Default for ClearSettings<'_> {
    fn default() -> Self {
        Self::Viewport {
            color: AlphaColor::TRANSPARENT,
        }
    }
}

/// Texture limits reported by the rendering device.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeviceLimits {
    /// Maximum width or height of a two-dimensional texture.
    pub(crate) max_texture_dimension_2d: u16,
}

impl DeviceLimits {
    pub(crate) fn resource_texture_dimension_2d(self) -> u32 {
        // See https://github.com/linebender/vello/pull/1830 for why
        // we enforce an additional limit on intermediate data textures.
        const MAX_RESOURCE_TEXTURE_DIMENSION_2D: u16 = 4096;

        let dimension = u32::from(
            self.max_texture_dimension_2d
                .min(MAX_RESOURCE_TEXTURE_DIMENSION_2D),
        );

        debug_assert!(
            dimension.is_power_of_two(),
            "Resource texture dimension must be a power of two"
        );

        dimension
    }
}

// This new type only serves the purpose of documenting the below invariants in a single place.
/// A scratch texture.
///
/// Right now, for a given scene, we always have _at most_ one scratch texture. At the moment,
/// it is used for the following purposes:
/// - Serving as the write destination for a blending operation of two textures.
/// - Serving as a temporary space for persisting the original version of a filter layer.
///
/// The scratch texture has the same dimensions as every intermediate layer page and does not have
/// separate atlas allocations. Blends and filters use the same rectangle coordinates in scratch as
/// in the corresponding parent or filter-layer texture. Every area later read from scratch is
/// therefore fully overwritten first. **Because of this, it is currently safe to never explicitly
/// clear the scratch texture after it was used, even across frames**. While it would still be best
/// practice to do so, in the interest of getting the best performance on low-tier devices, we take
/// this shortcut and explicitly document this constraint.
///
/// If the purpose of a scratch texture is ever expanded upon, this needs to be revisited.
#[derive(Debug)]
pub(crate) struct ScratchTexture<T>(T);

impl<T> ScratchTexture<T> {
    pub(crate) fn new(texture: T) -> Self {
        Self(texture)
    }

    pub(crate) fn get(&self) -> &T {
        &self.0
    }
}

/// Scratch allocations reused while rendering a frame.
#[derive(Debug, Default)]
pub(crate) struct ScratchBuffers {
    /// Clear instances scratch buffer.
    #[cfg(feature = "wgpu")]
    pub(crate) clear_instances: Vec<GpuClearInstance>,
    /// Blend instances scratch buffer.
    pub(crate) blend_instances: Vec<GpuBlendInstance>,
    /// Copy instances scratch buffer.
    pub(crate) copy_instances: Vec<GpuCopyInstance>,
}

/// Per-instance data for clearing a rectangle in a render target.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuClearInstance {
    /// Atlas-space rectangle origin.
    pub(crate) origin: [u32; 2],
    /// Width and height of the cleared rectangle.
    pub(crate) size: [u32; 2],
    /// Width and height of the target texture.
    pub(crate) target_size: [u32; 2],
    /// Premultiplied clear color.
    pub(crate) color: u32,
}

impl MemorySettings {
    pub(crate) fn normalize(&mut self, device_limits: &DeviceLimits) {
        let Self {
            image_atlas_config,
            layers_config,
        } = self;

        image_atlas_config.atlas_size = SizeU16::from(image_atlas_config.atlas_size)
            .clamp(1, device_limits.max_texture_dimension_2d)
            .into();

        image_atlas_config.initial_atlas_count = image_atlas_config
            .initial_atlas_count
            .min(image_atlas_config.max_atlases);

        layers_config.min_texture_size = layers_config
            .min_texture_size
            .clamp(1, device_limits.max_texture_dimension_2d)
            // In case the user erroneously provided a smaller max size than min size.
            .min(layers_config.max_texture_size);
        layers_config.max_texture_size = layers_config
            .max_texture_size
            .clamp(1, device_limits.max_texture_dimension_2d);
    }
}

impl LayersConfig {
    pub(crate) fn required_intermediate_texture_size(
        self,
        recorder: &CommandRecorder<RecordedDraw>,
    ) -> Result<SizeU16, IntermediateTextureError> {
        let min_size = self.min_texture_size;
        let max_size = self.max_texture_size;

        let checked_size = |width: u32, height: u32| {
            if width > u32::from(max_size.width()) || height > u32::from(max_size.height()) {
                return Err(IntermediateTextureError::TooLarge {
                    width,
                    height,
                    max_width: max_size.width(),
                    max_height: max_size.height(),
                });
            }

            Ok(SizeU16::from_wh(
                u16::try_from(width).unwrap(),
                u16::try_from(height).unwrap(),
            ))
        };

        let filter_padding = u32::from(FILTER_ATLAS_PADDING) * 2;
        let filter_size = if let Some(size) = recorder.largest_filter_layer_size {
            checked_size(
                u32::from(size.width()) + filter_padding,
                u32::from(size.height()) + filter_padding,
            )?
        } else {
            SizeU16::ZERO
        };

        let mut layer_size = recorder
            .largest_layer_size
            .unwrap_or(SizeU16::ZERO)
            .max(filter_size);

        // If we are blending into the root we will render the whole root into an
        // layer texture. Since we don't track the bbox of root draw commands, we need
        // to reserve space for the full size of the scene.
        if recorder.root_is_blend_target {
            layer_size = layer_size.max(recorder.scene_size);
        }

        Ok(checked_size(
            u32::from(layer_size.width()),
            u32::from(layer_size.height()),
        )?
        .max(min_size))
    }
}

/// Dimensions of the rendering target.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RenderSize {
    /// Width of the rendering target.
    pub width: u16,
    /// Height of the rendering target.
    pub height: u16,
}

/// Configuration for the GPU renderer.
#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct Config {
    /// Width of the rendering target.
    pub width: u32,
    /// Height of the rendering target.
    pub height: u32,
    /// Height of a strip in the rendering.
    pub strip_height: u32,
    /// Number of trailing zeros in `alphas_tex_width` (log2 of width).
    /// Pre-calculated on CPU since downlevel targets do not support `firstTrailingBit`.
    pub alphas_tex_width_bits: u32,
    /// Number of trailing zeros in the encoded paints texture width (log2 of width).
    /// Pre-calculated on CPU since downlevel targets do not support `firstTrailingBit`.
    pub encoded_paints_tex_width_bits: u32,
    /// A horizontal offset to apply to strips.
    pub strip_offset_x: i32,
    /// A vertical offset to apply to strips.
    pub strip_offset_y: i32,
    /// Whether to flip the y-component of the NDC position.
    ///
    /// When transpiling a shader with naga, it applies a y-flip transform when
    /// compiling to GLSL to account for the difference between the y-down
    /// coordinate of WebGPU and the y-up coordinate system of WebGL framebuffers.
    ///
    /// However, for the native WebGL backend, we want to _avoid_ this transform so that we
    /// can write directly into the user-provided framebuffer, but still ensure that the scene
    /// is correctly flipped. Otherwise, we would need to allocate another framebuffer,
    /// render into that and then pay the cost for flipping it.
    ///
    /// Therefore, we optionally negate the coordinates in NDC.
    /// (Note that naga provides a flag for disabling this behavior. However, the problem
    /// is that many parts of Vello GPU's code (such as slot textures) also assume a
    /// y-down coordinate system. Therefore, just disabling this flag causes complications in
    /// other places. From my experiments, it's much easier to enable the flag by default
    /// and just apply the second negation manually in case we render to the final output surface
    /// in the WebGL backend.
    pub negate_ndc: u32,
}

/// A GPU strip instance for rendering.
///
/// This struct corresponds to the `StripInstance` struct in the shader.
/// See the `StripInstance` documentation in `render.wesl` for detailed field descriptions.
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct GpuStrip {
    /// See `StripInstance::xy` documentation in `render.wesl`.
    pub x: u16,
    /// See `StripInstance::xy` documentation in `render.wesl`.
    pub y: u16,
    /// See `StripInstance::dense_width_or_rect_height` documentation in `render.wesl`.
    pub width: u16,
    /// See `StripInstance::dense_width_or_rect_height` documentation in `render.wesl`.
    pub dense_width_or_rect_height: u16,
    /// See `StripInstance::col_idx_or_rect_frac` documentation in `render.wesl`.
    pub col_idx_or_rect_frac: u32,
    /// See `StripInstance::payload` documentation in `render.wesl`.
    pub payload: u32,
    /// See `StripInstance::paint_and_rect_flag` documentation in `render.wesl`.
    pub paint_and_rect_flag: u32,
    /// Painter's-order index used to compute z-depth for early-z rejection in shader.
    /// In other words, the back-most draw has index 0 and every additional draw in front
    /// has an incrementing index.
    pub depth_index: u32,
}

#[cfg(all(target_arch = "wasm32", feature = "webgl", feature = "wgpu"))]
pub(crate) fn maybe_warn_about_webgl_feature_conflict() {
    use core::sync::atomic::{AtomicBool, Ordering};
    static HAS_WARNED: AtomicBool = AtomicBool::new(false);

    if !HAS_WARNED.swap(true, Ordering::Release)
        && wgpu::Backends::all().contains(wgpu::Backends::GL)
    {
        log::warn!(
            r#"Both WebGL and wgpu with the "webgl" feature are enabled.
For optimal performance and binary size on web targets, use only the dedicated WebGL renderer."#
        );
    }
}

#[cfg(all(
    any(feature = "webgl", feature = "wgpu"),
    not(all(target_arch = "wasm32", feature = "webgl", feature = "wgpu"))
))]
pub(crate) fn maybe_warn_about_webgl_feature_conflict() {}

#[cfg(test)]
mod tests {
    use super::DeviceLimits;
    use crate::scene::RecordedDraw;
    use crate::{IntermediateTextureError, LayersConfig, MemorySettings, SizeU16};
    use vello_common::multi_atlas::AtlasConfig;
    use vello_common::record::CommandRecorder;

    fn device_limits(max_texture_dimension_2d: u16) -> DeviceLimits {
        DeviceLimits {
            max_texture_dimension_2d,
        }
    }

    #[test]
    fn normalize_memory_settings_clamps_atlas_dimensions_to_backend_limits() {
        let mut settings = MemorySettings {
            image_atlas_config: AtlasConfig {
                initial_atlas_count: 8,
                max_atlases: 16,
                atlas_size: (8192, 2048),
                ..Default::default()
            },
            ..Default::default()
        };

        settings.normalize(&device_limits(4096));

        let config = settings.image_atlas_config;
        assert_eq!(config.initial_atlas_count, 8);
        assert_eq!(config.max_atlases, 16);
        assert_eq!(config.atlas_size, (4096, 2048));
    }

    #[test]
    fn resource_texture_dimension_is_capped_at_4k() {
        assert_eq!(device_limits(16384).resource_texture_dimension_2d(), 4096);
        assert_eq!(device_limits(2048).resource_texture_dimension_2d(), 2048);
    }

    #[test]
    fn normalize_memory_settings_allows_lazy_initial_atlas_allocation() {
        let mut settings = MemorySettings {
            image_atlas_config: AtlasConfig {
                initial_atlas_count: 0,
                atlas_size: (0, 0),
                ..Default::default()
            },
            ..Default::default()
        };

        settings.normalize(&device_limits(4096));

        let config = settings.image_atlas_config;
        assert_eq!(config.initial_atlas_count, 0);
        assert_eq!(config.max_atlases, 8);
        assert_eq!(config.atlas_size, (1, 1));
    }

    #[test]
    fn normalize_memory_settings_clamps_minimum_to_maximum() {
        let mut settings = MemorySettings {
            layers_config: LayersConfig {
                min_texture_size: SizeU16::from_wh(1024, 512),
                max_texture_size: SizeU16::from_wh(512, 4096),
                ..Default::default()
            },
            ..Default::default()
        };

        settings.normalize(&device_limits(8192));

        assert_eq!(
            settings.layers_config.min_texture_size,
            SizeU16::from_wh(512, 512)
        );
        assert_eq!(
            settings.layers_config.max_texture_size,
            SizeU16::from_wh(512, 4096)
        );
    }

    #[test]
    fn normalize_memory_settings_clamps_layer_sizes_to_backend_limit() {
        let mut settings = MemorySettings {
            layers_config: LayersConfig {
                min_texture_size: SizeU16::from_wh(1024, 512),
                max_texture_size: SizeU16::from_wh(2048, 4096),
                ..Default::default()
            },
            ..Default::default()
        };

        settings.normalize(&device_limits(768));

        assert_eq!(
            settings.layers_config.min_texture_size,
            SizeU16::from_wh(768, 512)
        );
        assert_eq!(settings.layers_config.max_texture_size, SizeU16::new(768));
    }

    #[test]
    fn required_intermediate_texture_size_rejects_oversized_layers() {
        let mut recorder = CommandRecorder::<RecordedDraw>::new(10, 10);
        recorder.largest_layer_size = Some(SizeU16::from_wh(513, 10));

        let config = LayersConfig {
            min_texture_size: SizeU16::new(1),
            max_texture_size: SizeU16::new(512),
            ..Default::default()
        };

        assert!(matches!(
            config.required_intermediate_texture_size(&recorder),
            Err(IntermediateTextureError::TooLarge {
                width: 513,
                height: 10,
                max_width: 512,
                max_height: 512,
            })
        ));
    }

    #[test]
    fn required_intermediate_texture_size_rejects_filter_padding_overflow() {
        let mut recorder = CommandRecorder::<RecordedDraw>::new(10, 10);
        recorder.largest_filter_layer_size = Some(SizeU16::from_wh(u16::MAX, 10));

        let config = LayersConfig {
            min_texture_size: SizeU16::new(1),
            max_texture_size: SizeU16::new(u16::MAX),
            ..Default::default()
        };

        let padding = u32::from(crate::filter::FILTER_ATLAS_PADDING) * 2;
        let error = config
            .required_intermediate_texture_size(&recorder)
            .unwrap_err();

        match error {
            IntermediateTextureError::TooLarge {
                width,
                height,
                max_width,
                max_height,
            } => {
                assert_eq!(width, u32::from(u16::MAX) + padding);
                assert_eq!(height, 10 + padding);
                assert_eq!(max_width, u16::MAX);
                assert_eq!(max_height, u16::MAX);
            }
            _ => panic!("expected TooLarge"),
        }
    }
}
