// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Backend agnostic renderer module.

#![allow(
    clippy::cast_possible_truncation,
    reason = "GPU paint structures have small, fixed sizes that fit in u32"
)]

use bytemuck::{Pod, Zeroable};

// GPU paint structure sizes in texels (1 texel = 16 bytes for RGBA32Uint texture format).
pub(crate) const GPU_ENCODED_IMAGE_SIZE_TEXELS: u32 = (size_of::<GpuEncodedImage>() / 16) as u32;
pub(crate) const GPU_LINEAR_GRADIENT_SIZE_TEXELS: u32 =
    (size_of::<GpuLinearGradient>() / 16) as u32;
pub(crate) const GPU_RADIAL_GRADIENT_SIZE_TEXELS: u32 =
    (size_of::<GpuRadialGradient>() / 16) as u32;
pub(crate) const GPU_SWEEP_GRADIENT_SIZE_TEXELS: u32 = (size_of::<GpuSweepGradient>() / 16) as u32;

/// Dimensions of the rendering target.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RenderSize {
    /// Width of the rendering target.
    pub width: u32,
    /// Height of the rendering target.
    pub height: u32,
}

/// Configuration for the GPU renderer.
#[repr(C)]
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
}

/// Represents a GPU strip for rendering.
///
/// This struct corresponds to the `StripInstance` struct in the shader.
/// See the `StripInstance` documentation in `render_strips.wgsl` for detailed field descriptions.
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct GpuStrip {
    /// See `StripInstance::xy` documentation in `render_strips.wgsl`.
    pub x: u16,
    /// See `StripInstance::xy` documentation in `render_strips.wgsl`.
    pub y: u16,
    /// See `StripInstance::widths` documentation in `render_strips.wgsl`.
    pub width: u16,
    /// See `StripInstance::widths` documentation in `render_strips.wgsl`.
    pub dense_width: u16,
    /// See `StripInstance::col_idx` documentation in `render_strips.wgsl`.
    pub col_idx: u32,
    /// See `StripInstance::payload` documentation in `render_strips.wgsl`.
    pub payload: u32,
    /// See `StripInstance::paint` documentation in `render_strips.wgsl`.
    pub paint: u32,
}

/// Different types of GPU encoded paints.
#[derive(Debug)]
pub(crate) enum GpuEncodedPaint {
    /// An encoded image.
    Image(GpuEncodedImage),
    /// An encoded linear gradient.
    LinearGradient(GpuLinearGradient),
    /// An encoded radial gradient.
    RadialGradient(GpuRadialGradient),
    /// An encoded sweep gradient.
    SweepGradient(GpuSweepGradient),
}

impl GpuEncodedPaint {
    /// Returns the byte representation of this paint.
    #[inline]
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Image(paint) => bytemuck::bytes_of(paint),
            Self::LinearGradient(paint) => bytemuck::bytes_of(paint),
            Self::RadialGradient(paint) => bytemuck::bytes_of(paint),
            Self::SweepGradient(paint) => bytemuck::bytes_of(paint),
        }
    }

    /// Serialize paint enums directly into the provided buffer. Returns the number of bytes written.
    pub(crate) fn serialize_to_buffer(paints: &[Self], buffer: &mut [u8]) {
        let mut offset = 0;
        for paint in paints {
            let paint_bytes = paint.as_bytes();
            let end_offset = offset + paint_bytes.len();
            buffer[offset..end_offset].copy_from_slice(paint_bytes);
            offset = end_offset;
        }
    }
}

/// GPU encoded image data.
/// Align to 16 bytes for `RGBA32Uint` alignment.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[allow(dead_code, reason = "Clippy fails when --no-default-features")]
pub(crate) struct GpuEncodedImage {
    /// Packed rendering quality, extend modes, and atlas index.
    /// Bits 6-13: `atlas_index` (8 bits, supports up to 256 atlases)
    /// Bits 4-5: `extend_y` (2 bits)
    /// Bits 2-3: `extend_x` (2 bits)  
    /// Bits 0-1: `quality` (2 bits)
    pub image_params: u32,
    /// Packed image width and height.
    pub image_size: u32,
    /// The offset of the image in the atlas texture in pixels.
    pub image_offset: u32,
    /// Transform matrix [a, b, c, d, tx, ty].
    pub transform: [f32; 6],
    /// Padding for 16-byte alignment.
    pub _padding: [u32; 3],
}

/// GPU encoded linear gradient data.
/// Align to 16 bytes for `RGBA32Uint` alignment.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[allow(dead_code, reason = "Clippy fails when --no-default-features")]
pub(crate) struct GpuLinearGradient {
    /// Packed texture width (bits 0-30) and extend mode (bit 31: 0=Pad, 1=Repeat).
    pub texture_width_and_extend_mode: u32,
    /// Start coordinate in the flat gradient texture.
    pub gradient_start: u32,
    /// Transform matrix [a, b, c, d, tx, ty].
    pub transform: [f32; 6],
}

/// GPU encoded radial gradient data.
/// Align to 16 bytes for `RGBA32Uint` alignment.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[allow(dead_code, reason = "Clippy fails when --no-default-features")]
pub(crate) struct GpuRadialGradient {
    /// Packed texture width (bits 0-30) and extend mode (bit 31: 0=Pad, 1=Repeat).
    pub texture_width_and_extend_mode: u32,
    /// Start coordinate in the flat gradient texture for dense packing.
    pub gradient_start: u32,
    /// Transform matrix [a, b, c, d, tx, ty].
    pub transform: [f32; 6],
    /// Packed kind (bits 0-1) and `f_is_swapped` (bit 2): 0=Radial, 1=Strip, 2=Focal; bit 2: swapped flag.
    pub kind_and_f_is_swapped: u32,
    /// Bias value for radial gradient calculation.
    pub bias: f32,
    /// Scale factor for radial gradient calculation.
    pub scale: f32,
    /// Focal point 0 parameter for radial gradient.
    pub fp0: f32,
    /// Focal point 1 parameter for radial gradient.
    pub fp1: f32,
    /// Focal radius 1 parameter for radial gradient.
    pub fr1: f32,
    /// Focal X coordinate for radial gradient.
    pub f_focal_x: f32,
    /// Scaled radius 0 squared parameter for radial gradient strip.
    pub scaled_r0_squared: f32,
}

/// GPU encoded sweep gradient data.
/// Align to 16 bytes for `RGBA32Uint` alignment.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[allow(dead_code, reason = "Clippy fails when --no-default-features")]
pub(crate) struct GpuSweepGradient {
    /// Packed texture width (bits 0-30) and extend mode (bit 31: 0=Pad, 1=Repeat).
    pub texture_width_and_extend_mode: u32,
    /// Start coordinate in the flat gradient texture for dense packing.
    pub gradient_start: u32,
    /// Transform matrix [a, b, c, d, tx, ty].
    pub transform: [f32; 6],
    /// Starting angle for sweep gradient.
    pub start_angle: f32,
    /// Inverse of angle delta for sweep gradient.
    pub inv_angle_delta: f32,
    /// Padding for 16-byte alignment.
    pub _padding: [u32; 2],
}

// Constants for packing extend_mode and texture_width.
const EXTEND_MODE_MASK: u32 = 1 << 30;
const TEXTURE_WIDTH_MASK: u32 = !EXTEND_MODE_MASK;

/// Pack `extend_mode` and `texture_width` into a single u32.
/// `extend_mode`: 0=Pad, 1=Repeat, 2=Reflect (stored in bits 30 & 31)
/// `texture_width`: stored in bits 0-29 (max value: 2^30-1)
#[inline(always)]
pub(crate) fn pack_texture_width_and_extend_mode(texture_width: u32, extend_mode: u32) -> u32 {
    debug_assert!(extend_mode <= 2, "extend_mode must be less or equal to 2");
    debug_assert!(
        texture_width <= TEXTURE_WIDTH_MASK,
        "texture_width {texture_width} exceeds maximum value {TEXTURE_WIDTH_MASK}"
    );
    (extend_mode << 30) | (texture_width & TEXTURE_WIDTH_MASK)
}

/// Pack radial gradient `kind` and `f_is_swapped` into a single u32.
/// `kind`: 0=Radial, 1=Strip, 2=Focal (stored in bits 0-1)
/// `f_is_swapped`: 0=false, 1=true (stored in bit 2)
#[inline(always)]
pub(crate) fn pack_radial_kind_and_swapped(kind: u32, f_is_swapped: u32) -> u32 {
    debug_assert!(kind <= 2, "kind must be 0, 1, or 2");
    debug_assert!(f_is_swapped <= 1, "f_is_swapped must be 0 or 1");
    (f_is_swapped << 2) | (kind & 0x3)
}

/// Pack image `width` and `height` into a single u32.
/// `width`: stored in bits 16-31 (upper 16 bits)
/// `height`: stored in bits 0-15 (lower 16 bits)
#[inline(always)]
pub(crate) fn pack_image_size(width: u16, height: u16) -> u32 {
    ((width as u32) << 16) | (height as u32)
}

/// Pack image offset coordinates `x` and `y` into a single u32.
/// `x`: stored in bits 16-31 (upper 16 bits)
/// `y`: stored in bits 0-15 (lower 16 bits)
#[inline(always)]
pub(crate) fn pack_image_offset(x: u16, y: u16) -> u32 {
    ((x as u32) << 16) | (y as u32)
}

/// Pack image `quality`, extend modes, and `atlas_index` into a single u32.
/// `atlas_index`: stored in bits 6-13 (8 bits, supports up to 256 atlases)
/// `extend_y`: stored in bits 4-5 (2 bits)
/// `extend_x`: stored in bits 2-3 (2 bits)
/// `quality`: stored in bits 0-1 (2 bits)
#[inline(always)]
pub(crate) fn pack_image_params(
    quality: u32,
    extend_x: u32,
    extend_y: u32,
    atlas_index: u32,
) -> u32 {
    debug_assert!(extend_x <= 3, "extend_x must be 0-3 (2 bits)");
    debug_assert!(extend_y <= 3, "extend_y must be 0-3 (2 bits)");
    debug_assert!(quality <= 3, "quality must be 0-3 (2 bits)");
    debug_assert!(atlas_index <= 255, "atlas_index must be 0-255 (8 bits)");
    (atlas_index << 6) | (extend_y << 4) | (extend_x << 2) | quality
}

#[cfg(all(target_arch = "wasm32", feature = "webgl", feature = "wgpu"))]
pub(crate) fn maybe_warn_about_webgl_feature_conflict() {
    use core::sync::atomic::{AtomicBool, Ordering};
    static HAS_WARNED: AtomicBool = AtomicBool::new(false);

    if !HAS_WARNED.swap(true, Ordering::Release)
        && wgpu::Backends::all().contains(wgpu::Backends::GL)
    {
        log::warn!(
            r#"Both WebGL and wgpu with the \"webgl\" feature are enabled.
For optimal performance and binary size on web targets, use only the dedicated WebGL renderer."#
        );
    }
}

#[cfg(all(
    any(all(target_arch = "wasm32", feature = "webgl"), feature = "wgpu"),
    not(all(target_arch = "wasm32", feature = "webgl", feature = "wgpu"))
))]
pub(crate) fn maybe_warn_about_webgl_feature_conflict() {}
