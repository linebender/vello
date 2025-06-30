// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Backend agnostic renderer module.

use bytemuck::{Pod, Zeroable};

/// Dimensions of the rendering target
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RenderSize {
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
}

/// Configuration for the GPU renderer
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct Config {
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
    /// Height of a strip in the rendering
    pub strip_height: u32,
    /// Number of trailing zeros in `alphas_tex_width` (log2 of width).
    /// Pre-calculated on CPU since downlevel targets do not support `firstTrailingBit`.
    pub alphas_tex_width_bits: u32,
}

/// Represents a GPU strip for rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct GpuStrip {
    /// X coordinate of the strip
    pub x: u16,
    /// Y coordinate of the strip
    pub y: u16,
    /// Width of the strip
    pub width: u16,
    /// Width of the portion where alpha blending should be applied.
    pub dense_width: u16,
    /// Column-index into the alpha texture where this strip's alpha values begin.
    ///
    /// There are [`Config::strip_height`] alpha values per column.
    pub col_idx: u32,
    /// Paint data
    pub rgba_or_slot: u32,
    /// Paint type: 0 = solid, 1 = alpha, 2 = image
    pub paint: u32,
}

/// Represents a GPU encoded image data for rendering
// Align to 16 bytes for RGBA32Uint alignment
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[allow(
    dead_code,
    reason = "Clippy fails when --no-default-features because webgl impl doesn't use this struct"
)]
pub(crate) struct GpuEncodedImage {
    // 1st 16 bytes
    /// The rendering quality of the image.
    pub quality: u32,
    /// The extends in the horizontal and vertical direction.
    pub extend_modes: [u32; 2],
    pub _padding0: u32,
    // 2nd 16 bytes
    /// The size of the image in pixels.
    pub image_size: [u32; 2],
    /// The offset of the image in pixels.
    pub image_offset: [u32; 2],
    // 3rd & 4th 16 bytes
    /// A transform to apply to the image.
    pub transform: [f32; 6],
    pub _padding1: [u32; 2],
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
