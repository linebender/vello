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

/// Represents a GPU encoded image data for rendering
// Align to 16 bytes for RGBA32Uint alignment
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[allow(dead_code, reason = "Clippy fails when --no-default-features")]
pub(crate) struct GpuEncodedImage {
    // 1st 16 bytes
    /// The rendering quality of the image.
    pub quality_and_extend_modes: u32,
    /// The extends in the horizontal and vertical direction.
    /// The size of the image in pixels.
    pub image_size: u32,
    /// The offset of the image in the atlas texture in pixels.
    pub image_offset: u32,
    pub _padding1: u32,
    // 2nd & 3rd 16 bytes
    /// A transform to apply to the image.
    pub transform: [f32; 6],
    /// Padding to align to 64 bytes (16-byte aligned)
    pub _padding2: [u32; 2],
}

/// Represents a GPU blend command for wide tile blending operations.
///
/// This struct corresponds to the `BlendCommand` struct in the blend_wide_tile.wgsl shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct GpuBlendCommand {
    /// [x, y] packed as u16's - coordinates of the top left of the source wide tile
    pub xy_src: u32,
    /// [x, y] packed as u16's - coordinates of the top left of the destination wide tile  
    pub xy_dst: u32,
    /// Bits 0-7: opacity
    /// Bits 8-11: compose
    /// Bits 12-15: mix
    /// Bits 16: source texture (0 = slots of ix=0, 1 = slots of ix=1)
    /// Bits 17-18: dest texture (0 = slots of ix=0, 1 = slots of ix=1, 2 = final target)
    /// Bits 19-26: blend slot index
    pub payload: u32,
}

/// Represents a GPU copy command for copying slots between textures.
///
/// This struct corresponds to the `CopyCommand` struct in the copy_slot.wgsl shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct GpuCopyCommand {
    /// [x, y] packed as u16's - coordinates of the top left of the target wide tile
    pub xy_target: u32,
    /// Slot index to identify the pixel position to sample from
    pub slot_ix: u32,
}

/// Configuration for the blend wide tile operations
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct BlendConfig {
    /// Width of a wide tile (matching `WideTile::WIDTH`).
    pub wide_tile_width: u32,
    /// Height of a wide tile (matching `WideTile::HEIGHT`).
    pub wide_tile_height: u32,
    /// Height of the slot texture.
    pub slot_texture_height: u32,
    /// Height of the final target texture.
    pub final_target_height: u32,
    /// Height of the blend texture.
    pub blend_texture_height: u32,
    /// Padding for 16-byte alignment
    pub _padding: [u32; 3],
}

/// Configuration for the copy slot operations
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CopyConfig {
    /// Width of a wide tile (matching `WideTile::WIDTH`).
    pub wide_tile_width: u32,
    /// Height of a wide tile (matching `WideTile::HEIGHT`).
    pub wide_tile_height: u32,
    /// Height of the slot texture (source).
    pub slot_texture_height: u32,
    /// Width of the target texture (destination).
    pub target_texture_width: u32,
    /// Height of the target texture (destination).
    pub target_texture_height: u32,
    /// Padding for 16-byte alignment
    pub _padding: [u32; 3],
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
