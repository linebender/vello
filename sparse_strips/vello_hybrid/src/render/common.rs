// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Backend agnostic renderer module.

use bytemuck::{Pod, Zeroable};
use vello_common::tile::Tile;

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

// Constants used for bit packing, matching `render_strips.wgsl`
const COLOR_SOURCE_SLOT: u32 = 1;
const COLOR_SOURCE_BLEND: u32 = 2;

/// Helper for more semantically constructing `GpuStrip`s.
pub(crate) struct GpuStripBuilder {
    x: u16,
    y: u16,
    width: u16,
    dense_width: u16,
    col_idx: u32,
}

impl GpuStripBuilder {
    /// Position at canvas/scene coordinates.
    pub(crate) fn at_canvas(x: u16, y: u16, width: u16) -> Self {
        Self {
            x,
            y,
            width,
            dense_width: 0,
            col_idx: 0,
        }
    }

    /// Position within a slot.
    pub(crate) fn at_slot(slot_idx: usize, x_offset: u16, width: u16) -> Self {
        Self {
            x: x_offset,
            y: slot_idx as u16 * Tile::HEIGHT,
            width,
            dense_width: 0,
            col_idx: 0,
        }
    }

    /// Add sparse strip parameters.
    pub(crate) fn with_sparse(mut self, dense_width: u16, col_idx: u32) -> Self {
        self.dense_width = dense_width;
        self.col_idx = col_idx;
        self
    }

    /// Paint into strip.
    pub(crate) fn paint(self, payload: u32, paint: u32) -> GpuStrip {
        GpuStrip {
            x: self.x,
            y: self.y,
            width: self.width,
            dense_width: self.dense_width,
            col_idx: self.col_idx,
            payload,
            paint,
        }
    }

    /// Copy from slot.
    pub(crate) fn copy_from_slot(self, from_slot: usize, opacity: u8) -> GpuStrip {
        GpuStrip {
            x: self.x,
            y: self.y,
            width: self.width,
            dense_width: self.dense_width,
            col_idx: self.col_idx,
            payload: from_slot as u32,
            paint: (COLOR_SOURCE_SLOT << 30) | (opacity as u32),
        }
    }

    /// Blend two slots.
    pub(crate) fn blend(
        self,
        src_slot: usize,
        dest_slot: usize,
        opacity: u8,
        mix_mode: u8,
        compose_mode: u8,
    ) -> GpuStrip {
        GpuStrip {
            x: self.x,
            y: self.y,
            width: self.width,
            dense_width: self.dense_width,
            col_idx: self.col_idx,
            payload: (src_slot as u32) | ((dest_slot as u32) << 16),
            paint: (COLOR_SOURCE_BLEND << 30) 
                | ((opacity as u32) << 16)
                | ((mix_mode as u32) << 8)
                | (compose_mode as u32),
        }
    }
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
