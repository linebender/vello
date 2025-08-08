// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Backend agnostic renderer module.

use core::fmt;

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
#[derive(Clone, Copy, Zeroable, Pod)]
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

// Constants must stay in sync with `render_strips.wgsl`.
const COLOR_SOURCE_PAYLOAD: u32 = 0;
const COLOR_SOURCE_SLOT: u32 = 1;
const COLOR_SOURCE_BLEND: u32 = 2;
const PAINT_TYPE_SOLID: u32 = 0;
const PAINT_TYPE_IMAGE: u32 = 1;

impl fmt::Debug for GpuStrip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let color_source = (self.paint >> 30) & 0x3;  // Changed to 2 bits for 3 source types
        
        let mut debug_struct = f.debug_struct("GpuStrip");
        
        debug_struct
            .field("x", &self.x)
            .field("y", &self.y)
            .field("width", &self.width)
            .field("dense_width", &self.dense_width)
            .field("col_idx", &self.col_idx);
        
        let paint_info = match color_source {
            COLOR_SOURCE_PAYLOAD => {
                let paint_type = (self.paint >> 28) & 0x3;  // Adjusted bit position
                if paint_type == PAINT_TYPE_SOLID {
                    format!("Solid(color_source=payload)")
                } else if paint_type == PAINT_TYPE_IMAGE {
                    let paint_tex_id = self.paint & 0x0FFFFFFF;  // Adjusted mask
                    format!("Image(color_source=payload, texture_id={})", paint_tex_id)
                } else {
                    format!("Unknown(color_source=payload, type={})", paint_type)
                }
            }
            COLOR_SOURCE_SLOT => {
                let opacity = self.paint & 0xFF;
                format!("Slot(color_source=slot, opacity={})", opacity)
            }
            COLOR_SOURCE_BLEND => {
                let dest_slot = (self.paint >> 16) & 0x3FFF;
                let mix = (self.paint >> 8) & 0xFF;
                let compose = self.paint & 0xFF;
                format!("Blend(dest_slot={}, mix={}, compose={})", dest_slot, mix, compose)
            }
            _ => format!("Unknown(color_source={})", color_source)
        };
        
        debug_struct.field("paint", &paint_info);
        
        // Decode payload based on paint configuration
        let payload_info = match color_source {
            COLOR_SOURCE_PAYLOAD => {
                let paint_type = (self.paint >> 28) & 0x3;
                if paint_type == PAINT_TYPE_SOLID {
                    let r = (self.payload >> 0) & 0xFF;
                    let g = (self.payload >> 8) & 0xFF;
                    let b = (self.payload >> 16) & 0xFF;
                    let a = (self.payload >> 24) & 0xFF;
                    format!("Color(r={}, g={}, b={}, a={})", r, g, b, a)
                } else if paint_type == PAINT_TYPE_IMAGE {
                    let x = self.payload & 0xFFFF;
                    let y = self.payload >> 16;
                    format!("ImageCoords(x={}, y={})", x, y)
                } else {
                    format!("Unknown(raw=0x{:08x})", self.payload)
                }
            }
            COLOR_SOURCE_SLOT => {
                format!("SlotIndex({})", self.payload)
            }
            COLOR_SOURCE_BLEND => {
                format!("SourceSlot({})", self.payload)
            }
            _ => format!("Unknown(raw=0x{:08x})", self.payload)
        };
        
        debug_struct.field("payload", &payload_info);
        debug_struct.finish()
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
