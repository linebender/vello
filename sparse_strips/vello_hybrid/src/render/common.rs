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

// Constants matching the shader
const COLOR_SOURCE_PAYLOAD: u32 = 0;
const COLOR_SOURCE_SLOT: u32 = 1;
const COLOR_SOURCE_BLEND: u32 = 2;

const PAINT_TYPE_SOLID: u32 = 0;
const PAINT_TYPE_IMAGE: u32 = 1;

impl fmt::Debug for GpuStrip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let color_source = (self.paint >> 30) & 0x3;
        
        let mut debug = f.debug_struct("GpuStrip");
        
        // Basic position and dimensions
        debug.field("position", &format!("({}, {} maybe slot: {})", self.x, self.y, self.y / 4));
        debug.field("widths", &format!("{}x{}", self.width, self.dense_width));
        
        // Alpha column info if sparse strip
        if self.dense_width > 0 {
            debug.field("alpha_col", &self.col_idx);
        }
        
        // Decode paint and payload based on color source
        match color_source {
            COLOR_SOURCE_PAYLOAD => {
                let paint_type = (self.paint >> 28) & 0x3;
                match paint_type {
                    PAINT_TYPE_SOLID => {
                        let a = (self.payload >> 24) & 0xFF;
                        let b = (self.payload >> 16) & 0xFF;
                        let g = (self.payload >> 8) & 0xFF;
                        let r = self.payload & 0xFF;
                        debug.field("color", &format!("Solid(rgba({}, {}, {}, {}))", r, g, b, a));
                    }
                    PAINT_TYPE_IMAGE => {
                        let scene_x = self.payload & 0xFFFF;
                        let scene_y = self.payload >> 16;
                        let paint_tex_id = self.paint & 0x0FFFFFFF;
                        debug.field("color", &format!(
                            "Image(scene_pos=({}, {}), tex_id={})", 
                            scene_x, scene_y, paint_tex_id
                        ));
                    }
                    _ => {
                        debug.field("color", &format!("Unknown(paint_type={})", paint_type));
                    }
                }
            }
            COLOR_SOURCE_SLOT => {
                let slot = self.payload;
                let opacity = self.paint & 0xFF;
                let opacity_f = opacity as f32 / 255.0;
                debug.field("color", &format!("Slot({}, opacity={:.2})", slot, opacity_f));
            }
            COLOR_SOURCE_BLEND => {
                let src_slot = self.payload & 0xFFFF;
                let dest_slot = (self.payload >> 16) & 0xFFFF;
                let opacity = (self.paint >> 16) & 0xFF;
                let opacity_f = opacity as f32 / 255.0;
                let mix_mode = (self.paint >> 8) & 0xFF;
                let compose_mode = self.paint & 0xFF;
                
                let compose_name = match compose_mode {
                    0 => "Clear",
                    1 => "Copy",
                    2 => "Dest",
                    3 => "SrcOver",
                    4 => "DestOver",
                    5 => "SrcIn",
                    6 => "DestIn",
                    7 => "SrcOut",
                    8 => "DestOut",
                    9 => "SrcAtop",
                    10 => "DestAtop",
                    11 => "Xor",
                    12 => "Plus",
                    13 => "PlusLighter",
                    _ => "Unknown",
                };
                
                let mix_name = match mix_mode {
                    0 => "Normal",
                    _ => "Unknown",
                };
                
                debug.field("color", &format!(
                    "Blend(src={}, dest={}, opacity={:.2}, mix={}, compose={})",
                    src_slot, dest_slot, opacity_f, mix_name, compose_name
                ));
            }
            _ => {
                debug.field("color", &format!("Unknown(source={})", color_source));
            }
        }
        
        debug.finish()
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
