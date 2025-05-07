// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU rendering module for the sparse strips CPU/GPU rendering engine.
//!
//! This module provides the GPU-side implementation of the hybrid rendering system.
//! It handles:
//! - GPU resource management (buffers, textures, pipelines)
//! - Surface/window management and presentation
//! - Shader execution and rendering
//!
//! This module contains backend agnostic parts of the renderer. For backend specific code see the
//! [`wgpu_render`] or [`webgl_render`].
//!
//! [`wgpu_render`]: TODO
//! [`webgl_render`]: TODO

use alloc::vec::Vec;
use bytemuck::{Pod, Zeroable};

/// Dimensions of the rendering target
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RenderSize {
    /// Width of the rendering target
    pub width: u32,
    /// Height of the rendering target
    pub height: u32,
}

/// Contains the data needed for rendering
#[derive(Debug, Default)]
pub struct RenderData {
    /// GPU strips to be rendered
    pub strips: Vec<GpuStrip>,
    /// Alpha values used in rendering
    pub alphas: Vec<u8>,
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
    pub col: u32,
    /// RGBA color value
    pub rgba: u32,
}
