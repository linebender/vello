// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// After you edit the crate's doc comment, run this command, then check README.md for any missing links
// cargo rdme --workspace-project=vello_hybrid

//! A hybrid CPU/GPU renderer for 2D vector graphics.
//!
//! This crate provides a rendering API that combines CPU and GPU operations for efficient
//! vector graphics processing.
//! The hybrid approach balances flexibility and performance by:
//!
//! - Using the CPU for path processing and initial geometry setup
//! - Leveraging the GPU for fast rendering and compositing
//! - Minimizing data transfer between CPU and GPU
//!
//! # Key Features
//!
//! - Efficient path rendering with CPU-side processing
//! - GPU-accelerated compositing and blending
//! - Support for both windowed and headless rendering
//!
//! # Feature Flags
//!
//! - `wgpu` (enabled by default): Enables the GPU rendering backend via wgpu and includes the required sparse shaders.
//! - `wgpu_default` (enabled by default): Enables wgpu with its default hardware backends (such as Vulkan, Metal, and DX12).
//! - `text` (enabled by default): Enables glyph rendering ([`Scene::glyph_run`]).
//! - `webgl`: Enables the WebGL rendering backend for browser support, using GLSL shaders for compatibility.
//!
//! If you need to customize the set of enabled wgpu features, disable this crate's default features then enable its `wgpu` feature.
//! You can then depend on wgpu directly, setting the specific features you require.
//! Don't forget to also disable wgpu's default features.
//!
//! # Architecture
//!
//! The renderer is split into several key components:
//!
//! - `Scene`: Manages the render context and path processing on the CPU
//! - `Renderer` or `WebGlRenderer`: Handles GPU resource management and executes draw operations
//!
//! See the individual module documentation for more details on usage and implementation.

#![no_std]

extern crate alloc;

pub(crate) mod blend;
pub(crate) mod copy;
pub(crate) mod filter;
mod gradient_cache;
mod paint;
mod rect;
mod render;
mod resources;
mod sampling;
mod scene;
#[cfg(any(feature = "webgl", feature = "wgpu"))]
mod schedule;
#[cfg(any(feature = "webgl", feature = "wgpu"))]
mod target;
#[cfg(feature = "text")]
mod text;

pub(crate) mod draw;
pub mod util;

#[cfg(feature = "wgpu")]
pub use render::{AtlasWriter, RenderTargetConfig, Renderer, TextureBindings};
pub use render::{Config, GpuStrip, RenderSize};
#[cfg(all(feature = "webgl", feature = "probe"))]
pub use render::{Probe, ProbeResult};
#[cfg(feature = "webgl")]
pub use render::{WebGlAtlasWriter, WebGlRenderer, WebGlTextureWithDimensions};
#[cfg(all(feature = "webgl", feature = "probe"))]
pub use render::{WebGlPendingProbe, WebGlProbeError, WebGlProbeStatus};
pub use resources::Resources;
pub use sampling::SampleRect;
pub use scene::{LayersConfig, MemorySettings, RenderSettings, Scene, TextureAllocationStrategy};
#[cfg(feature = "text")]
pub use text::{GlyphRunBuilder, HybridGlyphRunBackend};
pub use util::DimensionConstraints;
pub use vello_common::TextureId;
pub use vello_common::geometry::SizeU16;
pub use vello_common::multi_atlas::{AllocationStrategy, AtlasConfig, AtlasId};
pub use vello_common::pixmap::Pixmap;

use thiserror::Error;

/// Errors that can occur during rendering.
#[derive(Error, Debug, Clone)]
pub enum RenderError {
    /// An atlas allocation failed.
    #[error("Atlas allocation failed: {0}")]
    AtlasError(#[from] vello_common::multi_atlas::AtlasError),
    /// A draw referenced a [`TextureId`] that was not provided at render time.
    #[error("Missing texture binding for {0:?}")]
    MissingTextureBinding(TextureId),
    // TODO: Consider expanding `RenderError` to replace some `.unwrap` and `.expect`.
}

#[cfg(test)]
const _: () = if vello_common::tile::Tile::HEIGHT != 4 {
    panic!("`vello_hybrid` shaders currently require `Tile::HEIGHT` to be `4`");
};
