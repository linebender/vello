// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! # Vello Hybrid
//!
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
//! ## Key Features
//!
//! - Efficient path rendering with CPU-side processing
//! - GPU-accelerated compositing and blending
//! - Support for both windowed and headless rendering
//!
//! ## Architecture
//!
//! The renderer is split into several key components:
//!
//! - `Scene`: Manages the render context and path processing on the CPU
//! - `Renderer` or `WebGlRenderer`: Handles GPU resource management and executes draw operations
//! - `Scheduler`: Manages and schedules draw operations on the renderer.
//!
//! See the individual module documentation for more details on usage and implementation.

#![no_std]

extern crate alloc;

mod gradient_cache;
mod image_cache;
mod render;
mod scene;
#[cfg(any(all(target_arch = "wasm32", feature = "webgl"), feature = "wgpu"))]
mod schedule;
pub mod util;
#[cfg(feature = "wgpu")]
pub use render::{AtlasWriter, RenderTargetConfig, Renderer};
pub use render::{Config, GpuStrip, RenderSize};
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
pub use render::{WebGlAtlasWriter, WebGlRenderer, WebGlTextureWithDimensions};
pub use scene::{RenderSettings, Scene};
pub use util::DimensionConstraints;
pub use vello_common::pixmap::Pixmap;

use thiserror::Error;

/// Errors that can occur during rendering.
#[derive(Error, Debug)]
pub enum RenderError {
    /// No slots available for rendering.
    ///
    /// This error is likely to occur if a scene has an extreme number of nested layers
    /// (clipping, blending, masks, or opacity layers).
    ///
    /// TODO: Consider supporting more than a single column of slots in slot textures.
    #[error("No slots available for rendering")]
    SlotsExhausted,
    // TODO: Consider expanding `RenderError` to replace some `.unwrap` and `.expect`.
}

#[cfg(test)]
const _: () = if vello_common::tile::Tile::HEIGHT != 4 {
    panic!("`vello_hybrid` shaders currently require `Tile::HEIGHT` to be `4`");
};
