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
//! - `Renderer`: Handles GPU resource management and rendering operations
//! - `RenderData`: Contains the processed geometry ready for GPU consumption
//!
//! See the individual module documentation for more details on usage and implementation.

#![no_std]

extern crate alloc;

mod render;
mod scene;
pub mod util;
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
mod webgl_render;
#[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
mod wgpu_render;

pub use render::{GpuStrip, RenderData, RenderSize};
pub use scene::Scene;
pub use util::DimensionConstraints;
pub use vello_common::pixmap::Pixmap;
#[cfg(not(all(target_arch = "wasm32", feature = "webgl")))]
pub use wgpu_render::{Config, RenderTargetConfig, Renderer};
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
pub use webgl_render::WebGLRenderer;

#[cfg(test)]
const _: () = if vello_common::tile::Tile::HEIGHT != 4 {
    panic!("`vello_hybrid` shaders currently require `Tile::HEIGHT` to be `4`");
};
