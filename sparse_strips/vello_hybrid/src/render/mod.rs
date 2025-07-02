// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Provides renderer backends which handle GPU resource management and executes draw operations.
//!
//! ## Renderer Backends
//!
//! - `wgpu` contains the default renderer backend, leveraging `wgpu`.
//! - `webgl` contains a WebGL2 backend specifically for `wasm32` if the `webgl` feature is active.

pub(crate) mod common;
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
mod webgl;
#[cfg(feature = "wgpu")]
mod wgpu;

pub use common::{Config, GpuStrip, RenderSize};

#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
pub use webgl::{WebGlAtlasWriter, WebGlRenderer, WebGlTextureWithDimensions};
#[cfg(feature = "wgpu")]
pub use wgpu::{AtlasWriter, RenderTargetConfig, Renderer};
