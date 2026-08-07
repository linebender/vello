// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Provides renderer backends which handle GPU resource management and executes draw operations.
//!
//! ## Renderer Backends
//!
//! - `wgpu` contains the default renderer backend, leveraging `wgpu`.
//! - `webgl` contains a WebGL2 backend if the `webgl` feature is active.

pub(crate) mod common;
#[cfg(feature = "webgl")]
mod webgl;
#[cfg(feature = "wgpu")]
mod wgpu;

pub use common::{Config, GpuStrip, RenderSize};

#[cfg(all(feature = "webgl", feature = "probe"))]
pub use vello_common::probe::{PROBE_ELEMENTS, Probe, ProbeFeature, ProbeResult, ProbeStatistics};
#[cfg(all(feature = "webgl", feature = "probe"))]
pub use webgl::probe::{WebGlPendingProbe, WebGlProbeError, WebGlProbeStatus};
#[cfg(feature = "webgl")]
pub use webgl::{
    AtlasTextureInfo, WebGlAtlasWriter, WebGlRenderer, WebGlTextureBindings,
    WebGlTextureWithDimensions,
};
#[cfg(feature = "wgpu")]
pub use wgpu::{AtlasWriter, RenderTargetConfig, Renderer, TextureBindings};
