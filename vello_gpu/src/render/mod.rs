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

pub use common::{
    BiplanarLayout, ChromaSiting, ClearSettings, Config, GpuStrip, RenderSize, TargetInit,
    YuvFormat, YuvMatrix, YuvRange,
};

#[cfg(feature = "webgl")]
pub use webgl::{
    AtlasTextureInfo, IncompatibleContextReason, WebGlAtlasWriter, WebGlContextOperation,
    WebGlDataTransferOperation, WebGlError, WebGlExternalTextureBinding, WebGlOperation,
    WebGlRenderer, WebGlRendererInit, WebGlRendererInitStatus, WebGlResourceKind,
    WebGlShaderInterfaceOperation, WebGlShaderProgramOperation, WebGlShaderStage,
    WebGlTextureBindings, WebGlTextureWithDimensions,
};
#[cfg(all(feature = "webgl", feature = "probe"))]
pub use webgl::{
    WebGlProbeOperation,
    probe::{WebGlPendingProbe, WebGlProbeError, WebGlProbeStatus},
};
#[cfg(feature = "wgpu")]
pub use wgpu::{
    AtlasWriter, ExternalTextureBinding, RenderTargetConfig, Renderer, TextureBindings,
};
