// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![warn(clippy::doc_markdown, clippy::semicolon_if_nothing_returned)]

mod cpu_dispatch;
mod cpu_shader;
mod render;
mod scene;
mod shaders;
#[cfg(feature = "wgpu")]
mod wgpu_engine;
#[cfg(feature = "wgpu")]
mod wgpu_renderer;
mod workflow;

/// Styling and composition primitives.
pub use peniko;
/// 2D geometry, with a focus on curves.
pub use peniko::kurbo;

#[doc(hidden)]
pub use skrifa;

pub mod glyph;

#[cfg(feature = "wgpu")]
pub mod util;

pub use render::Render;
pub use scene::{DrawGlyphs, Scene};
#[cfg(feature = "wgpu")]
pub use util::block_on_wgpu;

#[cfg(feature = "wgpu")]
pub use wgpu_renderer::*;

pub use shaders::FullShaders;
pub use workflow::{
    BufferProxy, Command, ImageFormat, ImageProxy, ResourceId, ResourceProxy, ShaderId, Workflow,
};

/// Temporary export, used in `with_winit` for stats
pub use vello_encoding::BumpAllocators;

/// Catch-all error type.
pub type Error = Box<dyn std::error::Error>;

/// Specialization of `Result` for our catch-all error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Parameters used in a single render that are configurable by the client.
pub struct RenderParams {
    /// The background color applied to the target. This value is only applicable to the full
    /// pipeline.
    pub base_color: peniko::Color,

    /// Dimensions of the rasterization target
    pub width: u32,
    pub height: u32,

    /// The anti-aliasing algorithm. The selected algorithm must have been initialized while
    /// constructing the `Renderer`.
    pub antialiasing_method: AaConfig,
}

/// Represents the antialiasing method to use during a render pass.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum AaConfig {
    Area,
    Msaa8,
    Msaa16,
}

/// Represents the set of antialiasing configurations to enable during pipeline creation.
pub struct AaSupport {
    pub area: bool,
    pub msaa8: bool,
    pub msaa16: bool,
}

impl AaSupport {
    pub fn all() -> Self {
        Self {
            area: true,
            msaa8: true,
            msaa16: true,
        }
    }

    pub fn area_only() -> Self {
        Self {
            area: true,
            msaa8: false,
            msaa16: false,
        }
    }
}
