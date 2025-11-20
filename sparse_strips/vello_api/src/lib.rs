// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A stub crate.
//!
//! Planned to provide an abstraction between different renderers in the future.
//! This abstraction is not yet designed.

#![forbid(unsafe_code)]
#![no_std]

extern crate alloc;

mod design;
mod download;
mod painter;
mod renderer;

pub mod baseline;
pub mod dynamic;
pub mod prepared;
pub mod recording;
pub mod texture;

pub use self::download::DownloadId;
pub use self::painter::{PaintScene, SceneOptions};
pub use self::renderer::Renderer;
