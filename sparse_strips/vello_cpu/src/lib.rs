// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate implements the CPU-based renderer for Vello.
//! It is optimized for running on lower-end GPUs or CPU-only environments, leveraging parallelism and SIMD acceleration where possible.
//!
//! This crate implements a CPU-based renderer, optimized for SIMD and multithreaded execution.
//! It is optimized for CPU-bound workloads and serves as a standalone renderer for systems
//! without GPU acceleration.
//!
//! ## Features
//!
//! - Fully CPU-based path rendering pipeline
//! - SIMD and multithreading optimizations
//! - Fine rasterization and antialiasing techniques
//!
//! ## Usage
//!
//! This crate serves as a standalone renderer or as a fallback when GPU resources are constrained.

#![expect(
    clippy::cast_possible_truncation,
    reason = "We cast u16s to u8 in various places where we know for sure that it's < 256"
)]

extern crate alloc;

mod render;

#[doc(hidden)]
/// This is an internal module, do not access directly.
pub mod fine;
mod util;

pub use render::RenderContext;
pub use vello_common::glyph::Glyph;
pub use vello_common::mask::Mask;
pub use vello_common::paint::{Image, Paint, PaintType};
pub use vello_common::pixmap::Pixmap;

/// The selected rendering mode.
#[derive(Copy, Clone, Debug, Default)]
pub enum RenderMode {
    /// Optimize speed (by performing calculations with u8/16).
    #[default]
    OptimizeSpeed,
    /// Optimize quality (by performing calculations with f32).
    OptimizeQuality,
}
