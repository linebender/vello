// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate contains core data structures and utilities shared across crates. It includes
//! foundational types for path geometry, tiling, and other common operations used in both CPU and
//! hybrid CPU/GPU rendering.

#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]
#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]
#![no_std]

extern crate alloc;

pub mod blurred_rounded_rect;
pub mod coarse;
pub mod encode;
pub mod flatten;
pub mod glyph;
pub mod math;
pub mod pico_svg;
pub mod strip;
pub mod tile;

pub use vello_api::*;

/// The selected rendering mode.
///
/// Note that this does not have any effect on GPU rendering and is solely for the CPU-based
/// renderer.
#[derive(Copy, Clone, Debug, Default)]
pub enum RenderMode {
    /// Optimize speed (by performing calculations with u8/16).
    #[default]
    OptimizeSpeed,
    /// Optimize quality (by performing calculations with f32).
    OptimizeQuality,
}
