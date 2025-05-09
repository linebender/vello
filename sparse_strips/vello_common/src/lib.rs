// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate contains core data structures and utilities shared across crates. It includes
//! foundational types for path geometry, tiling, and other common operations used in both CPU and
//! hybrid CPU/GPU rendering.
//!
//! This crate contains core data structures and utilities shared across the Vello rendering.
//! It includes common geometry representations, tiling logic, and other fundamental components used by both `vello_cpu` and `vello_hybrid`.
//!
//! ## Features
//!
//! - Shared data structures for paths, tiles, and strips
//! - Geometry processing utilities
//! - Common logic for rendering stages
//!
//! ## Usage
//!
//! This crate acts as a foundation for `vello_cpu` and `vello_hybrid`, providing essential components to minimize duplication.

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
pub mod colr;
pub mod encode;
pub mod flatten;
pub mod glyph;
pub mod math;
#[cfg(feature = "pico_svg")]
pub mod pico_svg;
pub mod strip;
pub mod tile;

pub use vello_api::*;
