// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate includes common geometry representations, tiling logic, and other fundamental components used by both [Vello CPU][vello_cpu] and Vello Hybrid.
//!
//! ## Usage
//!
//! This crate should not be used on its own, and you should instead use one of the renderers which use it.
//! At the moment, only [Vello CPU][vello_cpu] is published, and you probably want to use that.
//!
//! We also develop [Vello](https://crates.io/crates/vello), which makes use of the GPU for 2D rendering and has higher performance than Vello CPU.
//! Vello CPU is being developed as part of work to address shortcomings in Vello.
//! Vello does not use this crate.
//!
//! ## Features
//!
//! - Shared data structures for paths, tiles, and strips
//! - Geometry processing utilities
//! - Common logic for rendering stages
//!
//! This crate acts as a foundation for `vello_cpu` and `vello_hybrid`, providing essential components to minimize duplication.
//!
//! [vello_cpu]: https://crates.io/crates/vello_cpu

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
