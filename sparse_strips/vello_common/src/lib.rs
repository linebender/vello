// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate contains core data structures and utilities shared across crates. It includes
//! foundational types for path geometry, tiling, and other common operations used in both CPU and
//! hybrid CPU/GPU rendering.

#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]
#![allow(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions between f32 and f64."
)]
mod footprint;

pub mod flatten;
pub mod strip;
pub mod tile;
pub mod wide;

pub use vello_api::*;
