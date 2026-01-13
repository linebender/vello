// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// After you edit the crate's doc comment, run this command, then check README.md for any missing links
// cargo rdme --workspace-project=vello_common

//! This crate includes common geometry representations, tiling logic, and other fundamental components used by both [Vello CPU][vello_cpu] and Vello Hybrid.
//!
//! # Usage
//!
//! This crate should not be used on its own, and you should instead use one of the renderers which use it.
//! At the moment, only [Vello CPU][vello_cpu] is published, and you probably want to use that.
//!
//! We also develop [Vello](https://crates.io/crates/vello), which makes use of the GPU for 2D rendering and has higher performance than Vello CPU.
//! Vello CPU is being developed as part of work to address shortcomings in Vello.
//! Vello does not use this crate.
//!
//! # Features
//!
//! - `std` (enabled by default): Get floating point functions from the standard library
//!   (likely using your target's libc).
//! - `libm`: Use floating point implementations from [libm][].
//! - `png` (enabled by default): Allow loading [`Pixmap`][crate::pixmap::Pixmap]s from PNG images.
//!   Also required for rendering glyphs with an embedded PNG.
//!   Implies `std`.
//! - `text` (enabled by default): Enables glyph rendering (see the [`glyph`][] module).
//!
//! At least one of `std` and `libm` is required; `std` overrides `libm`.
//!
//! # Contents
//!
//! - Shared data structures for paths, tiles, and strips
//! - Geometry processing utilities
//! - Common logic for rendering stages
//!
//! This crate acts as a foundation for `vello_cpu` and `vello_hybrid`, providing essential components to minimize duplication.
//!
//! [vello_cpu]: https://crates.io/crates/vello_cpu
#![cfg_attr(feature = "libm", doc = "[libm]: libm")]
#![cfg_attr(not(feature = "libm"), doc = "[libm]: https://crates.io/crates/libm")]
// LINEBENDER LINT SET - lib.rs - v3
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]
#![no_std]

// Suppress the unused_crate_dependencies lint when both std and libm are specified.
#[cfg(all(feature = "std", feature = "libm"))]
use libm as _;

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod blurred_rounded_rect;
pub mod clip;
pub mod coarse;
#[cfg(feature = "text")]
pub mod colr;
pub mod encode;
pub mod filter_effects;
pub mod flatten;
pub(crate) mod flatten_simd;
#[cfg(feature = "text")]
pub mod glyph;
pub mod mask;
pub mod math;
pub mod paint;
#[doc(hidden)]
#[cfg(feature = "pico_svg")]
pub mod pico_svg;
pub mod pixmap;
pub mod recording;
pub mod render_graph;
pub mod simd;
pub mod strip;
pub mod strip_generator;
pub mod tile;
pub mod util;

pub use fearless_simd;
pub use peniko;
pub use peniko::color;
pub use peniko::kurbo;
