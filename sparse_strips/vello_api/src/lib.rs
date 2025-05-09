// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate defines the public API types used by both Vello CPU and Vello Hybrid.
//!
//! ## Usage
//!
//! This crate should not be used on its own, and you should instead use one of the renderers which use it.
//! At the moment, only [Vello CPU](crates.io/crates/vello_cpu) is published, and you probably want to use that.
//!
//! We also develop [Vello](crates.io/crates/vello), which makes use of the GPU for 2D rendering and has higher performance than Vello CPU.
//! Vello CPU is being developed as part of work to address shortcomings in Vello.
//! Vello does not use this crate.
//!
//! ## Features
//!
//! - Shared API types for Vello's rendering pipeline.
// - Interfaces for render contexts and rendering options.
// - Designed for compatibility across CPU and GPU implementations.
//!
//! ## Usage
//!
//! This crate is intended to be used by other Vello components.
//and external consumers needing a stable API.

#![forbid(unsafe_code)]
#![no_std]
extern crate alloc;

pub use peniko;
pub use peniko::color;
pub use peniko::kurbo;
pub mod execute;
pub mod glyph;
pub mod mask;
pub mod paint;
pub mod pixmap;
