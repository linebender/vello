// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate defines the public API types, providing a stable interface for CPU and hybrid
//! CPU/GPU rendering implementations. It provides common interfaces and data structures used
//! across different implementations
//!
//! This crate defines the public API types for the Vello rendering.
//! It provides common interfaces and data structures used across different implementations, including CPU, GPU, and hybrid rendering backends.
//!
//! ## Features
//!
//! - Shared API types for Vello's rendering pipeline.
//! - Interfaces for render contexts and rendering options.
//! - Designed for compatibility across CPU and GPU implementations.
//!
//! ## Usage
//!
//! This crate is intended to be used by other Vello components and external consumers needing a stable API.

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
