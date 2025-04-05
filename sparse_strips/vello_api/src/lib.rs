// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate defines the public API types, providing a stable interface for CPU and hybrid
//! CPU/GPU rendering implementations. It provides common interfaces and data structures used
//! across different implementations

#![forbid(unsafe_code)]
#![no_std]

pub use peniko;
pub use peniko::color;
pub use peniko::kurbo;
pub mod execute;
pub mod glyph;
pub mod paint;
