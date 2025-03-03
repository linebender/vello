// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate defines the public API types, providing a stable interface for CPU and hybrid
//! CPU/GPU rendering implementations. It provides common interfaces and data structures used
//! across different implementations

#![forbid(unsafe_code)]

pub use peniko::*;
pub mod execute;
pub mod paint;
pub mod strip;
pub mod tile;
pub mod flatten;

