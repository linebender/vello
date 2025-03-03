// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate contains core data structures and utilities shared across crates. It includes
//! foundational types for path geometry, tiling, and other common operations used in both CPU and
//! hybrid CPU/GPU rendering.

mod footprint;

pub mod flatten;
pub mod strip;
pub mod tile;
