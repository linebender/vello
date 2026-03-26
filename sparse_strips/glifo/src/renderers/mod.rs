// Copyright 2025 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renderer backends for Vello CPU and Vello Hybrid (GPU).
//!
//! Shared orchestration logic lives in [`vello_renderer`]; backend-specific
//! code is in [`vello_cpu`] and [`vello_hybrid`].

#[cfg(any(feature = "vello_cpu", feature = "vello_hybrid"))]
pub mod vello_renderer;

#[cfg(feature = "vello_cpu")]
pub mod vello_cpu;

#[cfg(feature = "vello_hybrid")]
pub mod vello_hybrid;

#[doc(hidden)]
#[cfg(debug_assertions)]
pub mod debug;
