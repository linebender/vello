// Copyright 2026 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Debug and diagnostic utilities for renderer backends.
//!
//! Contains conditionally-compiled helpers for saving atlas pages to PNG,
//! querying cache statistics, and visualizing glyph bounds.

#[cfg(feature = "vello_cpu")]
mod vello_cpu_debug;
