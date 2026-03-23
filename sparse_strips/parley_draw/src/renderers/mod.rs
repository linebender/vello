// Copyright 2025 the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared renderer orchestration logic.
//!
//! Backend-specific `GlyphRenderer` implementations live in the renderer crates
//! (`vello_cpu`, `vello_hybrid`). This module contains only the shared,
//! backend-agnostic parts.

pub mod vello_renderer;
