// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared support for testing the Vello CPU and GPU renderers.

#![allow(missing_docs, reason = "test infrastructure is not a public API")]
#![allow(
    missing_debug_implementations,
    reason = "renderer harnesses contain backend types that are not uniformly debuggable"
)]

pub mod diff;
pub mod renderer;
