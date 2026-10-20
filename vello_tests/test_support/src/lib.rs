// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renderer-independent image comparison and snapshot support.

pub mod diff;
mod snapshot;

pub use snapshot::Snapshot;
