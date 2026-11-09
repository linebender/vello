// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Paired benchmarks for comparing a change against a baseline build with Tango.
//!
//! Only benchmarks written against `vello_bench::registry::Registry` can run here.

#![allow(missing_docs, reason = "Not needed for benchmarks")]

#[cfg(not(target_arch = "wasm32"))]
fn benchmarks() -> vello_bench::registry::TangoRegistry {
    let mut registry = vello_bench::registry::TangoRegistry::default();
    vello_bench::flatten::flatten(&mut registry);
    registry
}

#[cfg(not(target_arch = "wasm32"))]
tango_bench::tango_benchmarks!(benchmarks());

// Tango runs benchmarks in child processes, which wasm32 does not support.
#[cfg(target_arch = "wasm32")]
fn main() {}
