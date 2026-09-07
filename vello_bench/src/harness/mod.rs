// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark harness shared by the CLI and browser.

mod bencher;
#[cfg(not(target_arch = "wasm32"))]
mod compare;
mod registry;
mod runner;

pub use bencher::Bencher;
#[cfg(not(target_arch = "wasm32"))]
pub use compare::{Comparison, compare_workers, worker_main};
pub use registry::{BenchmarkCase, Registry, Selection};
pub use runner::{RunConfig, RunReport, Runner};
