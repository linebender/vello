// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod bencher;
#[cfg(not(target_arch = "wasm32"))]
mod compare;
mod registry;
mod runner;

pub use bencher::Bencher;
#[cfg(not(target_arch = "wasm32"))]
pub use compare::{Comparison, compare_libraries};
pub use registry::{BenchmarkCase, Registry, Selection};
pub use runner::{RunConfig, RunReport, Runner};
