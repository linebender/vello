// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Library interface consumed by the browser and native comparison runner.
//!
//! Benchmark identifiers remain owned by a thread-local registry for the lifetime of the
//! library instance.

use crate::harness::Registry;
use std::cell::OnceCell;

#[cfg(target_arch = "wasm32")]
type Iterations = u32;
#[cfg(not(target_arch = "wasm32"))]
type Iterations = u64;

thread_local! {
    static REGISTRY: OnceCell<Registry> = const { OnceCell::new() };
}

fn with_registry<R>(f: impl FnOnce(&Registry) -> R) -> R {
    REGISTRY.with(|registry| f(registry.get_or_init(crate::registry)))
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_count() -> u32 {
    with_registry(|registry| {
        u32::try_from(registry.cases().len()).expect("benchmark count must fit in the library ABI")
    })
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_name_ptr(index: u32) -> *const u8 {
    with_registry(|registry| registry.cases()[index as usize].id().as_ptr())
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_name_len(index: u32) -> u32 {
    with_registry(|registry| {
        u32::try_from(registry.cases()[index as usize].id().len())
            .expect("benchmark identifier length must fit in the library ABI")
    })
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_is_extended(index: u32) -> u32 {
    with_registry(|registry| u32::from(registry.cases()[index as usize].is_extended()))
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_is_non_simd(index: u32) -> u32 {
    with_registry(|registry| u32::from(registry.cases()[index as usize].is_non_simd()))
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_is_f32(index: u32) -> u32 {
    with_registry(|registry| u32::from(registry.cases()[index as usize].is_f32()))
}

/// Measure exactly one sample and return elapsed nanoseconds.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_run_sample(index: u32, iterations: Iterations) -> f64 {
    #[cfg(target_arch = "wasm32")]
    let iterations = u64::from(iterations);
    with_registry(|registry| registry.cases()[index as usize].sample(iterations))
}
