// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! WebAssembly interface consumed by `web/worker.js`.
//!
//! Benchmark identifiers remain owned by a thread-local registry for the lifetime of the
//! WebAssembly instance.

use crate::harness::Registry;
use std::cell::OnceCell;

thread_local! {
    static REGISTRY: OnceCell<Registry> = const { OnceCell::new() };
}

fn with_registry<R>(f: impl FnOnce(&Registry) -> R) -> R {
    REGISTRY.with(|registry| f(registry.get_or_init(crate::registry)))
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_count() -> u32 {
    with_registry(|registry| {
        u32::try_from(registry.cases().len()).expect("benchmark count must fit in the Wasm ABI")
    })
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_name_ptr(index: u32) -> *const u8 {
    with_registry(|registry| registry.cases()[index as usize].id().as_ptr())
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_name_len(index: u32) -> u32 {
    with_registry(|registry| {
        u32::try_from(registry.cases()[index as usize].id().len())
            .expect("benchmark identifier length must fit in the Wasm ABI")
    })
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_is_extended(index: u32) -> u32 {
    with_registry(|registry| u32::from(registry.cases()[index as usize].is_extended()))
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_is_non_simd(index: u32) -> u32 {
    with_registry(|registry| u32::from(registry.cases()[index as usize].is_non_simd()))
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_case_is_f32(index: u32) -> u32 {
    with_registry(|registry| u32::from(registry.cases()[index as usize].is_f32()))
}

/// Measure exactly one sample and return elapsed nanoseconds.
#[unsafe(no_mangle)]
pub(crate) extern "C" fn vello_bench_run_sample(index: u32, iterations: u32) -> f64 {
    with_registry(|registry| registry.cases()[index as usize].sample(u64::from(iterations)))
}
