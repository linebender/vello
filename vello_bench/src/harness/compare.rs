// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Paired native A/B measurements from two dynamic libraries.

use super::runner::{average, next_iteration_count, standard_deviation};
use super::{Registry, RunConfig, Selection};
use libloading::Library;
use std::io;
use std::path::Path;

/// Summary of paired samples from two libraries.
#[derive(Debug)]
pub struct Comparison {
    pub id: String,
    pub average_a_nanos: f64,
    pub standard_deviation_a_nanos: f64,
    pub average_b_nanos: f64,
    pub standard_deviation_b_nanos: f64,
    pub average_ratio: f64,
    pub standard_deviation_ratio: f64,
}

/// Compare matching cases from two already-built dynamic libraries.
pub fn compare_libraries(
    registry: &Registry,
    artifact_a: &Path,
    artifact_b: &Path,
    filter: &str,
    selection: Selection,
    config: RunConfig,
    mut report: impl FnMut(Comparison) -> io::Result<()>,
) -> io::Result<()> {
    let a = Artifact::load(artifact_a)?;
    let b = Artifact::load(artifact_b)?;
    if config.sample_count == 0 {
        return Err(invalid("sample count must be greater than zero"));
    }

    let target_sample_nanos = config.target_sample_nanos();

    for case in registry
        .cases()
        .iter()
        .filter(|case| selection.includes(case, filter))
    {
        let id = case.id().to_owned();
        let index_a = a.index(&id)?;
        let index_b = b.index(&id)?;
        let iterations_a = warm_up(&a, index_a, config.warmup_time, target_sample_nanos);
        let iterations_b = warm_up(&b, index_b, config.warmup_time, target_sample_nanos);

        let capacity = config.sample_count as usize;
        let mut times_a = Vec::with_capacity(capacity);
        let mut times_b = Vec::with_capacity(capacity);
        let mut ratios = Vec::with_capacity(capacity);
        for pair in 0..config.sample_count {
            let (elapsed_a, elapsed_b) = if pair % 4 == 0 || pair % 4 == 3 {
                (
                    a.sample(index_a, iterations_a),
                    b.sample(index_b, iterations_b),
                )
            } else {
                let elapsed_b = b.sample(index_b, iterations_b);
                let elapsed_a = a.sample(index_a, iterations_a);
                (elapsed_a, elapsed_b)
            };
            if elapsed_a <= 0.0 || elapsed_b <= 0.0 {
                return Err(invalid(
                    "benchmark timer could not measure a benchmark sample",
                ));
            }
            let nanos_a = elapsed_a / iterations_a as f64;
            let nanos_b = elapsed_b / iterations_b as f64;
            times_a.push(nanos_a);
            times_b.push(nanos_b);
            ratios.push(nanos_b / nanos_a);
        }
        let standard_deviation_a_nanos = standard_deviation(&times_a);
        let standard_deviation_b_nanos = standard_deviation(&times_b);
        let standard_deviation_ratio = standard_deviation(&ratios);
        report(Comparison {
            id,
            average_a_nanos: average(&times_a),
            standard_deviation_a_nanos,
            average_b_nanos: average(&times_b),
            standard_deviation_b_nanos,
            average_ratio: average(&ratios),
            standard_deviation_ratio,
        })?;
    }
    Ok(())
}

fn warm_up(
    artifact: &Artifact,
    index: u32,
    duration: core::time::Duration,
    target_sample_nanos: f64,
) -> u64 {
    let target = duration.as_secs_f64() * 1_000_000_000.0;
    let mut elapsed = 0.0;
    let mut iterations = 1;
    loop {
        let sample_nanos = artifact.sample(index, iterations);
        elapsed += sample_nanos.max(1.0);
        iterations = next_iteration_count(iterations, sample_nanos, target_sample_nanos);
        if elapsed >= target {
            break;
        }
    }
    iterations
}

type CaseCount = unsafe extern "C" fn() -> u32;
type CaseNamePtr = unsafe extern "C" fn(u32) -> *const u8;
type CaseNameLen = unsafe extern "C" fn(u32) -> u32;
type RunSample = unsafe extern "C" fn(u32, u64) -> f64;

struct Artifact {
    _library: Library,
    cases: Vec<String>,
    run_sample: RunSample,
}

impl Artifact {
    fn load(path: &Path) -> io::Result<Self> {
        // SAFETY: the library is kept alive while its exported functions and data are used.
        let library = unsafe { Library::new(path) }.map_err(io::Error::other)?;
        let case_count: CaseCount = symbol(&library, b"vello_bench_case_count")?;
        let case_name_ptr: CaseNamePtr = symbol(&library, b"vello_bench_case_name_ptr")?;
        let case_name_len: CaseNameLen = symbol(&library, b"vello_bench_case_name_len")?;
        let run_sample: RunSample = symbol(&library, b"vello_bench_run_sample")?;
        let mut cases = Vec::new();
        // SAFETY: these signatures match the exports in `abi.rs`.
        for index in 0..unsafe { case_count() } {
            let ptr = unsafe { case_name_ptr(index) };
            let len = unsafe { case_name_len(index) } as usize;
            // SAFETY: the library owns each case name for the lifetime of its registry.
            let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
            cases.push(
                std::str::from_utf8(bytes)
                    .map_err(io::Error::other)?
                    .to_owned(),
            );
        }
        Ok(Self {
            _library: library,
            cases,
            run_sample,
        })
    }

    fn index(&self, id: &str) -> io::Result<u32> {
        self.cases
            .binary_search_by(|case| case.as_str().cmp(id))
            .map(|index| u32::try_from(index).expect("case count fits in u32"))
            .map_err(|_| invalid("benchmark case missing from comparison library"))
    }

    fn sample(&self, index: u32, iterations: u64) -> f64 {
        // SAFETY: index came from this library's case list and the signature matches `abi.rs`.
        unsafe { (self.run_sample)(index, iterations) }
    }
}

fn symbol<T: Copy>(library: &Library, name: &[u8]) -> io::Result<T> {
    // SAFETY: callers use the exact C signatures of the exports in `abi.rs`.
    unsafe { library.get::<T>(name).map(|symbol| *symbol) }.map_err(io::Error::other)
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
