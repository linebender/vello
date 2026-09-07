// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::runner::{average, next_iteration_count, standard_deviation};
use super::{Registry, RunConfig, Selection};
use libloading::Library;
use std::io;
use std::path::Path;

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

/// Compare libraries built with the same benchmark definitions as `registry`.
pub fn compare_libraries(
    registry: &Registry,
    artifact_a: &Path,
    artifact_b: &Path,
    filter: &str,
    selection: Selection,
    config: RunConfig,
    mut report: impl FnMut(Comparison) -> io::Result<()>,
) -> io::Result<()> {
    if config.sample_count == 0 {
        return Err(invalid("sample count must be greater than zero"));
    }
    let case_count = u32::try_from(registry.cases().len())
        .map_err(|_| invalid("benchmark count does not fit in u32"))?;
    let a = Artifact::load(artifact_a, case_count)?;
    let b = Artifact::load(artifact_b, case_count)?;

    let target_sample_nanos = config.target_sample_nanos();

    for (index, case) in registry
        .cases()
        .iter()
        .enumerate()
        .filter(|(_, case)| selection.includes(case, filter))
    {
        let id = case.id().to_owned();
        let index = u32::try_from(index).expect("benchmark count fits in u32");
        let iterations_a = warm_up(&a, index, config.warmup_time, target_sample_nanos);
        let iterations_b = warm_up(&b, index, config.warmup_time, target_sample_nanos);

        let capacity = config.sample_count as usize;
        let mut times_a = Vec::with_capacity(capacity);
        let mut times_b = Vec::with_capacity(capacity);
        let mut ratios = Vec::with_capacity(capacity);
        for pair in 0..config.sample_count {
            let (elapsed_a, elapsed_b) = if pair % 4 == 0 || pair % 4 == 3 {
                (a.sample(index, iterations_a), b.sample(index, iterations_b))
            } else {
                let elapsed_b = b.sample(index, iterations_b);
                let elapsed_a = a.sample(index, iterations_a);
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
type RunSample = unsafe extern "C" fn(u32, u64) -> f64;

struct Artifact {
    _library: Library,
    run_sample: RunSample,
}

impl Artifact {
    fn load(path: &Path, expected_count: u32) -> io::Result<Self> {
        // SAFETY: the library is kept alive while its exported functions and data are used.
        let library = unsafe { Library::new(path) }.map_err(io::Error::other)?;
        let case_count: CaseCount = symbol(&library, b"vello_bench_case_count")?;
        let run_sample: RunSample = symbol(&library, b"vello_bench_run_sample")?;
        // SAFETY: the signature matches the export in `abi.rs`.
        if unsafe { case_count() } != expected_count {
            return Err(invalid(
                "comparison library has a different benchmark count",
            ));
        }
        Ok(Self {
            _library: library,
            run_sample,
        })
    }

    fn sample(&self, index: u32, iterations: u64) -> f64 {
        // SAFETY: the checked case count makes the controller's index valid for this library.
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
