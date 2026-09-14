// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod counter;
mod report;
pub mod tiger;

pub use counter::{AllocationStats, CountingAllocator, measure};
use report::{BenchmarkReport, BenchmarkResult};

const DEFAULT_MEASURED_FRAMES: usize = 1;

#[derive(Clone, Copy, Debug)]
pub struct AllocationBenchmark {
    name: &'static str,
    measure: fn(usize) -> AllocationStats,
    limits_per_frame: AllocationLimits,
}

impl AllocationBenchmark {
    pub const fn new(
        name: &'static str,
        measure: fn(usize) -> AllocationStats,
        limits_per_frame: AllocationLimits,
    ) -> Self {
        Self {
            name,
            measure,
            limits_per_frame,
        }
    }

    fn run(self, measured_frames: usize) -> BenchmarkResult {
        BenchmarkResult {
            measured_frames,
            name: self.name,
            stats: (self.measure)(measured_frames),
            limits: self.limits_per_frame.for_frames(measured_frames),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AllocationLimits {
    allocations: usize,
    reallocations: usize,
    allocated_bytes: usize,
    additional_peak_bytes: usize,
    retained_bytes: isize,
}

impl AllocationLimits {
    pub const ZERO: Self = Self::new(0, 0, 0, 0, 0);

    pub const fn new(
        allocations: usize,
        reallocations: usize,
        allocated_bytes: usize,
        additional_peak_bytes: usize,
        retained_bytes: isize,
    ) -> Self {
        Self {
            allocations,
            reallocations,
            allocated_bytes,
            additional_peak_bytes,
            retained_bytes,
        }
    }

    fn for_frames(self, frames: usize) -> Self {
        Self {
            allocations: self.allocations.checked_mul(frames).unwrap(),
            reallocations: self.reallocations.checked_mul(frames).unwrap(),
            allocated_bytes: self.allocated_bytes.checked_mul(frames).unwrap(),
            additional_peak_bytes: self.additional_peak_bytes,
            retained_bytes: self.retained_bytes,
        }
    }
}

pub fn run(benchmarks: &[AllocationBenchmark]) {
    let measured_frame_counts = measured_frame_counts();
    let mut results = Vec::with_capacity(measured_frame_counts.len() * benchmarks.len());
    for measured_frames in measured_frame_counts {
        results.extend(
            benchmarks
                .iter()
                .map(|benchmark| benchmark.run(measured_frames)),
        );
    }

    let report = BenchmarkReport::new(results);
    report.print();
    if report.has_failures() {
        std::process::exit(1);
    }
}

fn measured_frame_counts() -> Vec<usize> {
    let mut measured_frame_counts = Vec::new();
    let mut args = std::env::args().skip(1).peekable();

    while let Some(arg) = args.next() {
        if arg == "--frames" {
            let initial_count = measured_frame_counts.len();
            while args.peek().is_some_and(|arg| !arg.starts_with('-')) {
                parse_frame_counts(&args.next().unwrap(), &mut measured_frame_counts);
            }
            assert!(
                measured_frame_counts.len() > initial_count,
                "expected at least one frame count after `--frames`"
            );
        } else if let Some(value) = arg.strip_prefix("--frames=") {
            parse_frame_counts(value, &mut measured_frame_counts);
        }
    }

    if measured_frame_counts.is_empty() {
        measured_frame_counts.push(DEFAULT_MEASURED_FRAMES);
    }
    measured_frame_counts
}

fn parse_frame_counts(value: &str, measured_frame_counts: &mut Vec<usize>) {
    for value in value.split(',').filter(|value| !value.is_empty()) {
        let measured_frames = value
            .parse()
            .expect("frame count must be a positive integer");
        assert!(measured_frames > 0, "frame count must be greater than zero");
        if !measured_frame_counts.contains(&measured_frames) {
            measured_frame_counts.push(measured_frames);
        }
    }
}
