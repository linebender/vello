// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::BenchmarkCase;
use core::time::Duration;

/// Measurement settings used by every frontend.
#[derive(Debug, Clone, Copy)]
pub struct RunConfig {
    /// Target total execution time recorded for each benchmark.
    pub measurement_time: Duration,
    /// Number of measured samples.
    pub sample_count: u32,
    /// Approximate warmup duration.
    pub warmup_time: Duration,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            measurement_time: Duration::from_secs(1),
            sample_count: 20,
            warmup_time: Duration::from_millis(250),
        }
    }
}

impl RunConfig {
    pub(crate) fn target_sample_nanos(self) -> f64 {
        self.measurement_time.as_secs_f64() * 1_000_000_000.0 / f64::from(self.sample_count)
    }
}

/// Results for one benchmark case.
#[derive(Debug)]
pub struct RunReport {
    pub average_nanos_per_iteration: f64,
    pub standard_deviation_nanos_per_iteration: f64,
}

/// Time-based warmup and sampling engine.
#[derive(Debug)]
pub struct Runner {
    config: RunConfig,
}

impl Runner {
    pub fn new(config: RunConfig) -> Self {
        assert!(
            !config.measurement_time.is_zero(),
            "measurement time must be greater than zero"
        );
        assert!(
            config.sample_count > 0,
            "sample count must be greater than zero"
        );
        Self { config }
    }

    pub fn run(&self, case: &BenchmarkCase) -> RunReport {
        let target_sample_nanos = self.config.target_sample_nanos();
        let iterations = self.warm_up(case, target_sample_nanos);

        let mut nanos_per_iteration = Vec::with_capacity(self.config.sample_count as usize);
        for _ in 0..self.config.sample_count {
            let elapsed_nanos = case.sample(iterations);
            assert!(
                elapsed_nanos > 0.0,
                "benchmark timer could not measure a sample of {}",
                case.id()
            );
            nanos_per_iteration.push(elapsed_nanos / iterations as f64);
        }

        let average_nanos_per_iteration = average(&nanos_per_iteration);
        let standard_deviation_nanos_per_iteration = standard_deviation(&nanos_per_iteration);
        RunReport {
            average_nanos_per_iteration,
            standard_deviation_nanos_per_iteration,
        }
    }

    fn warm_up(&self, case: &BenchmarkCase, target_sample_nanos: f64) -> u64 {
        let target = self.config.warmup_time.as_secs_f64() * 1_000_000_000.0;
        let mut elapsed = 0.0;
        let mut iterations = 1;
        loop {
            let sample_nanos = case.sample(iterations);
            elapsed += sample_nanos.max(1.0);
            iterations = next_iteration_count(iterations, sample_nanos, target_sample_nanos);
            if elapsed >= target {
                break;
            }
        }
        iterations
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the estimated iteration count is rounded and clamped to the range of u64"
)]
pub(crate) fn next_iteration_count(
    current: u64,
    elapsed_nanos: f64,
    target_sample_nanos: f64,
) -> u64 {
    let scale = if elapsed_nanos > 0.0 {
        (target_sample_nanos / elapsed_nanos).clamp(0.01, 100.0)
    } else {
        10.0
    };
    ((current as f64 * scale).round().clamp(1.0, u64::MAX as f64)) as u64
}

pub(crate) fn average(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation. A single observation has no measured variation and reports zero.
pub(crate) fn standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = average(values);
    let squared_deviations = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    (squared_deviations / (values.len() - 1) as f64).sqrt()
}
