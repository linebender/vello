// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::io::IsTerminal;

use super::{AllocationLimits, AllocationStats};

pub(super) struct BenchmarkReport {
    results: Vec<BenchmarkResult>,
    failed_workloads: Vec<FailedWorkload>,
}

impl BenchmarkReport {
    pub(super) fn new(results: Vec<BenchmarkResult>) -> Self {
        let failed_workloads = results
            .iter()
            .filter_map(|result| {
                let failures = result.limit_failures();
                (!failures.is_empty()).then_some(FailedWorkload {
                    measured_frames: result.measured_frames,
                    name: result.name,
                    failures,
                })
            })
            .collect();

        Self {
            results,
            failed_workloads,
        }
    }

    pub(super) fn print(&self) {
        print_results(&self.results);

        if self.failed_workloads.is_empty() {
            println!(
                "\nAll allocation limits passed ({}/{} workloads).",
                self.results.len(),
                self.results.len()
            );
            return;
        }

        eprintln!(
            "\nAllocation limits failed for {}/{} workloads:",
            self.failed_workloads.len(),
            self.results.len()
        );
        for workload in &self.failed_workloads {
            eprintln!("  {} ({} frames)", workload.name, workload.measured_frames);
            for failure in &workload.failures {
                eprintln!(
                    "    {}: {} (limit: {})",
                    failure.metric, failure.actual, failure.limit
                );
            }
        }
    }

    pub(super) fn has_failures(&self) -> bool {
        !self.failed_workloads.is_empty()
    }
}

fn print_results(results: &[BenchmarkResult]) {
    let use_color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    println!("CPU allocation benchmarks (averages per frame)\n");
    println!("  Allocations/frame   New heap blocks allocated per average frame.");
    println!("  Reallocations/frame Existing heap blocks resized per average frame.");
    println!("  Allocated bytes     New bytes requested per average frame.");
    println!("  Peak bytes          Maximum additional live bytes during the complete run.");
    println!("  Retained bytes      Net live-byte increase after the complete run.\n");
    println!(
        "{:>6} {:<27} {:>20} {:>18} {:>25} {:>23} {:>16}",
        "Frames",
        "Workload",
        "Allocations/frame",
        "Reallocations/frame",
        "Allocated bytes/frame",
        "Peak bytes",
        "Retained bytes",
    );
    println!("{}", "-".repeat(142));
    let mut previous_frame_count = None;
    for result in results {
        if previous_frame_count.is_some_and(|count| count != result.measured_frames) {
            println!();
        }
        previous_frame_count = Some(result.measured_frames);

        println!(
            "{:>6} {:<27} {} {} {} {} {}",
            result.measured_frames,
            result.name,
            result.format_average_metric(
                result.stats.allocations,
                result.limits.allocations,
                20,
                use_color,
            ),
            result.format_average_metric(
                result.stats.reallocations,
                result.limits.reallocations,
                18,
                use_color,
            ),
            result.format_average_metric(
                result.stats.allocated_bytes,
                result.limits.allocated_bytes,
                25,
                use_color,
            ),
            format_metric(
                result.stats.additional_peak_bytes as i128,
                result.limits.additional_peak_bytes as i128,
                23,
                use_color,
            ),
            format_metric(
                result.stats.retained_bytes as i128,
                result.limits.retained_bytes as i128,
                16,
                use_color,
            ),
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BenchmarkResult {
    pub(super) measured_frames: usize,
    pub(super) name: &'static str,
    pub(super) stats: AllocationStats,
    pub(super) limits: AllocationLimits,
}

impl BenchmarkResult {
    fn format_average_metric(
        self,
        value: usize,
        baseline: usize,
        width: usize,
        use_color: bool,
    ) -> String {
        let measured_frames = self.measured_frames as f64;
        let average = value as f64 / measured_frames;
        let baseline_average = baseline as f64 / measured_frames;
        let delta = value as i128 - baseline as i128;
        let value = if delta == 0 {
            format_average(average)
        } else if baseline == 0 {
            format!(
                "{} ({:+}, new)",
                format_average(average),
                delta as f64 / measured_frames
            )
        } else {
            let delta_average = delta as f64 / measured_frames;
            let percentage = (average - baseline_average) / baseline_average.abs() * 100.0;
            format!(
                "{} ({delta_average:+.2}, {percentage:+.1}%)",
                format_average(average)
            )
        };
        let value = format!("{value:>width$}");

        color_metric(value, delta, use_color)
    }

    fn limit_failures(self) -> Vec<LimitFailure> {
        let metrics = [
            (
                "allocations",
                self.stats.allocations as i128,
                self.limits.allocations as i128,
            ),
            (
                "reallocations",
                self.stats.reallocations as i128,
                self.limits.reallocations as i128,
            ),
            (
                "allocated bytes",
                self.stats.allocated_bytes as i128,
                self.limits.allocated_bytes as i128,
            ),
            (
                "additional peak bytes",
                self.stats.additional_peak_bytes as i128,
                self.limits.additional_peak_bytes as i128,
            ),
            (
                "retained bytes",
                self.stats.retained_bytes as i128,
                self.limits.retained_bytes as i128,
            ),
        ];

        metrics
            .into_iter()
            .filter_map(|(metric, actual, limit)| {
                (actual > limit).then_some(LimitFailure {
                    metric,
                    actual,
                    limit,
                })
            })
            .collect()
    }
}

fn format_average(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        format!("{value:.2}")
    }
}

fn format_metric(value: i128, baseline: i128, width: usize, use_color: bool) -> String {
    let delta = value - baseline;
    let value = if delta != 0 {
        if baseline == 0 {
            format!("{value} ({delta:+}, new)")
        } else {
            let percentage = delta as f64 / baseline.abs() as f64 * 100.0;
            format!("{value} ({delta:+}, {percentage:+.1}%)")
        }
    } else {
        value.to_string()
    };
    let value = format!("{value:>width$}");

    color_metric(value, delta, use_color)
}

fn color_metric(value: String, delta: i128, use_color: bool) -> String {
    if !use_color || delta == 0 {
        value
    } else if delta < 0 {
        format!("\u{1b}[32m{value}\u{1b}[0m")
    } else {
        format!("\u{1b}[31m{value}\u{1b}[0m")
    }
}

#[derive(Debug)]
struct FailedWorkload {
    measured_frames: usize,
    name: &'static str,
    failures: Vec<LimitFailure>,
}

#[derive(Clone, Copy, Debug)]
struct LimitFailure {
    metric: &'static str,
    actual: i128,
    limit: i128,
}
