// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Command-line interface for Vello benchmarks.

use std::env;
use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;
use vello_bench::harness::{RunConfig, Runner, Selection, compare_workers, worker_main};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("vello-bench: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("list") => list(args.collect())?,
        Some("run") => run_benchmarks(args.collect())?,
        Some("worker") => worker_main(&vello_bench::registry())?,
        Some("compare") => compare(args.collect())?,
        _ => print_help(),
    }
    Ok(())
}

fn list(args: Vec<String>) -> io::Result<()> {
    let mut filter = "";
    let mut selection = Selection::default();
    for argument in &args {
        match argument.as_str() {
            argument if include_flag(&mut selection, argument) => {}
            argument if !argument.starts_with('-') && filter.is_empty() => filter = argument,
            argument => return Err(unknown(argument)),
        }
    }

    for case in vello_bench::registry()
        .cases()
        .iter()
        .filter(|case| selection.includes(case, filter))
    {
        println!("{}", case.id());
    }
    Ok(())
}

fn run_benchmarks(args: Vec<String>) -> io::Result<()> {
    let (filter, selection, config) = parse_bench_args(&args)?;

    let registry = vello_bench::registry();
    let runner = Runner::new(config);
    println!(
        "Warming up each benchmark for {}, then measuring {} samples over approximately {}.\n",
        format_duration(config.warmup_time.as_secs_f64() * 1_000_000_000.0),
        config.sample_count,
        format_duration(config.measurement_time.as_secs_f64() * 1_000_000_000.0),
    );
    for case in registry
        .cases()
        .iter()
        .filter(|case| selection.includes(case, filter))
    {
        let report = runner.run(case);
        println!(
            "{:<64} {} (± {:.2}%)",
            case.id(),
            format_duration(report.average_nanos_per_iteration),
            relative_deviation(
                report.standard_deviation_nanos_per_iteration,
                report.average_nanos_per_iteration,
            ),
        );
    }
    Ok(())
}

fn compare(args: Vec<String>) -> io::Result<()> {
    let artifact_a = args.first().ok_or_else(|| missing("artifact A"))?;
    let artifact_b = args.get(1).ok_or_else(|| missing("artifact B"))?;
    let (filter, selection, config) = parse_bench_args(&args[2..])?;
    let registry = vello_bench::registry();
    println!(
        "Warming up each artifact for {}, then measuring {} samples over approximately {} per artifact.\n",
        format_duration(config.warmup_time.as_secs_f64() * 1_000_000_000.0),
        config.sample_count,
        format_duration(config.measurement_time.as_secs_f64() * 1_000_000_000.0),
    );
    let use_color = io::stdout().is_terminal() && env::var_os("NO_COLOR").is_none();
    compare_workers(
        &registry,
        Path::new(artifact_a),
        Path::new(artifact_b),
        filter,
        selection,
        config,
        |result| {
            println!("{}", result.id);
            println!(
                "  A {} (± {:.2}%)",
                format_duration(result.average_a_nanos),
                relative_deviation(result.standard_deviation_a_nanos, result.average_a_nanos,),
            );
            println!(
                "  B {} (± {:.2}%)",
                format_duration(result.average_b_nanos),
                relative_deviation(result.standard_deviation_b_nanos, result.average_b_nanos,),
            );
            let change = (result.average_ratio - 1.0) * 100.0;
            let color = if !use_color {
                ""
            } else if change >= 5.0 {
                "\x1b[31m"
            } else if change <= -5.0 {
                "\x1b[32m"
            } else {
                ""
            };
            let reset = if color.is_empty() { "" } else { "\x1b[0m" };
            println!(
                "  {color}change {change:+.2}% (± {:.2} pp){reset}",
                result.standard_deviation_ratio * 100.0,
            );
            io::stdout().flush()
        },
    )
}

fn parse_bench_args(args: &[String]) -> io::Result<(&str, Selection, RunConfig)> {
    let mut filter = "";
    let mut selection = Selection::default();
    let mut config = RunConfig::default();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--measurement-ms" => {
                config.measurement_time = measurement_time(args.get(index + 1))?;
                index += 2;
            }
            "--warmup-ms" => {
                config.warmup_time = duration(args.get(index + 1), "--warmup-ms")?;
                index += 2;
            }
            "--samples" => {
                config.sample_count = sample_count(args.get(index + 1))?;
                index += 2;
            }
            argument if include_flag(&mut selection, argument) => {
                index += 1;
            }
            argument if !argument.starts_with('-') && filter.is_empty() => {
                filter = argument;
                index += 1;
            }
            argument => {
                return Err(unknown(argument));
            }
        }
    }
    Ok((filter, selection, config))
}

fn include_flag(selection: &mut Selection, argument: &str) -> bool {
    match argument {
        "--extended" => selection.extended = true,
        "--non-simd" => selection.non_simd = true,
        "--f32" => selection.f32 = true,
        _ => return false,
    }
    true
}

fn missing(argument: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("missing required {argument} argument"),
    )
}

fn unknown(argument: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("unknown argument: {argument}"),
    )
}

fn duration(value: Option<&String>, option: &str) -> io::Result<Duration> {
    let value = value.ok_or_else(|| missing(option))?;
    let millis = value.parse::<u64>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid value for {option}"),
        )
    })?;
    Ok(Duration::from_millis(millis))
}

fn measurement_time(value: Option<&String>) -> io::Result<Duration> {
    let duration = duration(value, "--measurement-ms")?;
    if duration.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--measurement-ms must be greater than zero",
        ));
    }
    Ok(duration)
}

fn sample_count(value: Option<&String>) -> io::Result<u32> {
    let value = value.ok_or_else(|| missing("--samples"))?;
    let count = value
        .parse::<u32>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid value for --samples"))?;
    if count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--samples must be greater than zero",
        ));
    }
    Ok(count)
}

fn format_duration(nanos: f64) -> String {
    if nanos < 1_000.0 {
        format!("{nanos:.2} ns")
    } else if nanos < 1_000_000.0 {
        format!("{:.2} µs", nanos / 1_000.0)
    } else if nanos < 1_000_000_000.0 {
        format!("{:.2} ms", nanos / 1_000_000.0)
    } else {
        format!("{:.2} s", nanos / 1_000_000_000.0)
    }
}

fn relative_deviation(standard_deviation: f64, average: f64) -> f64 {
    standard_deviation / average * 100.0
}

fn print_help() {
    println!(
        "\
vello-bench list [FILTER] [--extended] [--non-simd] [--f32]
vello-bench run [FILTER] [--extended] [--non-simd] [--f32] [--warmup-ms MILLIS] [--measurement-ms MILLIS] [--samples COUNT]
vello-bench compare PATH_A PATH_B [FILTER] [--extended] [--non-simd] [--f32] [--warmup-ms MILLIS] [--measurement-ms MILLIS] [--samples COUNT]
vello-bench worker"
    );
}
