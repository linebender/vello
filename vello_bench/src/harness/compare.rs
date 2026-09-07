// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Worker protocol for paired native A/B measurements.

use super::runner::{average, next_iteration_count, standard_deviation};
use super::{Registry, RunConfig, Selection};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Summary of paired samples from two worker artifacts.
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

/// Serve benchmark requests over stdin/stdout.
pub fn worker_main(registry: &Registry) -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line == "QUIT" {
            break;
        }
        let (id, iterations) = line
            .split_once('\t')
            .ok_or_else(|| invalid("invalid sample request"))?;
        let iterations = iterations
            .parse::<u64>()
            .map_err(|_| invalid("invalid iteration count"))?;
        let case = registry
            .find(id)
            .ok_or_else(|| invalid("unknown benchmark id"))?;
        writeln!(stdout, "{}", case.sample(iterations))?;
        stdout.flush()?;
    }
    Ok(())
}

/// Compare matching cases from two already-built worker executables.
pub fn compare_workers(
    registry: &Registry,
    artifact_a: &Path,
    artifact_b: &Path,
    filter: &str,
    selection: Selection,
    config: RunConfig,
    mut report: impl FnMut(Comparison) -> io::Result<()>,
) -> io::Result<()> {
    let mut a = Worker::spawn(artifact_a)?;
    let mut b = Worker::spawn(artifact_b)?;
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
        let iterations_a = warm_up(&mut a, &id, config.warmup_time, target_sample_nanos)?;
        let iterations_b = warm_up(&mut b, &id, config.warmup_time, target_sample_nanos)?;

        let capacity = config.sample_count as usize;
        let mut times_a = Vec::with_capacity(capacity);
        let mut times_b = Vec::with_capacity(capacity);
        let mut ratios = Vec::with_capacity(capacity);
        for pair in 0..config.sample_count {
            let (elapsed_a, elapsed_b) = if pair % 4 == 0 || pair % 4 == 3 {
                (a.sample(&id, iterations_a)?, b.sample(&id, iterations_b)?)
            } else {
                let elapsed_b = b.sample(&id, iterations_b)?;
                let elapsed_a = a.sample(&id, iterations_a)?;
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
    worker: &mut Worker,
    id: &str,
    duration: core::time::Duration,
    target_sample_nanos: f64,
) -> io::Result<u64> {
    let target = duration.as_secs_f64() * 1_000_000_000.0;
    let mut elapsed = 0.0;
    let mut iterations = 1;
    loop {
        let sample_nanos = worker.sample(id, iterations)?;
        elapsed += sample_nanos.max(1.0);
        iterations = next_iteration_count(iterations, sample_nanos, target_sample_nanos);
        if elapsed >= target {
            break;
        }
    }
    Ok(iterations)
}

struct Worker {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl Worker {
    fn spawn(path: &Path) -> io::Result<Self> {
        let mut child = Command::new(path)
            .arg("worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let input = child.stdin.take().expect("worker stdin must be piped");
        let output = BufReader::new(child.stdout.take().expect("worker stdout must be piped"));
        Ok(Self {
            child,
            input,
            output,
        })
    }

    fn sample(&mut self, id: &str, iterations: u64) -> io::Result<f64> {
        self.request(&format!("{id}\t{iterations}"))?;
        self.response()?
            .parse()
            .map_err(|_| invalid("invalid elapsed time"))
    }

    fn request(&mut self, request: &str) -> io::Result<()> {
        writeln!(self.input, "{request}")?;
        self.input.flush()
    }

    fn response(&mut self) -> io::Result<String> {
        let mut response = String::new();
        if self.output.read_line(&mut response)? == 0 {
            return Err(invalid("worker exited unexpectedly"));
        }
        Ok(response.trim_end().to_owned())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.request("QUIT");
        let _ = self.child.wait();
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
