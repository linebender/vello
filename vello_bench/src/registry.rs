// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Benchmark registration for [Tango](https://github.com/bazhenov/tango).
//!
//! Tango runs the baseline and candidate builds alternately, each in its own process, so noise
//! that hits both sides cancels out. This keeps comparisons stable on noisy machines such as CI
//! runners.

use std::cell::RefCell;
use std::hint::black_box;
use std::rc::Rc;
use std::time::{Duration, Instant};

use tango_bench::{Benchmark, ErasedSampler, IntoBenchmarks};

/// Collects benchmarks for `tango_benchmarks!`.
#[derive(Default)]
#[expect(
    missing_debug_implementations,
    reason = "`tango_bench::Benchmark` does not implement `Debug`"
)]
pub struct Registry(Vec<Benchmark>);

impl Registry {
    /// Registers a benchmark.
    ///
    /// `routine` is called once per sample. It builds any state it needs, which is not measured,
    /// and then must call exactly one of the [`Bencher`] methods, which times the iterations.
    pub fn bench(&mut self, name: impl Into<String>, routine: impl FnMut(&mut Bencher) + 'static) {
        // Tango asks for a new sampler for every sample, and they all share the routine.
        let routine = Rc::new(RefCell::new(routine));
        self.0.push(tango_bench::benchmark_fn(name, move |_| {
            Box::new(Sampler(Rc::clone(&routine)))
        }));
    }
}

impl IntoBenchmarks for Registry {
    fn into_benchmarks(self) -> Vec<Benchmark> {
        self.0
    }
}

struct Sampler<F>(Rc<RefCell<F>>);

impl<F: FnMut(&mut Bencher)> ErasedSampler for Sampler<F> {
    fn measure(&mut self, iterations: usize) -> u64 {
        let mut bencher = Bencher {
            iterations,
            elapsed: None,
        };
        (self.0.borrow_mut())(&mut bencher);
        let elapsed = bencher
            .elapsed
            .expect("the benchmark routine must call a `Bencher` method");
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
    }
}

/// Times the iterations of a single sample.
#[derive(Debug)]
pub struct Bencher {
    iterations: usize,
    elapsed: Option<Duration>,
}

impl Bencher {
    /// Times `routine` over all iterations of the sample.
    pub fn iter<O>(&mut self, mut routine: impl FnMut() -> O) {
        let start = Instant::now();
        for _ in 0..self.iterations {
            black_box(routine());
        }
        self.elapsed = Some(start.elapsed());
    }

    /// Times `routine` with a fresh input from `setup` on every iteration.
    ///
    /// Neither `setup` nor dropping the output is timed. Each iteration is timed separately, so
    /// prefer [`Bencher::iter`] for routines that take less than a few microseconds.
    pub fn iter_batched<I, O>(
        &mut self,
        mut setup: impl FnMut() -> I,
        mut routine: impl FnMut(I) -> O,
    ) {
        let mut elapsed = Duration::ZERO;
        for _ in 0..self.iterations {
            let input = setup();
            let start = Instant::now();
            let output = black_box(routine(black_box(input)));
            elapsed += start.elapsed();
            drop(output);
        }
        self.elapsed = Some(elapsed);
    }

    /// Lets `routine` run and time the given number of iterations itself.
    pub fn iter_custom(&mut self, routine: impl FnOnce(u64) -> Duration) {
        self.elapsed = Some(routine(self.iterations as u64));
    }
}
