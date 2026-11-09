// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A backend-independent way to declare benchmarks.
//!
//! Benchmarks written against [`Registry`] run under Criterion through [`CriterionRegistry`],
//! and can be run by other harnesses through further implementations.

use criterion::Criterion;

/// Collects benchmarks for a specific harness.
pub trait Registry {
    /// Registers the benchmark `group/name`.
    ///
    /// `setup` builds the state that `routine` reuses across iterations, and is not measured.
    /// Harnesses may call it more than once, for example once per sample, so it must not
    /// depend on being called only once. Only `routine` is timed.
    fn bench<S: 'static>(
        &mut self,
        group: &str,
        name: &str,
        setup: impl Fn() -> S + 'static,
        routine: impl Fn(&mut S) + 'static,
    );
}

/// Registers benchmarks with Criterion, using the same benchmark IDs as a hand-written
/// `benchmark_group(group).bench_function(name, ..)`, so existing baselines stay comparable.
#[expect(
    missing_debug_implementations,
    reason = "`Criterion` does not implement `Debug`"
)]
pub struct CriterionRegistry<'a>(pub &'a mut Criterion);

impl Registry for CriterionRegistry<'_> {
    fn bench<S: 'static>(
        &mut self,
        group: &str,
        name: &str,
        setup: impl Fn() -> S + 'static,
        routine: impl Fn(&mut S) + 'static,
    ) {
        let mut g = self.0.benchmark_group(group);
        g.bench_function(name, |b| {
            let mut state = setup();
            b.iter(|| routine(&mut state));
        });
        g.finish();
    }
}
