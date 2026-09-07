// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::Bencher;
use core::fmt;

pub struct BenchmarkCase {
    id: String,
    run: Box<dyn Fn(&mut Bencher)>,
    extended: bool,
    non_simd: bool,
    f32: bool,
}

impl fmt::Debug for BenchmarkCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BenchmarkCase")
            .field("id", &self.id)
            .field("extended", &self.extended)
            .field("non_simd", &self.non_simd)
            .field("f32", &self.f32)
            .finish_non_exhaustive()
    }
}

impl BenchmarkCase {
    /// Stable identifier used for filtering and matching A/B artifacts.
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn is_extended(&self) -> bool {
        self.extended
    }

    pub fn is_non_simd(&self) -> bool {
        self.non_simd
    }

    pub fn is_f32(&self) -> bool {
        self.f32
    }

    pub(crate) fn sample(&self, iterations: u64) -> f64 {
        let mut bencher = Bencher::new(iterations);
        (self.run)(&mut bencher);
        bencher.finish()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Selection {
    pub extended: bool,
    pub non_simd: bool,
    pub f32: bool,
}

impl Selection {
    pub fn includes(&self, case: &BenchmarkCase, filter: &str) -> bool {
        case.id().contains(filter)
            && (self.extended || !case.is_extended())
            && (self.non_simd || !case.is_non_simd())
            && (self.f32 || !case.is_f32())
    }
}

#[derive(Debug, Default)]
pub struct Registry {
    cases: Vec<BenchmarkCase>,
    registering_extended: bool,
    registering_non_simd: bool,
    registering_f32: bool,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, id: impl Into<String>, run: impl Fn(&mut Bencher) + 'static) {
        self.cases.push(BenchmarkCase {
            id: id.into(),
            run: Box::new(run),
            extended: self.registering_extended,
            non_simd: self.registering_non_simd,
            f32: self.registering_f32,
        });
    }

    pub(crate) fn extended(&mut self, register: impl FnOnce(&mut Self)) {
        let previous = self.registering_extended;
        self.registering_extended = true;
        register(self);
        self.registering_extended = previous;
    }

    pub(crate) fn non_simd(&mut self, register: impl FnOnce(&mut Self)) {
        let previous = self.registering_non_simd;
        self.registering_non_simd = true;
        register(self);
        self.registering_non_simd = previous;
    }

    pub(crate) fn f32(&mut self, register: impl FnOnce(&mut Self)) {
        let previous = self.registering_f32;
        self.registering_f32 = true;
        register(self);
        self.registering_f32 = previous;
    }

    pub fn finish(&mut self) {
        self.cases.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        for pair in self.cases.windows(2) {
            assert_ne!(pair[0].id, pair[1].id, "duplicate benchmark identifier");
        }
    }

    pub fn cases(&self) -> &[BenchmarkCase] {
        &self.cases
    }

    pub fn find(&self, id: &str) -> Option<&BenchmarkCase> {
        self.cases
            .binary_search_by(|case| case.id.as_str().cmp(id))
            .ok()
            .map(|index| &self.cases[index])
    }
}
