// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

/// Executes one measured sample of a benchmark.
///
/// Warmup, time-based sampling, and presentation live in the runner rather than in benchmark
/// definitions.
#[derive(Debug)]
pub struct Bencher {
    iterations: u64,
    elapsed_nanos: Option<f64>,
}

impl Bencher {
    pub(crate) fn new(iterations: u64) -> Self {
        Self {
            iterations,
            elapsed_nanos: None,
        }
    }

    /// Run `routine` for the iteration count selected by the harness.
    pub fn iter<R>(&mut self, mut routine: impl FnMut() -> R) {
        assert!(
            self.elapsed_nanos.is_none(),
            "a benchmark may only measure once"
        );
        self.elapsed_nanos = Some(measure(|| {
            for _ in 0..self.iterations {
                core::hint::black_box(routine());
            }
        }));
    }

    /// Run `setup` outside the measured region before passing each input to `routine`.
    /// `batch_size` limits how many inputs and outputs are held at once.
    pub fn iter_batched<I, R>(
        &mut self,
        mut setup: impl FnMut() -> I,
        mut routine: impl FnMut(I) -> R,
        batch_size: usize,
    ) {
        assert!(batch_size > 0, "batch size must be greater than zero");
        assert!(
            self.elapsed_nanos.is_none(),
            "a benchmark may only measure once"
        );
        let mut remaining = usize::try_from(self.iterations)
            .expect("benchmark iteration count does not fit in usize");
        let mut elapsed_nanos = 0.0;
        while remaining > 0 {
            let batch_len = remaining.min(batch_size);
            let inputs = (0..batch_len).map(|_| setup()).collect::<Vec<_>>();
            let mut outputs = Vec::with_capacity(batch_len);
            elapsed_nanos += measure(|| {
                outputs.extend(inputs.into_iter().map(&mut routine));
            });
            core::hint::black_box(outputs);
            remaining -= batch_len;
        }
        self.elapsed_nanos = Some(elapsed_nanos);
    }

    pub(crate) fn finish(self) -> f64 {
        self.elapsed_nanos
            .expect("benchmark function returned without calling Bencher::iter")
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn measure(routine: impl FnOnce()) -> f64 {
    let start = std::time::Instant::now();
    routine();
    start.elapsed().as_secs_f64() * 1_000_000_000.0
}

#[cfg(target_arch = "wasm32")]
fn measure(routine: impl FnOnce()) -> f64 {
    let start = web_now_millis();
    routine();
    (web_now_millis() - start) * 1_000_000.0
}

#[cfg(target_arch = "wasm32")]
fn web_now_millis() -> f64 {
    #[link(wasm_import_module = "vello_bench")]
    unsafe extern "C" {
        fn now() -> f64;
    }

    // SAFETY: `web/worker.js` provides this import with the declared `() -> f64` signature.
    unsafe { now() }
}
