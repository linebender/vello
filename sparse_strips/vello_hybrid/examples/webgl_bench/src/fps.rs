// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Frame timing with rolling averages for both wall-clock frame time and CPU render time.

const RING_SIZE: usize = 60;

/// Rolling average over a ring buffer of `f64` samples.
#[derive(Debug)]
struct RollingAvg {
    samples: [f64; RING_SIZE],
    index: usize,
    count: usize,
}

impl RollingAvg {
    fn new() -> Self {
        Self {
            samples: [0.0; RING_SIZE],
            index: 0,
            count: 0,
        }
    }

    fn push(&mut self, value: f64) {
        self.samples[self.index] = value;
        self.index = (self.index + 1) % RING_SIZE;
        if self.count < RING_SIZE {
            self.count += 1;
        }
    }

    fn avg(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.samples[..self.count].iter().sum::<f64>() / self.count as f64
    }
}

/// Tracks wall-clock frame time (vsync-limited, rolling average) and instantaneous CPU render time.
#[derive(Debug)]
pub(crate) struct FpsTracker {
    last_time: f64,
    frame_times: RollingAvg,
}

impl FpsTracker {
    /// Create a new tracker starting at the given timestamp (ms).
    pub(crate) fn new(now: f64) -> Self {
        Self {
            last_time: now,
            frame_times: RollingAvg::new(),
        }
    }

    /// Record a frame. `now` is the current `performance.now()` timestamp.
    ///
    /// Returns `(fps, avg_frame_time_ms)`.
    pub(crate) fn frame(&mut self, now: f64) -> (f64, f64) {
        let dt = now - self.last_time;
        self.last_time = now;

        self.frame_times.push(dt);

        let avg_frame = self.frame_times.avg();
        let fps = if avg_frame > 0.0 {
            1000.0 / avg_frame
        } else {
            0.0
        };
        (fps, avg_frame)
    }
}
