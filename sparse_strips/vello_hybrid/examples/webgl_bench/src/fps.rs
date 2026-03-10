// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! FPS tracking with a rolling average over recent frames.

const RING_SIZE: usize = 60;

/// Tracks frame durations and computes rolling average FPS.
#[derive(Debug)]
pub(crate) struct FpsTracker {
    /// Ring buffer of frame durations in milliseconds.
    durations: [f64; RING_SIZE],
    /// Current write index into the ring buffer.
    index: usize,
    /// Number of samples collected so far (up to `RING_SIZE`).
    count: usize,
    /// Timestamp of the previous frame (ms), from `performance.now()`.
    last_time: f64,
}

impl FpsTracker {
    /// Create a new FPS tracker starting at the given timestamp.
    pub(crate) fn new(now: f64) -> Self {
        Self {
            durations: [0.0; RING_SIZE],
            index: 0,
            count: 0,
            last_time: now,
        }
    }

    /// Record a new frame at the given timestamp (ms). Returns `(avg_fps, avg_frame_time_ms)`.
    pub(crate) fn frame(&mut self, now: f64) -> (f64, f64) {
        let dt = now - self.last_time;
        self.last_time = now;

        self.durations[self.index] = dt;
        self.index = (self.index + 1) % RING_SIZE;
        if self.count < RING_SIZE {
            self.count += 1;
        }

        let sum: f64 = self.durations[..self.count].iter().sum();
        let avg_ms = sum / self.count as f64;
        let fps = if avg_ms > 0.0 { 1000.0 / avg_ms } else { 0.0 };
        (fps, avg_ms)
    }
}
