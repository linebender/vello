// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Minimal xorshift64 pseudo-random number generator.

use vello_common::peniko::Color;

/// A simple xorshift64 PRNG.
#[derive(Debug, Clone)]
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new PRNG with the given seed. Must be non-zero.
    pub(crate) fn new(seed: u64) -> Self {
        debug_assert!(seed != 0, "seed must be non-zero");
        Self { state: seed }
    }

    /// Return the next random `u64`.
    fn next_u64(&mut self) -> u64 {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
        s
    }

    /// Return a random `f64` in `[0, 1)`.
    pub(crate) fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    /// Return a random `u8`.
    pub(crate) fn u8(&mut self) -> u8 {
        self.next_u64() as u8
    }

    /// Return a random `Color` with the given alpha.
    pub(crate) fn color(&mut self, alpha: u8) -> Color {
        Color::from_rgba8(self.u8(), self.u8(), self.u8(), alpha)
    }
}
