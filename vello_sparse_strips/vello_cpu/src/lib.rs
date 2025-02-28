// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate implements a CPU-based renderer, optimized for SIMD and multithreaded execution.
//! It is optimized for CPU-bound workloads and serves as a standalone renderer for systems
//! without GPU acceleration.

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
