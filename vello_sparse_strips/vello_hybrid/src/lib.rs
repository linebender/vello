// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate implements a hybrid CPU/GPU renderer. It offloads key rendering tasks to the GPU
//! while retaining CPU-driven control, enabling optimized performance on various hardware
//! configurations.

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
