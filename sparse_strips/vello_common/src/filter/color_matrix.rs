// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The color matrix filter.

/// Matrix-based color transformation filter.
///
/// The matrix is stored as four rows of five values. Each row computes one output
/// channel (`R`, `G`, `B`, `A`) from the four input channels plus a constant offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorMatrix {
    /// The 4x5 color transformation matrix in row-major order.
    pub matrix: [f32; 20],
}

impl ColorMatrix {
    /// Create a new color matrix filter.
    pub fn new(matrix: [f32; 20]) -> Self {
        Self { matrix }
    }
}
