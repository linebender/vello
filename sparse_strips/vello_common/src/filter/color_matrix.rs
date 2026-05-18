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

    /// Return true if this matrix can be applied directly to premultiplied colors.
    ///
    /// A premultiplied-compatible matrix preserves alpha, does not read alpha
    /// from the RGB rows, and has no RGB offsets. For this subset, renderers can
    /// apply the RGB rows directly to premultiplied RGB and clamp the result to
    /// the unchanged alpha channel.
    pub fn is_premul_compatible(&self) -> bool {
        self.matrix[3] == 0.0
            && self.matrix[4] == 0.0
            && self.matrix[8] == 0.0
            && self.matrix[9] == 0.0
            && self.matrix[13] == 0.0
            && self.matrix[14] == 0.0
            && self.matrix[15] == 0.0
            && self.matrix[16] == 0.0
            && self.matrix[17] == 0.0
            && self.matrix[18] == 1.0
            && self.matrix[19] == 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_effects::matrices;

    #[test]
    fn premul_compatible_matrices_are_rgb_only_and_alpha_preserving() {
        assert!(ColorMatrix::new(matrices::GRAYSCALE).is_premul_compatible());
        assert!(ColorMatrix::new(matrices::SEPIA).is_premul_compatible());
        assert!(!ColorMatrix::new(matrices::ALPHA_TO_BLACK).is_premul_compatible());
    }

    #[test]
    fn premul_compatible_matrix_rejects_rgb_offsets_and_alpha_changes() {
        let mut offset_matrix = matrices::IDENTITY;
        offset_matrix[4] = 0.25;
        assert!(!ColorMatrix::new(offset_matrix).is_premul_compatible());

        let mut opacity_matrix = matrices::IDENTITY;
        opacity_matrix[18] = 0.5;
        assert!(!ColorMatrix::new(opacity_matrix).is_premul_compatible());
    }
}
