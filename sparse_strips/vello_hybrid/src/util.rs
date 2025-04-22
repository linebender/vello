// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// This file is a modified version of the vello/src/util.rs file.

//! Simple helpers for managing wgpu state and surfaces.

use core::ops::RangeInclusive;

/// Represents dimension constraints for surfaces
#[derive(Debug)]
pub struct DimensionConstraints {
    /// The valid range for width, inclusive of min and max values
    pub width_range: RangeInclusive<f64>,
    /// The valid range for height, inclusive of min and max values
    pub height_range: RangeInclusive<f64>,
}

impl DimensionConstraints {
    /// Create new constraints with given min/max dimensions
    pub fn new(min_width: f64, min_height: f64, max_width: f64, max_height: f64) -> Self {
        Self {
            width_range: min_width..=max_width,
            height_range: min_height..=max_height,
        }
    }

    /// Calculate dimensions while preserving aspect ratio within constraints
    ///
    /// Some viewboxes could never fit inside this constraint. For example, if the constraint for both axes
    /// is 100.0..=2000.0, if `original_width` is `2.` and `original_height` is `1000.`, there is clearly
    /// no way for that to fit within the constraints.
    /// In these cases, this method clamps to within the ranges (respecting the constraints but losing the aspect ratio).
    pub fn calculate_dimensions(&self, original_width: f64, original_height: f64) -> (f64, f64) {
        // Ensure we have non-zero input dimensions
        let original_width = original_width.max(1.0);
        let original_height = original_height.max(1.0);

        let min_width = *self.width_range.start();
        let max_width = *self.width_range.end();
        let min_height = *self.height_range.start();
        let max_height = *self.height_range.end();

        let (width, height) = if original_width > max_width || original_height > max_height {
            // Scale down if dimensions exceed maximum limits
            let width_ratio = max_width / original_width;
            let height_ratio = max_height / original_height;
            let ratio = width_ratio.min(height_ratio);

            ((original_width * ratio), (original_height * ratio))
        } else if original_width < min_width || original_height < min_height {
            // Scale up if dimensions are below minimum limits
            let width_ratio = min_width / original_width;
            let height_ratio = min_height / original_height;
            let ratio = width_ratio.max(height_ratio);

            ((original_width * ratio), (original_height * ratio))
        } else {
            (original_width, original_height)
        };
        (
            width.clamp(min_width, max_width),
            height.clamp(min_height, max_height),
        )
    }

    /// Converts a floating point dimension to a u16.
    /// For the [default](DimensionConstraints::default) constraints, if the input value
    /// was returned from [`calculate_dimensions`](Self::calculate_dimensions), this function can't
    /// fail. But note that this does not necessarily apply for custom constraints.
    ///
    /// # Panics
    ///
    /// If `value` is negative or larger than [`u16::MAX`].
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Any truncation will be caught by the (saturating) casts into a wider type"
    )]
    #[track_caller]
    pub fn convert_dimension(value: f64) -> u16 {
        (value.ceil() as i32)
            .try_into()
            .expect("Dimensions are clamped into a reasonable range")
    }
}

impl Default for DimensionConstraints {
    /// Creates default constraints with reasonable values
    fn default() -> Self {
        Self {
            width_range: 100.0..=2000.0,
            height_range: 100.0..=2000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DimensionConstraints;

    #[test]
    fn calculate_dimensions_in_range() {
        let new = DimensionConstraints::new;
        for (test_name, constraints) in [
            ("Default", DimensionConstraints::default()),
            (
                "Max width different to max height",
                new(100., 100., 500., 1000.),
            ),
            ("Max width equal to min width", new(100., 100., 100., 1000.)),
            ("Very loose constraints", new(10., 10., 10_000., 10_000.)),
        ] {
            for [test_width, test_height] in [
                [100., 100.],
                [50., 200.],
                [10., 2_000.],
                [10_000., 10_000.],
                // Larger than `u16::MAX`
                [128_000., 128_000.],
            ] {
                let (width, height) = constraints.calculate_dimensions(test_width, test_height);
                assert!(
                    constraints.width_range.contains(&width),
                    "Constraints in {test_name} should have a width in the supported range.\n\
                    Got {width}x{height} from {test_width}x{test_height} in {constraints:?}"
                );
                assert!(constraints.height_range.contains(&height));
            }
        }
    }
}
