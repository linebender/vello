// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::default::Default;
use std::ops::RangeInclusive;

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
    pub fn calculate_dimensions(&self, original_width: f64, original_height: f64) -> (f64, f64) {
        // Ensure we have non-zero input dimensions
        let original_width = original_width.max(1.0);
        let original_height = original_height.max(1.0);

        let min_width = *self.width_range.start();
        let max_width = *self.width_range.end();
        let min_height = *self.height_range.start();
        let max_height = *self.height_range.end();

        if original_width > max_width || original_height > max_height {
            // Scale down if dimensions exceed maximum limits
            let width_ratio = max_width / original_width;
            let height_ratio = max_height / original_height;
            let ratio = width_ratio.min(height_ratio);

            (
                (original_width * ratio).max(1.0),
                (original_height * ratio).max(1.0),
            )
        } else if original_width < min_width || original_height < min_height {
            // Scale up if dimensions are below minimum limits
            let width_ratio = min_width / original_width;
            let height_ratio = min_height / original_height;
            let ratio = width_ratio.max(height_ratio);

            (
                (original_width * ratio).max(1.0),
                (original_height * ratio).max(1.0),
            )
        } else {
            (original_width, original_height)
        }
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
