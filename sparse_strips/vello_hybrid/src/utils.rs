// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::ops::RangeInclusive;

/// Represents dimension constraints for surfaces
pub struct DimensionConstraints {
    pub width_range: RangeInclusive<u32>,
    pub height_range: RangeInclusive<u32>,
}

impl DimensionConstraints {
    /// Create new constraints with given min/max dimensions
    pub fn new(min_width: u32, min_height: u32, max_width: u32, max_height: u32) -> Self {
        Self {
            width_range: min_width..=max_width,
            height_range: min_height..=max_height,
        }
    }

    /// Default constraints
    pub fn default() -> Self {
        Self {
            width_range: 100..=2000,
            height_range: 100..=2000,
        }
    }

    /// Calculate dimensions while preserving aspect ratio within constraints
    pub fn calculate_dimensions(&self, original_width: u32, original_height: u32) -> (u32, u32) {
        // Ensure we have non-zero input dimensions
        let original_width = original_width.max(1);
        let original_height = original_height.max(1);

        let min_width = *self.width_range.start();
        let max_width = *self.width_range.end();
        let min_height = *self.height_range.start();
        let max_height = *self.height_range.end();

        if original_width > max_width || original_height > max_height {
            // Scale down if dimensions exceed maximum limits
            let width_ratio = max_width as f64 / original_width as f64;
            let height_ratio = max_height as f64 / original_height as f64;
            let ratio = width_ratio.min(height_ratio);

            (
                (original_width as f64 * ratio).max(1.0) as u32,
                (original_height as f64 * ratio).max(1.0) as u32,
            )
        } else if original_width < min_width || original_height < min_height {
            // Scale up if dimensions are below minimum limits
            let width_ratio = min_width as f64 / original_width as f64;
            let height_ratio = min_height as f64 / original_height as f64;
            let ratio = width_ratio.max(height_ratio);

            (
                (original_width as f64 * ratio).max(1.0) as u32,
                (original_height as f64 * ratio).max(1.0) as u32,
            )
        } else {
            (original_width, original_height)
        }
    }
}
