// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// This file is a modified version of the vello/src/util.rs file.

//! A number of utility helper methods.

use alloc::vec::Vec;
use core::ops::{Range, RangeInclusive};
use vello_common::util::Clear;

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

pub(crate) fn pack_u16_pair(x: u16, y: u16) -> u32 {
    u32::from(x) | (u32::from(y) << 16)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "opacity is clamped to the normalized u8 range before packing"
)]
pub(crate) fn pack_opacity(opacity: f32) -> u8 {
    (opacity.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Coalesced ranges selecting values from a shared buffer.
#[derive(Debug, Default, Clone)]
pub(crate) struct Ranges {
    /// Non-contiguous ranges, with adjacent insertions merged.
    ranges: Vec<Range<usize>>,
    /// Total number of selected values across all ranges.
    len: usize,
}

impl Ranges {
    fn push(&mut self, range: Range<usize>) {
        self.len += range.len();
        if let Some(last) = self.ranges.last_mut()
            && last.end == range.start
        {
            last.end = range.end;
        } else {
            self.ranges.push(range);
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

impl Clear for Ranges {
    fn clear(&mut self) {
        self.ranges.clear();
        self.len = 0;
    }
}

pub(crate) trait VecExt<T> {
    fn push_ranged(&mut self, ranges: &mut Ranges, value: T);

    fn ranged<'a>(&'a self, ranges: &'a Ranges) -> RangedSlice<'a, T>;
}

impl<T> VecExt<T> for Vec<T> {
    fn push_ranged(&mut self, ranges: &mut Ranges, value: T) {
        self.push(value);
        let end = self.len();
        ranges.push(end - 1..end);
    }

    fn ranged<'a>(&'a self, ranges: &'a Ranges) -> RangedSlice<'a, T> {
        RangedSlice::new(self, ranges)
    }
}

/// Read-only view of values selected from a shared buffer by [`Ranges`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct RangedSlice<'a, T> {
    /// Shared buffer from which values are selected.
    buffer: &'a [T],
    /// Ranges selecting values from `buffer`.
    ranges: &'a [Range<usize>],
    /// Total number of selected values.
    len: usize,
}

impl<'a, T> RangedSlice<'a, T> {
    pub(crate) const fn empty() -> Self {
        Self {
            buffer: &[],
            ranges: &[],
            len: 0,
        }
    }

    fn new(buffer: &'a [T], ranges: &'a Ranges) -> Self {
        Self {
            buffer,
            ranges: &ranges.ranges,
            len: ranges.len,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn slices(&self) -> impl Iterator<Item = &'a [T]> + '_ {
        self.ranges.iter().map(|range| &self.buffer[range.clone()])
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &'a T> + '_ {
        self.slices().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::{DimensionConstraints, Ranges, VecExt};
    use alloc::vec;
    use alloc::vec::Vec;
    use vello_common::util::Clear;

    #[test]
    fn ranged_slices() {
        let mut buffer = Vec::new();
        let mut selected = Ranges::default();
        let mut other = Ranges::default();

        buffer.push_ranged(&mut selected, 1);
        buffer.push_ranged(&mut selected, 2);
        buffer.push_ranged(&mut other, 10);
        buffer.push_ranged(&mut selected, 3);
        buffer.push_ranged(&mut selected, 4);
        buffer.push_ranged(&mut selected, 5);
        buffer.push_ranged(&mut other, 11);
        buffer.push_ranged(&mut other, 12);
        buffer.push_ranged(&mut selected, 6);

        let view = buffer.ranged(&selected);
        assert_eq!(view.len(), 6);
        assert_eq!(
            view.slices().collect::<Vec<_>>(),
            vec![&[1, 2][..], &[3, 4, 5], &[6]]
        );
        assert_eq!(view.iter().copied().collect::<Vec<_>>(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn ranges_clear() {
        let mut buffer = Vec::new();
        let mut ranges = Ranges::default();

        buffer.push_ranged(&mut ranges, 1);
        buffer.push_ranged(&mut ranges, 2);
        ranges.clear();

        assert_eq!(ranges.len(), 0);
        assert_eq!(buffer.ranged(&ranges).iter().count(), 0);

        buffer.push_ranged(&mut ranges, 3);
        assert_eq!(
            buffer.ranged(&ranges).iter().copied().collect::<Vec<_>>(),
            [3]
        );
    }

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
