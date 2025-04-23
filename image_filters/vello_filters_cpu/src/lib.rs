// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A library of image filters which run on the CPU.
//!
//! ## Color spaces
//!
//! Implementations of filters in this library are designed for the [sRGB][color::Srgb] primaries.
//! However, in most cases these should be applied to.
//!
//!

// LINEBENDER LINT SET - lib.rs - v3
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![no_std]

// We currently use it for docs
use color as _;

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

pub mod blur;

// /// The color space filters should operate in
// ///
// /// TODO: Should this be in the filter, or handled beforehand?
// pub enum ColorInterpolationFilters {
//     LinearRgb,
//     SRgb,
// }

#[derive(Debug)]
/// An image.
pub struct Image<Pixel> {
    pub width: u16,
    pub height: u16,
    /// Pixels, stored in row-major order.
    ///
    /// Note that in some cases this might store *too many* pixels.
    pub pixels: Vec<Pixel>,
}

impl<Pixel> Image<Pixel> {
    /// Calculate the size of the data vector for this image.
    pub fn calc_data_size(width: u16, height: u16) -> usize {
        usize::from(width) * usize::from(height)
    }

    /// Calculate the size of the pixels needed.
    pub fn total_data_len(&self) -> usize {
        Self::calc_data_size(self.width, self.height)
    }
}

impl Image<NaivePremulPixel> {
    pub fn empty_scratch() -> Self {
        Self {
            width: 0,
            height: 0,
            pixels: vec![],
        }
    }
    /// Resize this `Image` to be used as a scratch buffer.
    pub fn resize_for_scratch(&mut self, width: u16, height: u16) {
        let total_size = Self::calc_data_size(width, height);
        // We don't want to shrink here, because the garbage data will be handled later.
        if total_size > self.pixels.len() {
            self.pixels.resize(total_size, [0.; 4]);
        }
        self.height = height;
        self.width = width;
    }
}

/// The type used internally in most filters, to share core implementations between instantiations.
///
/// That is, this type is equivalent to [`PremulColor`][color::PremulColor], but without a color
/// space at all.
// TODO: How reasonable is it to use f32 here?
pub type NaivePremulPixel = [f32; 4];

// TODO: This will be useful for filters which only operate on alpha values. We don't currently have any of these.
// pub type NaiveAlpha = f32;
