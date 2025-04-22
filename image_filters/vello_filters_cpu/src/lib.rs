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

extern crate alloc;

use alloc::vec::Vec;

pub mod blur;

/// The colour space filters should operate in
///
/// TODO: Should this be in the filter, or handled beforehand?
pub enum ColorInterpolationFilters {
    LinearRgb,
    SRgb,
}

pub struct Image<Pixel> {
    width: u16,
    height: u16,
    pixels: Vec<Pixel>,
}

/// The type used internally in most filters, to share core implementations between instantiations.
///
/// That is, this type is equivalent to [`PremulColor`][color::PremulColor], but without a color
/// space at all.
// TODO: How reasonable is it to use f32 here?
pub type NaivePremulPixel = [f32; 4];

pub type NaiveAlpha = f32;
