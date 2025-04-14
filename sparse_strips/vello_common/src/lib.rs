// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This crate contains core data structures and utilities shared across crates. It includes
//! foundational types for path geometry, tiling, and other common operations used in both CPU and
//! hybrid CPU/GPU rendering.

#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]
#![expect(
    clippy::cast_possible_truncation,
    reason = "We temporarily ignore those because the casts\
only break in edge cases, and some of them are also only related to conversions from f64 to f32."
)]

pub mod coarse;
pub mod flatten;
pub mod glyph;
pub mod pico_svg;
pub mod pixmap;
pub mod strip;
pub mod tile;

use crate::kurbo::{Affine, Rect};
pub use vello_api::*;

/// Additional methods for affine transformations.
pub trait AffineExt {
    /// Whether the affine transformation has a skewing factor.
    ///
    /// Note that this also includes rotations!
    fn has_skew(&self) -> bool;
}

impl AffineExt for Affine {
    fn has_skew(&self) -> bool {
        let coeffs = self.as_coeffs();

        coeffs[1] != 0.0 || coeffs[2] != 0.0
    }
}

/// Transform a rect using a transform with just a scaling and translation factor.
///
/// This method should not be called with a transform that has a skewing factor!
pub fn transform_non_skewed_rect(rect: &Rect, affine: Affine) -> Rect {
    debug_assert!(
        !affine.has_skew(),
        "this method should only be called with non-skewing transforms"
    );
    let [a, _, _, d, _, _] = affine.as_coeffs();

    Rect::new(a * rect.x0, d * rect.y0, a * rect.x1, d * rect.y1) + affine.translation()
}
