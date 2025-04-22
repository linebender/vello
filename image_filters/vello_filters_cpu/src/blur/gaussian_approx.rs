// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Approximations of a Gaussian blur using three box blurs along each axis.

use crate::NaivePremulPixel;

pub fn approx_gauss_box_blur(x_radius: u16, y_radius: u16) {
    let x_radius: usize = x_radius.into();
    let y_radius: usize = y_radius.into();
    box_blur_row(x_radius, a, b);
    box_blur_row(x_radius, b, a);
    box_blur_row(x_radius, a, b);
    todo!("Transpose b to be column-major");
    // Transposing into column-major here means that the following operations can be
    // much more cache friendly, and more easily parallelised.
    box_blur_row(radius, b, a);
    box_blur_row(radius, a, b);
    box_blur_row(radius, b, a);
}

pub fn box_blur_row(radius: usize, input: &[NaivePremulPixel], output: &mut [NaivePremulPixel]) {}
