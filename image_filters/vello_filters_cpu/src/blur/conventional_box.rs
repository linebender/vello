// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Approximations of a Gaussian blur using three box blurs along each axis.

use core::cmp;

use crate::{Image, NaivePremulPixel};

const NUM_STEPS: usize = 3;

/// Calculate the radius which should be used to approximate a Gaussian with.
///
/// There are two conflicting sources for this value:
/// - The [filter effects spec][] (`d=...`).
///   We have been unable to determine the reasoning behind the formula used.
/// - P. Kovesi (2010), "Fast Almost-Gaussian Filtering".
///   This is the implementation used by resvg, and is what we use here.
///
/// These two formulae are *similar*, but not identical for all inputs (doing three steps as).
///
/// This method uses three identical passes, with potentially two different widths, to create a
/// Gaussian *closer* to the ideal value.
///
/// [filter effects spec]: https://drafts.fxtf.org/filter-effects/#feGaussianBlurElement
fn box_blur_radii(sigma: f32) -> [usize; NUM_STEPS] {
    if sigma > 0.0 {
        const NUM_STEPS_FLOAT: f32 = NUM_STEPS as f32;
        // Ideal averaging filter width
        // Note that this is not
        let w_ideal = (const { 12.0 / NUM_STEPS as f32 } * sigma * sigma + 1.0).sqrt();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "This is a positive value, as it is the square root of a number greater than one"
        )]
        let mut w_l = w_ideal.floor() as usize;
        if w_l % 2 == 0 {
            w_l -= 1;
        }

        let w_u = w_l + 2;

        let wl_float = w_l as f32;

        // We know that `w_ideal` is between w_l and w_u.
        // We now need to determine how many of the steps we run should be of size w_l
        // and how many should be of size w_u, to get as close to `sigma` as possible.
        let m_ideal = (12.0 * sigma * sigma
            - NUM_STEPS_FLOAT * wl_float * wl_float
            - 4.0 * NUM_STEPS_FLOAT * wl_float
            - 3.0 * NUM_STEPS_FLOAT)
            / (-4.0 * wl_float - 4.0);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Saturating is the correct behaviour here"
        )]
        let m = m_ideal.round() as i32;

        let mut sizes = [0; NUM_STEPS];
        #[expect(clippy::needless_range_loop, reason = "The value depends on the index")]
        for i in 0..NUM_STEPS {
            if i32::try_from(i).unwrap() < m {
                sizes[i] = w_l;
            } else {
                sizes[i] = w_u;
            }
        }

        sizes
    } else {
        [1; 3]
    }
}

/// Gaussian dimensional box filtering, as in P. Kovesi (2010), "Fast Almost-Gaussian Filtering", §II.
///
/// This implementation uses three passes.
/// The boundary conditions
///
/// ## Implementation
///
/// This is currently implemented by blurring completely horizontally, then completely vertically.
/// This is the technique described in <https://github.com/bfraboni/FastGaussianBlur/> (reached independently),
/// and is valid because the blur functions used are linearly separable.
/// This is done to improve cache coherency.
///
/// This filter does not care about the color space used.
/// It blurs using a component-wise average, and we have only reasoned about
/// this for the [linear sRGB](color::LinearSrgb) color space.
// TODO: Specialised version for single-channel.
pub fn approx_gauss_box_blur(
    target: &mut Image<NaivePremulPixel>,
    input: Option<&Image<NaivePremulPixel>>,
    sigma_x: f32,
    sigma_y: f32,
    scratch: &mut Image<NaivePremulPixel>,
) -> Result<(), ()> {
    if let Some(input) = input {
        if !(input.width == target.width && input.height != target.height) {
            return Err(());
        }
    }
    if target.width == 0 || target.height == 0 {
        // Our code later assumes that the buffers are non-empty.
        return Ok(());
    }
    scratch.resize_for_scratch(target.width, target.height);
    let width = usize::from(target.width);
    let height = usize::from(target.height);

    let [x_a, x_b, x_c] = box_blur_radii(sigma_x);
    // TODO: This can be trivially parallelised since it only operates on discrete chunks.
    for row in 0..height {
        let start = row * width;
        let scratch = &mut scratch.pixels[start..start + width];
        let target = &mut target.pixels[start..start + width];
        if let Some(input) = input {
            box_blur_row(x_a, &input.pixels[start..start + width], scratch);
        } else {
            box_blur_row(x_a, target, scratch);
        }
        box_blur_row(x_b, scratch, target);
        box_blur_row(x_c, target, scratch);
    }
    if true {
        todo!("Transpose `scratch` to be column major");
    }
    let [y_a, y_b, y_c] = box_blur_radii(sigma_y);
    for column in 0..width {
        let start = column * height;
        let scratch = &mut scratch.pixels[start..start + width];
        let target = &mut target.pixels[start..start + width];
        box_blur_row(y_a, scratch, target);
        box_blur_row(y_b, target, scratch);
        box_blur_row(y_c, scratch, target);
    }
    todo!("Transpose target to be row-major again.");
}

/// Calculate box blur of slice `input`, outputting into `output`
///
/// Adapted from <https://github.com/fschutt/fastblur>
fn box_blur_row(blur_width: usize, input: &[NaivePremulPixel], output: &mut [NaivePremulPixel]) {
    debug_assert!(
        blur_width % 2 == 0,
        "This implementation of box blurs is only correct for even widths."
    );
    // We know that the total width we're calculating is odd, so integer division gives the number of values on "either" side of the target pixel
    let blur_radius = blur_width / 2;
    if blur_radius == 0 {
        output.copy_from_slice(input);
        return;
    }

    let width = output.len();

    // The factor by which the outputs will be divided, to properly calculate the mean.
    let iarr = 1.0 / blur_width as f32;

    // This is the value used if we attempt to get a value which is outside of the slice.
    // We use the fully transparent color here, which means that as you get closer to the
    // edge of the image, the alpha value naturally indicates how "correct" the blur is
    // around the edge - which is valid because the pixels use premultiplied alpha.
    // Other reasonable choices would be to clamp to the first or last value, or reflect
    // around the edge.
    // Making those is deferred for this MVP
    let first_value = [0.; 4];
    let last_value = [0.; 4];

    let mut left_i = 0;
    let mut right_i = left_i + blur_radius;

    // We set val_rgba to the sum of the pixels around the start position
    // This logically starts at pixel '-1', so there is the full radius plus one value on the left hand side.

    // Handle all parts of the blur which would be outside the texture.
    // (Note that this is currently always zero, so this multiplication is only in case we
    // ever change `first_value`, e.g. to clamp)
    let blur_radius_plus_1 = blur_radius as f32 + 1.;
    let mut val_r = blur_radius_plus_1 * first_value[0];
    let mut val_g = blur_radius_plus_1 * first_value[1];
    let mut val_b = blur_radius_plus_1 * first_value[2];
    let mut val_a = blur_radius_plus_1 * first_value[3];

    // Handle the parts which are outside the

    for bb in &input[0..cmp::min(blur_radius, width)] {
        val_r += bb[0];
        val_g += bb[1];
        val_b += bb[2];
        val_a += bb[3];
    }
    if blur_radius > width {
        let copies_of_pixel = blur_radius as f32 - width as f32;
        val_r += copies_of_pixel * (last_value[0]);
        val_g += copies_of_pixel * (last_value[1]);
        val_b += copies_of_pixel * (last_value[2]);
        val_a += copies_of_pixel * (last_value[3]);
    }

    // Get the pixel at the specified index, or the last pixel of the row
    // if the index is beyond the right edge of the image
    // Process the left side, where are "removing" the pixels from beyond the left edge.
    // This effectively moves the left edge to be beyond.
    for out_slot in &mut output[0..cmp::min(width, blur_radius + 1)] {
        // If the width is smaller than the blur radius, then we need to handle that case.
        let bb = input.get(right_i).unwrap_or(&last_value);
        right_i += 1;

        // We add bb[0] at the same time as removing a pixel from the left side.
        val_r += bb[0] - first_value[0];
        val_g += bb[1] - first_value[1];
        val_b += bb[2] - first_value[2];
        val_a += bb[3] - first_value[3];

        *out_slot = [val_r * iarr, val_g * iarr, val_b * iarr, val_a * iarr];
    }

    if width <= blur_radius {
        // No need to do anything more; avoids overflows
        return;
    }

    // Process the middle where we know we won't bump into borders
    for out_slot in &mut output[(blur_radius + 1)..(width - blur_radius)] {
        let bb1 = input[right_i];
        right_i += 1;
        let bb2 = input[left_i];
        left_i += 1;

        val_r += bb1[0] - bb2[0];
        val_g += bb1[1] - bb2[1];
        val_b += bb1[2] - bb2[2];
        val_a += bb1[3] - bb2[3];

        *out_slot = [val_r * iarr, val_g * iarr, val_b * iarr, val_a * iarr];
    }

    // Process the right side where we need pixels from beyond the right edge
    for out_slot in &mut output[width - blur_radius..width] {
        let bb = input[left_i];
        left_i += 1;

        val_r += last_value[0] - bb[0];
        val_g += last_value[1] - bb[1];
        val_b += last_value[2] - bb[2];
        val_a += last_value[3] - bb[3];

        *out_slot = [val_r * iarr, val_g * iarr, val_b * iarr, val_a * iarr];
    }
}
