// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter effect implementations for CPU rendering.
//!
//! This module provides implementations of various filter effects that can be applied
//! to rendered layers. Each filter takes an input buffer and produces an output buffer.

/// Apply an identity filter to the layer.
///
/// This is a no-op filter that simply copies the input to the output.
/// It's primarily used for validation and testing of the filter infrastructure.
///
/// # Parameters
/// - `input`: The input layer buffer (RGBA, premultiplied)
/// - `output`: The output buffer to write to (RGBA, premultiplied)
/// - `width`: Width of the layer in pixels
/// - `height`: Height of the layer in pixels
pub fn apply_identity_filter(input: &[u8], output: &mut [u8], width: u16, height: u16) {
    let expected_size = usize::from(width) * usize::from(height) * 4;
    assert_eq!(
        input.len(),
        expected_size,
        "Input buffer size mismatch: expected {}, got {}",
        expected_size,
        input.len()
    );
    assert_eq!(
        output.len(),
        expected_size,
        "Output buffer size mismatch: expected {}, got {}",
        expected_size,
        output.len()
    );

    output.copy_from_slice(input);
}

/// Apply a no-op filter that leaves the layer unchanged.
///
/// Unlike `apply_identity_filter`, this doesn't even copy the data.
/// It's used when we want to test the filter infrastructure without any actual filtering.
pub fn apply_nop_filter(_input: &[u8], _output: &mut [u8], _width: u16, _height: u16) {
    // Intentionally empty - no operation
}

/// Apply an inversion filter to the layer.
///
/// Inverts the RGB color channels while preserving the alpha channel.
/// This implements the CSS `invert(1.0)` filter function.
///
/// # Parameters
/// - `input`: The input layer buffer (RGBA, premultiplied)
/// - `output`: The output buffer to write to (RGBA, premultiplied)
/// - `width`: Width of the layer in pixels
/// - `height`: Height of the layer in pixels
///
/// # Premultiplied Alpha Handling
/// Since the input is premultiplied, we need to:
/// 1. Unpremultiply to get the original RGB values
/// 2. Invert the RGB channels
/// 3. Premultiply again with the original alpha
pub fn apply_invert_filter(input: &[u8], output: &mut [u8], width: u16, height: u16) {
    let expected_size = usize::from(width) * usize::from(height) * 4;
    assert_eq!(
        input.len(),
        expected_size,
        "Input buffer size mismatch: expected {}, got {}",
        expected_size,
        input.len()
    );
    assert_eq!(
        output.len(),
        expected_size,
        "Output buffer size mismatch: expected {}, got {}",
        expected_size,
        output.len()
    );

    for i in (0..input.len()).step_by(4) {
        let r = input[i];
        let g = input[i + 1];
        let b = input[i + 2];
        let a = input[i + 3];

        // For premultiplied colors, we need to unpremultiply, invert, and re-premultiply
        if a == 0 {
            // Fully transparent pixel - output is also transparent
            output[i] = 0;
            output[i + 1] = 0;
            output[i + 2] = 0;
            output[i + 3] = 0;
        } else if a == 255 {
            // Fully opaque - simple inversion
            output[i] = 255 - r;
            output[i + 1] = 255 - g;
            output[i + 2] = 255 - b;
            output[i + 3] = a;
        } else {
            // Partial transparency - unpremultiply, invert, premultiply
            let alpha_f = f32::from(a) / 255.0;

            // Unpremultiply
            let r_unpremul = (f32::from(r) / alpha_f).min(255.0);
            let g_unpremul = (f32::from(g) / alpha_f).min(255.0);
            let b_unpremul = (f32::from(b) / alpha_f).min(255.0);

            // Invert
            let r_inverted = 255.0 - r_unpremul;
            let g_inverted = 255.0 - g_unpremul;
            let b_inverted = 255.0 - b_unpremul;

            // Premultiply again
            output[i] = (r_inverted * alpha_f) as u8;
            output[i + 1] = (g_inverted * alpha_f) as u8;
            output[i + 2] = (b_inverted * alpha_f) as u8;
            output[i + 3] = a;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_filter_copies_data() {
        let input = vec![255u8; 100 * 100 * 4];
        let mut output = vec![0u8; 100 * 100 * 4];

        apply_identity_filter(&input, &mut output, 100, 100);

        assert_eq!(input, output);
    }

    #[test]
    fn identity_filter_preserves_colors() {
        let mut input = vec![0u8; 10 * 10 * 4];
        // Set some known pixel values (R, G, B, A)
        input[0] = 255; // R
        input[1] = 128; // G
        input[2] = 64; // B
        input[3] = 255; // A

        let mut output = vec![0u8; 10 * 10 * 4];

        apply_identity_filter(&input, &mut output, 10, 10);

        assert_eq!(output[0], 255);
        assert_eq!(output[1], 128);
        assert_eq!(output[2], 64);
        assert_eq!(output[3], 255);
    }

    #[test]
    #[should_panic(expected = "Input buffer size mismatch")]
    fn identity_filter_validates_input_size() {
        let input = vec![0u8; 50]; // Wrong size
        let mut output = vec![0u8; 100 * 100 * 4];

        apply_identity_filter(&input, &mut output, 100, 100);
    }

    #[test]
    #[should_panic(expected = "Output buffer size mismatch")]
    fn identity_filter_validates_output_size() {
        let input = vec![0u8; 100 * 100 * 4];
        let mut output = vec![0u8; 50]; // Wrong size

        apply_identity_filter(&input, &mut output, 100, 100);
    }

    #[test]
    fn invert_filter_inverts_opaque_colors() {
        let mut input = vec![0u8; 10 * 10 * 4];
        // Set a red pixel (opaque)
        input[0] = 255; // R
        input[1] = 0; // G
        input[2] = 0; // B
        input[3] = 255; // A

        let mut output = vec![0u8; 10 * 10 * 4];
        apply_invert_filter(&input, &mut output, 10, 10);

        // Should be inverted to cyan
        assert_eq!(output[0], 0); // R inverted
        assert_eq!(output[1], 255); // G inverted
        assert_eq!(output[2], 255); // B inverted
        assert_eq!(output[3], 255); // A preserved
    }

    #[test]
    fn invert_filter_preserves_transparent_pixels() {
        let input = vec![0u8; 10 * 10 * 4];
        let mut output = vec![0u8; 10 * 10 * 4];

        apply_invert_filter(&input, &mut output, 10, 10);

        // Should remain transparent
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 0);
        assert_eq!(output[2], 0);
        assert_eq!(output[3], 0);
    }

    #[test]
    fn invert_filter_inverts_white_to_black() {
        let mut input = vec![0u8; 10 * 10 * 4];
        // White pixel
        input[0] = 255;
        input[1] = 255;
        input[2] = 255;
        input[3] = 255;

        let mut output = vec![0u8; 10 * 10 * 4];
        apply_invert_filter(&input, &mut output, 10, 10);

        // Should be black
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 0);
        assert_eq!(output[2], 0);
        assert_eq!(output[3], 255);
    }
}
