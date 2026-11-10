// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Blurred, rounded rectangles.
use crate::color::{AlphaColor, Srgb};
use crate::kurbo::Rect;

/// A blurred, rounded rectangle.
#[derive(Debug)]
pub struct BlurredRoundedRectangle {
    /// The base rectangle to use for the blur effect.
    pub rect: Rect,
    /// The color of the blurred rectangle.
    pub color: AlphaColor<Srgb>,
    /// The radius of the rounded rectangle's corners.
    pub radius: f32,
    /// The standard deviation of the blur effect.
    pub std_dev: f32,
    /// Whether to paint the inverse (`1 - alpha`) of the blur coverage.
    ///
    /// When `true`, the paint is fully opaque outside the blurred rectangle and fades to
    /// transparent inside it. This is useful for implementing inset box shadows.
    pub invert: bool,
}

/// The rectangle that `rect` blurred with a standard deviation of `std_dev` is painted in: `rect`
/// grown by three standard deviations, beyond which less than half a level of an 8-bit target
/// remains. Negative and NaN values of `std_dev` are treated as zero.
pub fn painted_rect(rect: Rect, std_dev: f32) -> Rect {
    let extent = 3.0 * f64::from(std_dev.max(0.0));
    rect.abs().inflate(extent, extent)
}

#[cfg(test)]
mod tests {
    use super::painted_rect;
    use crate::kurbo::Rect;

    #[test]
    fn painted_rect_reaches_three_standard_deviations() {
        assert_eq!(
            painted_rect(Rect::new(10.0, 20.0, 50.0, 40.0), 2.0),
            Rect::new(4.0, 14.0, 56.0, 46.0)
        );
    }

    #[test]
    fn painted_rect_without_a_blur_is_the_rect() {
        let rect = Rect::new(10.0, 20.0, 50.0, 40.0);
        for std_dev in [0.0, -2.0, f32::NAN] {
            assert_eq!(painted_rect(rect, std_dev), rect, "std_dev {std_dev}");
        }
    }

    #[test]
    fn painted_rect_of_a_flipped_rect_is_normalized() {
        assert_eq!(
            painted_rect(Rect::new(50.0, 40.0, 10.0, 20.0), 2.0),
            Rect::new(4.0, 14.0, 56.0, 46.0)
        );
    }
}
