///! Blurred rectangles.
use crate::color::{AlphaColor, Srgb};
use crate::kurbo::Rect;

/// A blurred rectangle.
#[derive(Debug)]
pub struct BlurredRectangle {
    /// The base rectangle to use for the blur effect.
    pub rect: Rect,
    /// The color of the blurred rectangle.
    pub color: AlphaColor<Srgb>,
    /// The radius of the blur effect.
    pub radius: f32,
    /// The standard deviation of the blur effect.
    pub std_dev: f32,
}
