use crate::color::{AlphaColor, Srgb};
use crate::kurbo::Rect;

#[derive(Debug)]
pub struct BlurredRectangle {
    pub rect: Rect,
    pub color: AlphaColor<Srgb>,
    pub radius: f32,
    pub std_dev: f32,
}
