use crate::color::{AlphaColor, Srgb};
use crate::kurbo::Rect;

#[derive(Debug)]
pub struct BlurredRectangle {
    pub(crate) rect: Rect,
    pub(crate) color: AlphaColor<Srgb>,
    pub(crate) radius: f32,
    pub(crate) std_dev: f32,
}
