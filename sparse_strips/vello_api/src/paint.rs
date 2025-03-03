//! Types for paints.

use peniko::color::{AlphaColor, Srgb};

// TODO: This will probably turn into a generic type where
// vello-hybrid and vello-cpu provide their own instantiations for
// a `Pattern` type.
/// A paint used for filling or stroking paths.
#[derive(Debug, Clone)]
pub enum Paint {
    /// A solid color.
    Solid(AlphaColor<Srgb>),
    /// A gradient.
    Gradient(()),
    /// A pattern.
    Pattern(()),
}

impl From<AlphaColor<Srgb>> for Paint {
    fn from(value: AlphaColor<Srgb>) -> Self {
        Paint::Solid(value)
    }
}
