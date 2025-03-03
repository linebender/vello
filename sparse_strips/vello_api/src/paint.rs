use peniko::color::{AlphaColor, Srgb};

// TODO: Use `peniko::Brush` here? Though it will be tricky
// because `Image` might require a different implementation on GPU, and
// `Gradient` is also missing a `transform` attribute.
/// A paint used for filling or stroking paths.
#[derive(Debug, Clone)]
pub enum Paint {
    Solid(AlphaColor<Srgb>),
}

impl From<AlphaColor<Srgb>> for Paint {
    fn from(value: AlphaColor<Srgb>) -> Self {
        Paint::Solid(value)
    }
}