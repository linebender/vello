//! The flood filter.

use crate::color::{AlphaColor, Srgb};

/// A flood filter.
#[derive(Debug)]
pub struct Flood {
    /// The flood color.
    pub color: AlphaColor<Srgb>,
}

impl Flood {
    /// Create a new flood filter with the specified color.
    pub fn new(color: AlphaColor<Srgb>) -> Self {
        Self { color }
    }
}
