//! The offset filter.

/// Translation/shift filter.
///
/// This shifts the input image by `(dx, dy)` in device pixel space.
#[derive(Clone, Copy, Debug)]
pub struct Offset {
    /// The x-offset that should be applied.
    pub dx: f32,
    /// The y-offset that should be applied.
    pub dy: f32,
}

impl Offset {
    /// Create a new offset filter.
    pub fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }
}
