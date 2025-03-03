// Note that this will probably disappear and be turned into a const generic in the future.
/// The height of a strip.
pub const STRIP_HEIGHT: usize = 4;

/// A strip.
#[derive(Debug, Clone, Copy)]
pub struct Strip {
    /// The x coordinate of the strip, in user coordinates.
    pub x: i32,
    /// The y coordinate of the strip, in user coordinates.
    pub y: u16,
    /// The index into the alpha buffer
    pub col: u32,
    /// The winding number at the start of the strip.
    pub winding: i32,
}