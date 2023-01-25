//! Types that are shared between the main crate and build.

/// The type of resource that will be bound to a slot in a shader.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BindType {
    /// A storage buffer with read/write access.
    Buffer,
    /// A storage buffer with read only access.
    BufReadOnly,
    /// A small storage buffer to be used as uniforms.
    Uniform,
    /// A storage image.
    Image,
    /// A storage image with read only access.
    ImageRead,
    // TODO: Sampler, maybe others
}

impl BindType {
    pub fn is_mutable(self) -> bool {
        matches!(self, Self::Buffer | Self::Image)
    }
}

#[derive(Clone, Debug)]
pub struct BindingInfo {
    pub name: Option<String>,
    pub location: (u32, u32),
    pub ty: BindType,
}
