// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

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

#[derive(Clone, Debug)]
pub struct WorkgroupBufferInfo {
    pub size_in_bytes: u32,
    /// The order in which the workgroup variable is declared in the shader module.
    pub index: u32,
}

#[cfg(feature = "msl")]
pub mod msl {
    use std::fmt;

    #[derive(Clone)]
    pub enum BindingIndex {
        Buffer(u8),
        Texture(u8),
    }

    impl fmt::Debug for BindingIndex {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match *self {
                Self::Buffer(i) => write!(f, "msl::BindingIndex::Buffer({i})"),
                Self::Texture(i) => write!(f, "msl::BindingIndex::Texture({i})"),
            }
        }
    }
}
