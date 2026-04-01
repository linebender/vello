// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This is a utility library to help integrate `vello_hybrid` WebGPU wgsl shaders into glsl.

extern crate alloc;

#[cfg(feature = "gles")]
mod compile;
mod specialize;
#[cfg(feature = "gles")]
mod types;

pub use specialize::ShaderConstants;

mod generated {
    include!(concat!(env!("OUT_DIR"), "/compiled_shaders.rs"));
}

/// WGSL shader sources.
pub mod wgsl {
    use alloc::string::String;

    use crate::ShaderConstants;

    /// Returns specialised WGSL source.
    pub fn render_strips(constants: &ShaderConstants) -> String {
        constants.specialize_wgsl(super::generated::wgsl::RENDER_STRIPS)
    }

    /// WGSL source for the `clear_slots` shader.
    pub const CLEAR_SLOTS: &str = super::generated::wgsl::CLEAR_SLOTS;

    /// WGSL source for the `filters` shader.
    pub const FILTERS: &str = super::generated::wgsl::FILTERS;
}

/// GLSL ES shader sources.
#[cfg(feature = "gles")]
pub mod gles {
    /// GLSL ES sources for the `render_strips` shader.
    pub mod render_strips {
        use alloc::string::String;

        use crate::ShaderConstants;

        /// Returns specialised GLSL ES vertex source.
        pub fn vertex_source(constants: &ShaderConstants) -> String {
            constants.specialize_glsl(crate::generated::gles::render_strips::VERTEX_SOURCE)
        }

        /// Returns specialised GLSL ES fragment source.
        pub fn fragment_source(constants: &ShaderConstants) -> String {
            constants.specialize_glsl(crate::generated::gles::render_strips::FRAGMENT_SOURCE)
        }

        /// Vertex-stage reflection metadata.
        pub mod vertex {
            pub use crate::generated::gles::render_strips::vertex::*;
        }

        /// Fragment-stage reflection metadata.
        pub mod fragment {
            pub use crate::generated::gles::render_strips::fragment::*;
        }
    }

    /// GLSL ES sources for the `clear_slots` shader.
    pub mod clear_slots {
        pub use crate::generated::gles::clear_slots::*;
    }

    /// GLSL ES sources for the `filters` shader.
    pub mod filters {
        pub use crate::generated::gles::filters::*;
    }
}
