// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This is a utility library to help integrate the [Vello] shader modules into any renderer
//! project. It provides the necessary metadata to construct the individual compute pipelines
//! on any GPU API while leaving the responsibility of all API interactions
//! (such as resource management and command encoding) up to the client.
//!
//! The shaders can be pre-compiled to any target shading language at build time based on
//! feature flags. Currently only WGSL and Metal Shading Language are supported.
//!
//! Your first choice should be to use the build time generated [`SHADERS`].
//!
//! Alternatively you could use runtime compiled shaders via the [`compile`] module
//! by enabling the `compile` feature. This option is useful for active development
//! as it enables you to do hot reloading of shaders.
//!
//! [Vello]: https://github.com/linebender/vello

// LINEBENDER LINT SET - lib.rs - v2
// See https://linebender.org/wiki/canonical-lints/
// These lints aren't included in Cargo.toml because they
// shouldn't apply to examples and tests
#![warn(unused_crate_dependencies)]
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
// Need to allow instead of expect until Rust 1.83 https://github.com/rust-lang/rust/pull/130025
#![allow(missing_docs, reason = "We have many as-yet undocumented items.")]
#![expect(missing_debug_implementations, reason = "Deferred")]
#![allow(
    unnameable_types,
    clippy::cast_possible_truncation,
    clippy::missing_assert_message,
    clippy::print_stdout,
    clippy::todo,
    reason = "Deferred, only apply in some feature sets so not expect"
)]

mod types;

#[cfg(feature = "compile")]
pub mod compile;
#[cfg(feature = "cpu")]
pub mod cpu;

#[cfg(feature = "msl")]
pub use types::msl;
pub use types::{BindType, BindingInfo, WorkgroupBufferInfo};

use std::borrow::Cow;

#[derive(Clone, Debug)]
pub struct ComputeShader<'a> {
    pub name: Cow<'a, str>,
    pub workgroup_size: [u32; 3],
    pub bindings: Cow<'a, [BindType]>,
    pub workgroup_buffers: Cow<'a, [WorkgroupBufferInfo]>,

    #[cfg(feature = "wgsl")]
    pub wgsl: WgslSource<'a>,

    #[cfg(feature = "msl")]
    pub msl: MslSource<'a>,
}

#[cfg(feature = "wgsl")]
#[derive(Clone, Debug)]
pub struct WgslSource<'a> {
    pub code: Cow<'a, str>,

    /// Contains the binding index of each resource listed in `ComputeShader::bindings`.
    /// This is guaranteed to have the same element count as `ComputeShader::bindings`.
    ///
    /// In WGSL, each index directly corresponds to the value of the corresponding
    /// `@binding(..)` declaration in the shader source. The bind group index (i.e.
    /// value of `@group(..)`) is always 0.
    ///
    /// Example:
    /// --------
    /// ```wgsl
    /// // An unused binding (i.e. declaration is not reachable from the entry-point)
    /// @group(0) @binding(0) var<uniform> foo: Foo;
    ///
    /// // Used bindings:
    /// @group(0) @binding(1) var<storage> buffer: Buffer;
    /// @group(0) @binding(2) var tex: texture_2d<f32>;
    /// ```
    /// This results in the following bindings:
    /// ```rust,ignore
    ///   bindings: [BindType::Buffer, BindType::ImageRead],
    ///   // ...
    ///   wgsl: WgslSource {
    ///       code: /* ... */,
    ///       binding_indices: [1, 2],
    ///   },
    /// ```
    pub binding_indices: Cow<'a, [u8]>,
}

#[cfg(feature = "msl")]
#[derive(Clone, Debug)]
pub struct MslSource<'a> {
    pub code: Cow<'a, str>,

    /// Contains the binding index of each resource listed in [`ComputeShader::bindings`].
    /// This is guaranteed to have the same element count as `ComputeShader::bindings`.
    ///
    /// In MSL, each index is scoped to the index range of the corresponding resource type.
    ///
    /// Example:
    /// --------
    /// ```wgsl
    /// // An unused binding (i.e. declaration is not reachable from the entry-point)
    /// @group(0) @binding(0) var<uniform> foo: Foo;
    ///
    /// // Used bindings:
    /// @group(0) @binding(1) var<storage> buffer: Buffer;
    /// @group(0) @binding(2) var tex: texture_2d<f32>;
    /// ```
    /// This results in the following bindings:
    /// ```rust,ignore
    ///   bindings: [BindType::Buffer, BindType::ImageRead],
    ///   // ...
    ///   msl: MslSource {
    ///       code: /* ... */,
    ///       // In MSL these would be declared as `[[buffer(0)]]` and `[[texture(0)]]`.
    ///       binding_indices: [msl::BindingIndex::Buffer(0), msl::BindingIndex::Texture(0)],
    ///   },
    /// ```
    pub binding_indices: Cow<'a, [msl::BindingIndex]>,
}

include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

pub use generated::SHADERS;
