// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

mod types;

#[cfg(feature = "compile")]
pub mod compile;

pub use types::{BindType, BindingInfo, WorkgroupBufferInfo};

use std::borrow::Cow;

#[derive(Clone, Debug)]
pub struct ComputeShader<'a> {
    pub name: Cow<'a, str>,
    pub workgroup_size: [u32; 3],
    pub bindings: Cow<'a, [BindType]>,
    pub workgroup_buffers: Cow<'a, [WorkgroupBufferInfo]>,

    #[cfg(feature = "wgsl")]
    pub wgsl: Cow<'a, str>,

    #[cfg(feature = "msl")]
    pub msl: Cow<'a, str>,
}

pub trait PipelineHost {
    type Device;
    type ComputePipeline;
    type Error;

    fn new_compute_pipeline(
        &mut self,
        device: &Self::Device,
        shader: &ComputeShader,
    ) -> Result<Self::ComputePipeline, Self::Error>;
}

include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

pub use gen::SHADERS;
