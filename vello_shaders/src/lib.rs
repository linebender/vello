mod types;

#[cfg(feature = "compile")]
pub mod compile;

pub use types::{BindType, BindingInfo};

use std::borrow::Cow;

#[derive(Clone, Debug)]
pub struct ComputeShader<'a> {
    pub name: Cow<'a, str>,
    pub code: Cow<'a, [u8]>,
    pub workgroup_size: [u32; 3],
    pub bindings: Cow<'a, [BindType]>,
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
