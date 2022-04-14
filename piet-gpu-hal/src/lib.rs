//! The cross-platform abstraction for a GPU device.
//!
//! This abstraction is inspired by gfx-hal, but is specialized to the needs of piet-gpu.
//! In time, it may go away and be replaced by either gfx-hal or wgpu.

use bitflags::bitflags;

mod backend;
mod bestfit;
mod bufwrite;
mod hub;

#[macro_use]
mod macros;

mod mux;

pub use crate::mux::{
    DescriptorSet, Device, Fence, Instance, Pipeline, QueryPool, Sampler, Semaphore, ShaderCode,
    Surface, Swapchain,
};
pub use bufwrite::BufWrite;
pub use hub::{
    BufReadGuard, BufWriteGuard, Buffer, CmdBuf, DescriptorSetBuilder, Image, RetainResource,
    Session, SubmittedCmdBuf,
};

// TODO: because these are conditionally included, "cargo fmt" does not
// see them. Figure that out, possibly including running rustfmt manually.
mux_cfg! {
    #[cfg(vk)]
    mod vulkan;
}
mux_cfg! {
    #[cfg(dx12)]
    mod dx12;
}
#[cfg(target_os = "macos")]
mod metal;

/// The common error type for the crate.
///
/// This keeps things simple and can be expanded later.
pub type Error = Box<dyn std::error::Error>;

bitflags! {
    /// Options when creating an instance.
    #[derive(Default)]
    pub struct InstanceFlags: u32 {
        /// Prefer DX12 over Vulkan.
        const DX12 = 0x1;
        // TODO: discrete vs integrated selection
    }
}

/// The GPU backend that was selected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendType {
    Vulkan,
    Dx12,
    Metal,
}

/// An image layout state.
///
/// An image must be in a particular layout state to be used for
/// a purpose such as being bound to a shader.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    /// The initial state for a newly created image.
    Undefined,
    /// A swapchain ready to be presented.
    Present,
    /// The source for a copy operation.
    BlitSrc,
    /// The destination for a copy operation.
    BlitDst,
    /// Read/write binding to a shader.
    General,
    /// Able to be sampled from by shaders.
    ShaderRead,
}

/// The type of sampling for image lookup.
///
/// This could take a lot more params, such as filtering, repeat, behavior
/// at edges, etc., but for now we'll keep it simple.
#[derive(Copy, Clone, Debug)]
pub enum SamplerParams {
    Nearest,
    Linear,
}

/// Image format.
#[derive(Copy, Clone, Debug)]
pub enum ImageFormat {
    // 8 bit grayscale / alpha
    A8,
    // 8 bit per pixel RGBA
    Rgba8,
}

bitflags! {
    /// The intended usage for a buffer, specified on creation.
    pub struct BufferUsage: u32 {
        /// The buffer can be mapped for reading CPU-side.
        const MAP_READ = 0x1;
        /// The buffer can be mapped for writing CPU-side.
        const MAP_WRITE = 0x2;
        /// The buffer can be copied from.
        const COPY_SRC = 0x4;
        /// The buffer can be copied to.
        const COPY_DST = 0x8;
        /// The buffer can be bound to a compute shader.
        const STORAGE = 0x80;
        /// The buffer can be used to store the results of queries.
        const QUERY_RESOLVE = 0x200;
        /// The buffer may be cleared.
        const CLEAR = 0x8000;
        // May add other types.
    }
}

/// The type of resource that will be bound to a slot in a shader.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BindType {
    /// A storage buffer with read/write access.
    Buffer,
    /// A storage buffer with read only access.
    BufReadOnly,
    /// A storage image.
    Image,
    /// A storage image with read only access.
    ///
    /// A note on this. None of the backends are currently making a
    /// distinction between Image and ImageRead as far as bindings go,
    /// but the `--hlsl-nonwritable-uav-texture-as-srv` option to
    /// spirv-cross (marked as unstable) would do so.
    ImageRead,
    // TODO: Uniform, Sampler, maybe others
}

/// Whether to map a buffer in read or write mode.
pub enum MapMode {
    /// Map for reading.
    Read,
    /// Map for writing.
    Write,
}

#[derive(Clone, Debug)]
/// Information about the GPU.
pub struct GpuInfo {
    /// The GPU supports descriptor indexing.
    pub has_descriptor_indexing: bool,
    /// The GPU supports subgroups.
    ///
    /// Right now, this just checks for basic subgroup capability (as
    /// required in Vulkan 1.1), and we should have finer grained
    /// queries for shuffles, etc.
    pub has_subgroups: bool,
    /// Limits on workgroup size for compute shaders.
    pub workgroup_limits: WorkgroupLimits,
    /// Info about subgroup size control, if available.
    pub subgroup_size: Option<SubgroupSize>,
    /// The GPU supports a real, grown-ass memory model.
    pub has_memory_model: bool,
    /// Whether staging buffers should be used.
    pub use_staging_buffers: bool,
}

/// The range of subgroup sizes supported by a back-end, when available.
///
/// The subgroup size is always a power of 2. The ability to specify
/// subgroup size for a compute shader is a newer feature, not always
/// available.
#[derive(Clone, Debug)]
pub struct SubgroupSize {
    pub min: u32,
    pub max: u32,
}

/// The range of workgroup sizes supported by a back-end.
#[derive(Clone, Debug)]
pub struct WorkgroupLimits {
    /// The maximum size on each workgroup dimension can be.
    pub max_size: [u32; 3],
    /// The maximum overall invocations a workgroup can have. That is, the product of sizes in each
    /// dimension.
    pub max_invocations: u32,
}

/// Options for creating a compute pass.
#[derive(Default)]
pub struct ComputePassDescriptor<'a> {
    // Maybe label should go here? It does in wgpu and wgpu_hal.
    /// Timer query parameters.
    ///
    /// To record timer queries for a compute pass, set the query pool, start
    /// query index, and end query index here. The indices must be less than
    /// the size of the query pool.
    timer_queries: Option<(&'a QueryPool, u32, u32)>,
}

impl<'a> ComputePassDescriptor<'a> {
    pub fn timer(pool: &'a QueryPool, start_query: u32, end_query: u32) -> ComputePassDescriptor {
        ComputePassDescriptor {
            timer_queries: Some((pool, start_query, end_query)),
        }
    }
}
