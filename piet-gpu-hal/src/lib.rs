/// The cross-platform abstraction for a GPU device.
///
/// This abstraction is inspired by gfx-hal, but is specialized to the needs of piet-gpu.
/// In time, it may go away and be replaced by either gfx-hal or wgpu.
use bitflags::bitflags;

pub mod backend;
pub mod hub;

#[macro_use]
mod macros;

// TODO: Don't make the module pub, but do figure out which types to
// export at the root level.
pub mod mux;

// TODO: because these are conditionally included, "cargo fmt" does not
// see them. Figure that out, possibly including running rustfmt manually.
mux_cfg! {
    #[cfg(vk)]
    pub mod vulkan;
}
mux_cfg! {
    #[cfg(dx12)]
    pub mod dx12;
}
#[cfg(target_os = "macos")]
pub mod metal;

/// The common error type for the crate.
///
/// This keeps things imple and can be expanded later.
pub type Error = Box<dyn std::error::Error>;

pub use crate::backend::CmdBuf;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    Undefined,
    Present,
    BlitSrc,
    BlitDst,
    General,
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

bitflags! {
    /// The intended usage for this buffer.
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
        // May add other types.
    }
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
    /// Info about subgroup size control, if available.
    pub subgroup_size: Option<SubgroupSize>,
    /// The GPU supports a real, grown-ass memory model.
    pub has_memory_model: bool,
    /// Whether staging buffers should be used.
    pub use_staging_buffers: bool,
}

#[derive(Clone, Debug)]
pub struct SubgroupSize {
    min: u32,
    max: u32,
}
