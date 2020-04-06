/// The cross-platform abstraction for a GPU device.
///
/// This abstraction is inspired by gfx-hal, but is specialized to the needs of piet-gpu.
/// In time, it may go away and be replaced by either gfx-hal or wgpu.

pub mod vulkan;

/// This isn't great but is expedient.
type Error = Box<dyn std::error::Error>;

pub trait Device: Sized {
    type Buffer;
    type MemFlags: MemFlags;
    type Pipeline;
    type DescriptorSet;
    type CmdBuf: CmdBuf<Self>;

    fn create_buffer(&self, size: u64, mem_flags: Self::MemFlags) -> Result<Self::Buffer, Error>;

    unsafe fn create_simple_compute_pipeline(
        &self,
        code: &[u8],
        n_buffers: u32,
    ) -> Result<Self::Pipeline, Error>;

    unsafe fn create_descriptor_set(
        &self,
        pipeline: &Self::Pipeline,
        bufs: &[&Self::Buffer],
    ) -> Result<Self::DescriptorSet, Error>;

    fn create_cmd_buf(&self) -> Result<Self::CmdBuf, Error>;

    unsafe fn run_cmd_buf(&self, cmd_buf: &Self::CmdBuf) -> Result<(), Error>;

    unsafe fn read_buffer<T: Sized>(
        &self,
        buffer: &Self::Buffer,
        result: &mut Vec<T>,
    ) -> Result<(), Error>;

    unsafe fn write_buffer<T: Sized>(
        &self,
        buffer: &Self::Buffer,
        contents: &[T],
    ) -> Result<(), Error>;
}

pub trait CmdBuf<D: Device> {
    unsafe fn begin(&mut self);

    unsafe fn finish(&mut self);

    unsafe fn dispatch(&mut self, pipeline: &D::Pipeline, descriptor_set: &D::DescriptorSet);

    unsafe fn memory_barrier(&mut self);
}

pub trait MemFlags: Sized {
    fn host_coherent() -> Self;
}
