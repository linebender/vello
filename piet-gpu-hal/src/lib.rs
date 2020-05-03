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
    type QueryPool;
    type CmdBuf: CmdBuf<Self>;

    fn create_buffer(&self, size: u64, mem_flags: Self::MemFlags) -> Result<Self::Buffer, Error>;

    unsafe fn create_simple_compute_pipeline(
        &self,
        code: &[u8],
        n_buffers: u32,
        n_subgroups: Option<u32>,
    ) -> Result<Self::Pipeline, Error>;

    unsafe fn create_descriptor_set(
        &self,
        pipeline: &Self::Pipeline,
        bufs: &[&Self::Buffer],
    ) -> Result<Self::DescriptorSet, Error>;

    fn create_cmd_buf(&self) -> Result<Self::CmdBuf, Error>;

    fn create_query_pool(&self, n_queries: u32) -> Result<Self::QueryPool, Error>;

    /// Get results from query pool, destroying it in the process.
    ///
    /// The returned vector is one less than the number of queries; the first is used as
    /// a baseline.
    ///
    /// # Safety
    /// All submitted commands that refer to this query pool must have completed.
    unsafe fn reap_query_pool(&self, pool: Self::QueryPool) -> Result<Vec<f64>, Error>;

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

    unsafe fn dispatch(
        &mut self,
        pipeline: &D::Pipeline,
        descriptor_set: &D::DescriptorSet,
        size: (u32, u32, u32),
    );

    unsafe fn memory_barrier(&mut self);

    /// Clear the buffer.
    ///
    /// This is readily supported in Vulkan, but for portability it is remarkably
    /// tricky (unimplemented in gfx-hal right now). Possibly best to write a compute
    /// kernel, or organize the code not to need it.
    unsafe fn clear_buffer(&self, buffer: &D::Buffer);

    unsafe fn copy_buffer(&self, src: &D::Buffer, dst: &D::Buffer);

    /// Reset the query pool.
    ///
    /// The query pool must be reset before each use, to avoid validation errors.
    /// This is annoying, and we could tweak the API to make it implicit, doing
    /// the reset before the first timestamp write.
    unsafe fn reset_query_pool(&mut self, pool: &D::QueryPool);

    unsafe fn write_timestamp(&mut self, pool: &D::QueryPool, query: u32);
}

pub trait MemFlags: Sized + Clone + Copy {
    fn device_local() -> Self;

    fn host_coherent() -> Self;
}
