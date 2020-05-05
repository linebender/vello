/// The cross-platform abstraction for a GPU device.
///
/// This abstraction is inspired by gfx-hal, but is specialized to the needs of piet-gpu.
/// In time, it may go away and be replaced by either gfx-hal or wgpu.
pub mod vulkan;

/// This isn't great but is expedient.
pub type Error = Box<dyn std::error::Error>;

#[derive(Copy, Clone, Debug)]
pub enum ImageLayout {
    Undefined,
    Present,
    BlitSrc,
    BlitDst,
    General,
}

pub trait Device: Sized {
    type Buffer;
    type Image;
    type MemFlags: MemFlags;
    type Pipeline;
    type DescriptorSet;
    type QueryPool;
    type CmdBuf: CmdBuf<Self>;
    type Fence;
    type Semaphore;

    fn create_buffer(&self, size: u64, mem_flags: Self::MemFlags) -> Result<Self::Buffer, Error>;

    unsafe fn create_image2d(
        &self,
        width: u32,
        height: u32,
        mem_flags: Self::MemFlags,
    ) -> Result<Self::Image, Error>;

    unsafe fn create_simple_compute_pipeline(
        &self,
        code: &[u8],
        n_buffers: u32,
        n_images: u32,
    ) -> Result<Self::Pipeline, Error>;

    unsafe fn create_descriptor_set(
        &self,
        pipeline: &Self::Pipeline,
        bufs: &[&Self::Buffer],
        images: &[&Self::Image],
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
    unsafe fn reap_query_pool(&self, pool: &Self::QueryPool) -> Result<Vec<f64>, Error>;

    unsafe fn run_cmd_buf(
        &self,
        cmd_buf: &Self::CmdBuf,
        wait_semaphores: &[Self::Semaphore],
        signal_semaphores: &[Self::Semaphore],
        fence: Option<&Self::Fence>,
    ) -> Result<(), Error>;

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

    unsafe fn create_semaphore(&self) -> Result<Self::Semaphore, Error>;
    unsafe fn create_fence(&self, signaled: bool) -> Result<Self::Fence, Error>;
    unsafe fn wait_and_reset(&self, fences: &[Self::Fence]) -> Result<(), Error>;
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

    unsafe fn image_barrier(
        &mut self,
        image: &D::Image,
        src_layout: ImageLayout,
        dst_layout: ImageLayout,
    );

    /// Clear the buffer.
    ///
    /// This is readily supported in Vulkan, but for portability it is remarkably
    /// tricky (unimplemented in gfx-hal right now). Possibly best to write a compute
    /// kernel, or organize the code not to need it.
    unsafe fn clear_buffer(&self, buffer: &D::Buffer);

    unsafe fn copy_buffer(&self, src: &D::Buffer, dst: &D::Buffer);

    unsafe fn copy_image_to_buffer(&self, src: &D::Image, dst: &D::Buffer);

    // low portability, dx12 doesn't support it natively
    unsafe fn blit_image(&self, src: &D::Image, dst: &D::Image);

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
