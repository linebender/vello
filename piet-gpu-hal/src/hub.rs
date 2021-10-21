//! A somewhat higher level GPU abstraction.
//!
//! This layer is on top of the lower-level layer that multiplexes different
//! back-ends. It handles details such as managing staging buffers for creating
//! buffers with initial content, deferring dropping of resources until command
//! submission is complete, and a bit more. These conveniences might expand
//! even more in time.

use std::convert::TryInto;
use std::sync::{Arc, Mutex, Weak};

use smallvec::SmallVec;

use crate::mux;

use crate::{BufferUsage, Error, GpuInfo, ImageLayout, SamplerParams};

pub use crate::mux::{DescriptorSet, Fence, Pipeline, QueryPool, Sampler, Semaphore, ShaderCode};

/// A session of GPU operations.
///
/// This abstraction is generally called a "device" in other APIs, but that
/// term is very overloaded. It is the point to access resource creation,
/// work submission, and related concerns.
///
/// Most of the methods are `&self`, indicating that they can be called from
/// multiple threads.
#[derive(Clone)]
pub struct Session(Arc<SessionInner>);

struct SessionInner {
    device: mux::Device,
    /// A pool of command buffers that can be reused.
    ///
    /// Currently this is not used, as it only works well on Vulkan. At some
    /// point, we will want to efficiently reuse command buffers rather than
    /// allocating them each time, but that is a TODO.
    cmd_buf_pool: Mutex<Vec<(mux::CmdBuf, Fence)>>,
    /// Command buffers that are still pending (so resources can't be freed yet).
    pending: Mutex<Vec<SubmittedCmdBufInner>>,
    /// A command buffer that is used for copying from staging buffers.
    staging_cmd_buf: Mutex<Option<CmdBuf>>,
    gpu_info: GpuInfo,
}

/// A command buffer.
///
/// Actual work done by the GPU is encoded into a command buffer and then
/// submitted to the session in a batch.
pub struct CmdBuf {
    cmd_buf: mux::CmdBuf,
    fence: Fence,
    resources: Vec<RetainResource>,
    session: Weak<SessionInner>,
}

/// A command buffer in submitted state.
///
/// Submission of a command buffer is asynchronous, meaning that the submit
/// method returns immediately. The work done in the command buffer cannot
/// be accessed (for example, readback from buffers written) until the the
/// submission is complete. The main purpose of this structure is to wait on
/// that completion.
pub struct SubmittedCmdBuf(Option<SubmittedCmdBufInner>, Weak<SessionInner>);

struct SubmittedCmdBufInner {
    // It's inconsistent, cmd_buf is unpacked, staging_cmd_buf isn't. Probably
    // better to chose one or the other.
    cmd_buf: mux::CmdBuf,
    fence: Fence,
    resources: Vec<RetainResource>,
    staging_cmd_buf: Option<CmdBuf>,
}

/// An image or texture.
///
/// At the moment, images are limited to 2D.
#[derive(Clone)]
pub struct Image(Arc<ImageInner>);

struct ImageInner {
    image: mux::Image,
    session: Weak<SessionInner>,
}

/// A buffer.
///
/// A buffer is a segment of memory that can be accessed by the GPU, and
/// in some cases also by the host (if the appropriate [`BufferUsage`] flags
/// are set).
#[derive(Clone)]
pub struct Buffer(Arc<BufferInner>);

struct BufferInner {
    buffer: mux::Buffer,
    session: Weak<SessionInner>,
}

/// A builder for creating pipelines.
///
/// Configure the signature (buffers and images accessed) for a pipeline,
/// which is essentially compiled shader code, ready to be dispatched.
pub struct PipelineBuilder(mux::PipelineBuilder);

/// A builder for creating descriptor sets.
///
/// Add bindings to the descriptor set before dispatching a shader.
pub struct DescriptorSetBuilder(mux::DescriptorSetBuilder);

/// Data types that can be stored in a GPU buffer.
pub unsafe trait PlainData {}

unsafe impl PlainData for u8 {}
unsafe impl PlainData for u16 {}
unsafe impl PlainData for u32 {}
unsafe impl PlainData for u64 {}
unsafe impl PlainData for i8 {}
unsafe impl PlainData for i16 {}
unsafe impl PlainData for i32 {}
unsafe impl PlainData for i64 {}
unsafe impl PlainData for f32 {}
unsafe impl PlainData for f64 {}

/// A resource to retain during the lifetime of a command submission.
pub enum RetainResource {
    Buffer(Buffer),
    Image(Image),
}

impl Session {
    /// Create a new session, choosing the best backend.
    pub fn new(device: mux::Device) -> Session {
        let gpu_info = device.query_gpu_info();
        Session(Arc::new(SessionInner {
            device,
            gpu_info,
            cmd_buf_pool: Default::default(),
            pending: Default::default(),
            staging_cmd_buf: Default::default(),
        }))
    }

    /// Create a new command buffer.
    ///
    /// The caller is responsible for inserting pipeline barriers and other
    /// transitions. If one dispatch writes a buffer (or image), and another
    /// reads it, a barrier must intervene. No such barrier is needed for
    /// uploads by the host before command submission, but a host barrier is
    /// needed if the host will do readback of any buffers written by the
    /// command list.
    pub fn cmd_buf(&self) -> Result<CmdBuf, Error> {
        self.poll_cleanup();
        let (cmd_buf, fence) = if let Some(cf) = self.0.cmd_buf_pool.lock().unwrap().pop() {
            cf
        } else {
            let cmd_buf = self.0.device.create_cmd_buf()?;
            let fence = unsafe { self.0.device.create_fence(false)? };
            (cmd_buf, fence)
        };
        Ok(CmdBuf {
            cmd_buf,
            fence,
            resources: Vec::new(),
            session: Arc::downgrade(&self.0),
        })
    }

    fn poll_cleanup(&self) {
        let mut pending = self.0.pending.lock().unwrap();
        unsafe {
            let mut i = 0;
            while i < pending.len() {
                if let Ok(true) = self.0.device.get_fence_status(&mut pending[i].fence) {
                    let mut item = pending.swap_remove(i);
                    // TODO: wait is superfluous, can just reset
                    let _ = self.0.device.wait_and_reset(vec![&mut item.fence]);
                    self.0.cleanup_submitted_cmd_buf(item);
                } else {
                    i += 1;
                }
            }
        }
    }

    /// Run a command buffer.
    ///
    /// The semaphores are for swapchain presentation and can be empty for
    /// compute-only work. When provided, work is synchronized to start only
    /// when the wait semaphores are signaled, and when work is complete, the
    /// signal semaphores are signaled.
    pub unsafe fn run_cmd_buf(
        &self,
        mut cmd_buf: CmdBuf,
        wait_semaphores: &[&Semaphore],
        signal_semaphores: &[&Semaphore],
    ) -> Result<SubmittedCmdBuf, Error> {
        // Again, SmallVec here?
        let mut cmd_bufs = Vec::with_capacity(2);
        let mut staging_cmd_buf = self.0.staging_cmd_buf.lock().unwrap().take();
        if let Some(staging) = &mut staging_cmd_buf {
            // With finer grained resource tracking, we might be able to avoid this in
            // some cases.
            staging.memory_barrier();
            staging.finish();
            cmd_bufs.push(&staging.cmd_buf);
        }
        cmd_bufs.push(&cmd_buf.cmd_buf);
        self.0.device.run_cmd_bufs(
            &cmd_bufs,
            wait_semaphores,
            signal_semaphores,
            Some(&mut cmd_buf.fence),
        )?;
        Ok(SubmittedCmdBuf(
            Some(SubmittedCmdBufInner {
                cmd_buf: cmd_buf.cmd_buf,
                fence: cmd_buf.fence,
                resources: cmd_buf.resources,
                staging_cmd_buf,
            }),
            cmd_buf.session,
        ))
    }

    /// Create a buffer.
    ///
    /// The `usage` flags must be specified to indicate what the buffer will
    /// be used for. In general, when no `MAP_` flags are specified, the buffer
    /// will be created in device memory, which means they are not host
    /// accessible, but GPU access is much higher performance (at least on
    /// discrete GPUs).
    pub fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Buffer, Error> {
        let buffer = self.0.device.create_buffer(size, usage)?;
        Ok(Buffer(Arc::new(BufferInner {
            buffer,
            session: Arc::downgrade(&self.0),
        })))
    }

    /// Create a buffer with initialized data.
    ///
    /// This method takes care of creating a staging buffer if needed, so
    /// it is not necessary to specify `MAP_WRITE` usage, unless of course
    /// the buffer will subsequently be written by the host.
    pub fn create_buffer_init(
        &self,
        contents: &[impl PlainData],
        usage: BufferUsage,
    ) -> Result<Buffer, Error> {
        unsafe {
            self.create_buffer_init_raw(
                contents.as_ptr() as *const u8,
                std::mem::size_of_val(contents).try_into()?,
                usage,
            )
        }
    }

    /// Create a buffer with initialized data, from a raw pointer memory region.
    pub unsafe fn create_buffer_init_raw(
        &self,
        contents: *const u8,
        size: u64,
        usage: BufferUsage,
    ) -> Result<Buffer, Error> {
        let use_staging_buffer = !usage.intersects(BufferUsage::MAP_READ | BufferUsage::MAP_WRITE)
            && self.gpu_info().use_staging_buffers;
        let create_usage = if use_staging_buffer {
            BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC
        } else {
            usage | BufferUsage::MAP_WRITE
        };
        let create_buf = self.create_buffer(size, create_usage)?;
        self.0
            .device
            .write_buffer(&create_buf.mux_buffer(), contents, 0, size)?;
        if use_staging_buffer {
            let buf = self.create_buffer(size, usage | BufferUsage::COPY_DST)?;
            let mut staging_cmd_buf = self.0.staging_cmd_buf.lock().unwrap();
            if staging_cmd_buf.is_none() {
                let mut cmd_buf = self.cmd_buf()?;
                cmd_buf.begin();
                *staging_cmd_buf = Some(cmd_buf);
            }
            let staging_cmd_buf = staging_cmd_buf.as_mut().unwrap();
            // This will ensure the staging buffer is deallocated.
            staging_cmd_buf.copy_buffer(&create_buf, &buf);
            staging_cmd_buf.add_resource(create_buf);
            Ok(buf)
        } else {
            Ok(create_buf)
        }
    }

    /// Create an image.
    ///
    /// Currently this creates only a 2D image in RGBA8 format, with usage
    /// so that it can be accessed by shaders and used for transfer.
    pub unsafe fn create_image2d(&self, width: u32, height: u32) -> Result<Image, Error> {
        let image = self.0.device.create_image2d(width, height)?;
        Ok(Image(Arc::new(ImageInner {
            image,
            session: Arc::downgrade(&self.0),
        })))
    }

    /// Create a semaphore.
    ///
    /// These "semaphores" are only for swapchain integration and may be
    /// stubs on back-ends that don't require semaphore synchronization.
    pub unsafe fn create_semaphore(&self) -> Result<Semaphore, Error> {
        self.0.device.create_semaphore()
    }

    /// This creates a pipeline that operates on some buffers and images.
    ///
    /// The descriptor set layout is just some number of storage buffers
    /// and storage images (this might change).
    pub unsafe fn create_simple_compute_pipeline<'a>(
        &self,
        code: ShaderCode<'a>,
        n_buffers: u32,
    ) -> Result<Pipeline, Error> {
        self.pipeline_builder()
            .add_buffers(n_buffers)
            .create_compute_pipeline(self, code)
    }

    /// Start building a pipeline.
    ///
    /// A pipeline is essentially a compiled shader, with more specific
    /// details about what resources may be bound to it.
    pub unsafe fn pipeline_builder(&self) -> PipelineBuilder {
        PipelineBuilder(self.0.device.pipeline_builder())
    }

    /// Create a descriptor set for a simple pipeline that just references buffers.
    pub unsafe fn create_simple_descriptor_set<'a>(
        &self,
        pipeline: &Pipeline,
        buffers: impl IntoRefs<'a, Buffer>,
    ) -> Result<DescriptorSet, Error> {
        self.descriptor_set_builder()
            .add_buffers(buffers)
            .build(self, pipeline)
    }

    /// Start building a descriptor set.
    ///
    /// A descriptor set is a binding of actual resources (buffers and
    /// images) to slots as specified in the pipeline.
    pub unsafe fn descriptor_set_builder(&self) -> DescriptorSetBuilder {
        DescriptorSetBuilder(self.0.device.descriptor_set_builder())
    }

    /// Create a query pool for timestamp queries.
    pub fn create_query_pool(&self, n_queries: u32) -> Result<QueryPool, Error> {
        self.0.device.create_query_pool(n_queries)
    }

    /// Fetch the contents of the query pool.
    ///
    /// This should be called after waiting on the command buffer that wrote the
    /// timer queries.
    pub unsafe fn fetch_query_pool(&self, pool: &QueryPool) -> Result<Vec<f64>, Error> {
        self.0.device.fetch_query_pool(pool)
    }

    #[doc(hidden)]
    /// Create a sampler.
    ///
    /// Noy yet implemented.
    pub unsafe fn create_sampler(&self, params: SamplerParams) -> Result<Sampler, Error> {
        todo!()
        //self.0.device.create_sampler(params)
    }

    /// Query the GPU info.
    pub fn gpu_info(&self) -> &GpuInfo {
        &self.0.gpu_info
    }

    /// Choose shader code from the available choices.
    pub fn choose_shader<'a>(&self, spv: &'a [u8], hlsl: &'a str, msl: &'a str) -> ShaderCode<'a> {
        self.0.device.choose_shader(spv, hlsl, msl)
    }
}

impl SessionInner {
    /// Clean up a submitted command buffer.
    ///
    /// This drops the resources used by the command buffer and also cleans up the command
    /// buffer itself. Currently that means destroying it, but at some point we'll want to
    /// be better at reuse.
    unsafe fn cleanup_submitted_cmd_buf(&self, item: SubmittedCmdBufInner) {
        let _should_handle_err = self.device.destroy_cmd_buf(item.cmd_buf);
        let _should_handle_err = self.device.destroy_fence(item.fence);

        std::mem::drop(item.resources);
        if let Some(staging_cmd_buf) = item.staging_cmd_buf {
            let _should_handle_err = self.device.destroy_cmd_buf(staging_cmd_buf.cmd_buf);
            let _should_handle_err = self.device.destroy_fence(staging_cmd_buf.fence);
            std::mem::drop(staging_cmd_buf.resources);
        }
    }
}

impl CmdBuf {
    /// Begin recording into a command buffer.
    ///
    /// Always call this before encoding any actual work.
    ///
    /// Discussion question: can this be subsumed?
    pub unsafe fn begin(&mut self) {
        self.cmd_buf.begin();
    }

    /// Finish recording into a command buffer.
    ///
    /// Always call this as the last method before submitting the command
    /// buffer.
    pub unsafe fn finish(&mut self) {
        self.cmd_buf.finish();
    }

    /// Dispatch a compute shader.
    ///
    /// Request a compute shader to be run, using the pipeline to specify the
    /// code, and the descriptor set to address the resources read and written.
    ///
    /// Both the workgroup count (number of workgroups) and the workgroup size
    /// (number of threads in a workgroup) must be specified here, though not
    /// all back-ends require the latter info.
    pub unsafe fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        descriptor_set: &DescriptorSet,
        workgroup_count: (u32, u32, u32),
        workgroup_size: (u32, u32, u32),
    ) {
        self.cmd_buf
            .dispatch(pipeline, descriptor_set, workgroup_count, workgroup_size);
    }

    /// Insert an execution and memory barrier.
    ///
    /// Compute kernels (and other actions) after this barrier may read from buffers
    /// that were written before this barrier.
    pub unsafe fn memory_barrier(&mut self) {
        self.cmd_buf.memory_barrier();
    }

    /// Insert a barrier for host access to buffers.
    ///
    /// The host may read buffers written before this barrier, after the fence for
    /// the command buffer is signaled.
    ///
    /// See http://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/
    /// ("Host memory reads") for an explanation of this barrier.
    pub unsafe fn host_barrier(&mut self) {
        self.cmd_buf.memory_barrier();
    }

    /// Insert an image barrier, transitioning image layout.
    ///
    /// When an image is written by one command and then read by another, an image
    /// barrier must separate the uses. Also, the image layout must match the use
    /// of the image.
    ///
    /// Additionally, when writing to an image for the first time, it must be
    /// transitioned from an unknown layout to specify the layout.
    pub unsafe fn image_barrier(
        &mut self,
        image: &Image,
        src_layout: ImageLayout,
        dst_layout: ImageLayout,
    ) {
        self.cmd_buf
            .image_barrier(image.mux_image(), src_layout, dst_layout);
    }

    /// Clear the buffer.
    ///
    /// When the size is not specified, it clears the whole buffer.
    pub unsafe fn clear_buffer(&mut self, buffer: &Buffer, size: Option<u64>) {
        self.cmd_buf.clear_buffer(buffer.mux_buffer(), size);
    }

    /// Copy one buffer to another.
    ///
    /// When the buffers differ in size, the minimum of the sizes is used.
    pub unsafe fn copy_buffer(&mut self, src: &Buffer, dst: &Buffer) {
        self.cmd_buf.copy_buffer(src.mux_buffer(), dst.mux_buffer());
    }

    /// Copy an image to a buffer.
    ///
    /// The size of the image and buffer must match.
    pub unsafe fn copy_image_to_buffer(&mut self, src: &Image, dst: &Buffer) {
        self.cmd_buf
            .copy_image_to_buffer(src.mux_image(), dst.mux_buffer());
        // TODO: change the backend signature to allow failure, as in "not
        // implemented" or "unaligned", and fall back to compute shader
        // submission.
    }

    /// Copy a buffer to an image.
    ///
    /// The size of the image and buffer must match.
    pub unsafe fn copy_buffer_to_image(&mut self, src: &Buffer, dst: &Image) {
        self.cmd_buf
            .copy_buffer_to_image(src.mux_buffer(), dst.mux_image());
        // See above.
    }

    /// Copy an image to another.
    ///
    /// This is especially useful for writing to the swapchain image, as in
    /// general that can't be bound to a compute shader.
    ///
    /// Discussion question: we might have a specialized version of this
    /// function for copying to the swapchain image, and a separate type.
    pub unsafe fn blit_image(&mut self, src: &Image, dst: &Image) {
        self.cmd_buf.blit_image(src.mux_image(), dst.mux_image());
    }

    /// Reset the query pool.
    ///
    /// The query pool must be reset before each use, to avoid validation errors.
    /// This is annoying, and we could tweak the API to make it implicit, doing
    /// the reset before the first timestamp write.
    pub unsafe fn reset_query_pool(&mut self, pool: &QueryPool) {
        self.cmd_buf.reset_query_pool(pool);
    }

    /// Write a timestamp.
    ///
    /// The query index must be less than the size of the query pool on creation.
    pub unsafe fn write_timestamp(&mut self, pool: &QueryPool, query: u32) {
        self.cmd_buf.write_timestamp(pool, query);
    }

    /// Prepare the timestamps for reading. This isn't required on Vulkan but
    /// is required on (at least) DX12.
    ///
    /// It's possible we'll make this go away, by implicitly including it
    /// on command buffer submission when a query pool has been written.
    pub unsafe fn finish_timestamps(&mut self, pool: &QueryPool) {
        self.cmd_buf.finish_timestamps(pool);
    }

    /// Make sure the resource lives until the command buffer completes.
    ///
    /// The submitted command buffer will hold this reference until the corresponding
    /// fence is signaled.
    ///
    /// There are two choices for upholding the lifetime invariant: this function, or
    /// the caller can manually hold the reference. The latter is appropriate when it's
    /// part of retained state.
    pub fn add_resource(&mut self, resource: impl Into<RetainResource>) {
        self.resources.push(resource.into());
    }
}

impl SubmittedCmdBuf {
    /// Wait for the work to complete.
    ///
    /// After calling this function, buffers written by the command buffer
    /// can be read (assuming they were created with `MAP_READ` usage and also
    /// that a host barrier was placed in the command list).
    ///
    /// Further, resources referenced by the command list may be destroyed or
    /// reused; it is a safety violation to do so beforehand.
    ///
    /// Resources for which destruction was deferred through
    /// [`add_resource`][`CmdBuf::add_resource`] will actually be dropped here.
    pub fn wait(mut self) -> Result<(), Error> {
        let mut item = self.0.take().unwrap();
        if let Some(session) = Weak::upgrade(&self.1) {
            unsafe {
                session.device.wait_and_reset(vec![&mut item.fence])?;
                session.cleanup_submitted_cmd_buf(item);
            }
        }
        // else session dropped error?
        Ok(())
    }
}

impl Drop for SubmittedCmdBuf {
    fn drop(&mut self) {
        if let Some(inner) = self.0.take() {
            if let Some(session) = Weak::upgrade(&self.1) {
                session.pending.lock().unwrap().push(inner);
            }
        }
    }
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        if let Some(session) = Weak::upgrade(&self.session) {
            unsafe {
                let _ = session.device.destroy_buffer(&self.buffer);
            }
        }
    }
}

impl Drop for ImageInner {
    fn drop(&mut self) {
        if let Some(session) = Weak::upgrade(&self.session) {
            unsafe {
                let _ = session.device.destroy_image(&self.image);
            }
        }
    }
}

impl Image {
    /// Get a lower level image handle.
    pub(crate) fn mux_image(&self) -> &mux::Image {
        &self.0.image
    }

    /// Wrap a swapchain image so it can be exported to the hub level.
    /// Swapchain images don't need resource tracking (or at least we
    /// don't do it), so no session ref is needed.
    pub(crate) fn wrap_swapchain_image(image: mux::Image) -> Image {
        Image(Arc::new(ImageInner {
            image,
            session: Weak::new(),
        }))
    }
}

impl Buffer {
    /// Get a lower level buffer handle.
    pub(crate) fn mux_buffer(&self) -> &mux::Buffer {
        &self.0.buffer
    }

    /// Write the buffer contents.
    ///
    /// The buffer must have been created with `MAP_WRITE` usage, and with
    /// a size large enough to accommodate the given slice.
    pub unsafe fn write<T: PlainData>(&mut self, contents: &[T]) -> Result<(), Error> {
        if let Some(session) = Weak::upgrade(&self.0.session) {
            session.device.write_buffer(
                &self.0.buffer,
                contents.as_ptr() as *const u8,
                0,
                std::mem::size_of_val(contents).try_into()?,
            )?;
        }
        // else session lost error?
        Ok(())
    }

    /// Read the buffer contents.
    ///
    /// The buffer must have been created with `MAP_READ` usage. The caller
    /// is also responsible for ensuring that this does not read uninitialized
    /// memory.
    pub unsafe fn read<T: PlainData>(&self, result: &mut Vec<T>) -> Result<(), Error> {
        let size = self.mux_buffer().size();
        let len = size as usize / std::mem::size_of::<T>();
        if len > result.len() {
            result.reserve(len - result.len());
        }
        if let Some(session) = Weak::upgrade(&self.0.session) {
            session
                .device
                .read_buffer(&self.0.buffer, result.as_mut_ptr() as *mut u8, 0, size)?;
            result.set_len(len);
        }
        // else session lost error?
        Ok(())
    }

    /// The size of the buffer.
    ///
    /// This is at least as large as the value provided on creation.
    pub fn size(&self) -> u64 {
        self.0.buffer.size()
    }
}

impl PipelineBuilder {
    /// Add buffers to the pipeline. Each has its own binding.
    pub fn add_buffers(mut self, n_buffers: u32) -> Self {
        self.0.add_buffers(n_buffers);
        self
    }

    /// Add storage images to the pipeline. Each has its own binding.
    pub fn add_images(mut self, n_images: u32) -> Self {
        self.0.add_images(n_images);
        self
    }

    /// Add a binding with a variable-size array of textures.
    pub fn add_textures(mut self, max_textures: u32) -> Self {
        self.0.add_textures(max_textures);
        self
    }

    /// Create the compute pipeline.
    ///
    /// The shader code must be given in an appropriate format for
    /// the back-end. See [`Session::choose_shader`] for a helper.
    pub unsafe fn create_compute_pipeline<'a>(
        self,
        session: &Session,
        code: ShaderCode<'a>,
    ) -> Result<Pipeline, Error> {
        self.0.create_compute_pipeline(&session.0.device, code)
    }
}

impl DescriptorSetBuilder {
    pub fn add_buffers<'a>(mut self, buffers: impl IntoRefs<'a, Buffer>) -> Self {
        let mux_buffers = buffers
            .into_refs()
            .map(|b| b.mux_buffer())
            .collect::<SmallVec<[_; 8]>>();
        self.0.add_buffers(&mux_buffers);
        self
    }

    pub fn add_images<'a>(mut self, images: impl IntoRefs<'a, Image>) -> Self {
        let mux_images = images
            .into_refs()
            .map(|i| i.mux_image())
            .collect::<Vec<_>>();
        self.0.add_images(&mux_images);
        self
    }

    pub fn add_textures<'a>(mut self, images: impl IntoRefs<'a, Image>) -> Self {
        let mux_images = images
            .into_refs()
            .map(|i| i.mux_image())
            .collect::<Vec<_>>();
        self.0.add_textures(&mux_images);
        self
    }

    pub unsafe fn build(
        self,
        session: &Session,
        pipeline: &Pipeline,
    ) -> Result<DescriptorSet, Error> {
        self.0.build(&session.0.device, pipeline)
    }
}

// This lets us use either a slice or a vector. The type is clunky but it
// seems fine enough to use.
pub trait IntoRefs<'a, T: 'a> {
    type Iterator: Iterator<Item = &'a T>;

    fn into_refs(self) -> Self::Iterator;
}

impl<'a, T> IntoRefs<'a, T> for &'a [T] {
    type Iterator = std::slice::Iter<'a, T>;
    fn into_refs(self) -> Self::Iterator {
        self.into_iter()
    }
}

impl<'a, T> IntoRefs<'a, T> for &'a [&'a T] {
    type Iterator = std::iter::Copied<std::slice::Iter<'a, &'a T>>;
    fn into_refs(self) -> Self::Iterator {
        self.into_iter().copied()
    }
}

impl<'a, T, const N: usize> IntoRefs<'a, T> for &'a [&'a T; N] {
    type Iterator = std::iter::Copied<std::slice::Iter<'a, &'a T>>;
    fn into_refs(self) -> Self::Iterator {
        self.into_iter().copied()
    }
}

impl<'a, T> IntoRefs<'a, T> for Vec<&'a T> {
    type Iterator = std::vec::IntoIter<&'a T>;
    fn into_refs(self) -> Self::Iterator {
        self.into_iter()
    }
}

impl From<Buffer> for RetainResource {
    fn from(buf: Buffer) -> Self {
        RetainResource::Buffer(buf)
    }
}

impl From<Image> for RetainResource {
    fn from(img: Image) -> Self {
        RetainResource::Image(img)
    }
}

impl<'a, T: Clone + Into<RetainResource>> From<&'a T> for RetainResource {
    fn from(resource: &'a T) -> Self {
        resource.clone().into()
    }
}
