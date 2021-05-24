//! A convenience layer on top of raw hal.
//!
//! This layer takes care of some lifetime and synchronization bookkeeping.
//! It is likely that it will also take care of compile time and runtime
//! negotiation of backends (Vulkan, DX12), but right now it's Vulkan-only.

use std::convert::TryInto;
use std::sync::{Arc, Mutex, Weak};

use crate::vulkan;
use crate::CmdBuf as CmdBufTrait;
use crate::DescriptorSetBuilder as DescriptorSetBuilderTrait;
use crate::PipelineBuilder as PipelineBuilderTrait;
use crate::{BufferUsage, Device, Error, GpuInfo, SamplerParams};

pub type Semaphore = <vulkan::VkDevice as Device>::Semaphore;
pub type Pipeline = <vulkan::VkDevice as Device>::Pipeline;
pub type DescriptorSet = <vulkan::VkDevice as Device>::DescriptorSet;
pub type QueryPool = <vulkan::VkDevice as Device>::QueryPool;
pub type Sampler = <vulkan::VkDevice as Device>::Sampler;

type Fence = <vulkan::VkDevice as Device>::Fence;

type VkImage = <vulkan::VkDevice as Device>::Image;
type VkBuffer = <vulkan::VkDevice as Device>::Buffer;

#[derive(Clone)]
pub struct Session(Arc<SessionInner>);

struct SessionInner {
    device: vulkan::VkDevice,
    cmd_buf_pool: Mutex<Vec<(vulkan::CmdBuf, Fence)>>,
    /// Command buffers that are still pending (so resources can't be freed).
    pending: Mutex<Vec<SubmittedCmdBufInner>>,
    /// A command buffer that is used for copying from staging buffers.
    staging_cmd_buf: Mutex<Option<CmdBuf>>,
    gpu_info: GpuInfo,
}

pub struct CmdBuf {
    cmd_buf: vulkan::CmdBuf,
    fence: Fence,
    resources: Vec<RetainResource>,
    session: Weak<SessionInner>,
}

// Maybe "pending" is a better name?
pub struct SubmittedCmdBuf(Option<SubmittedCmdBufInner>, Weak<SessionInner>);

struct SubmittedCmdBufInner {
    // It's inconsistent, cmd_buf is unpacked, staging_cmd_buf isn't. Probably
    // better to chose one or the other.
    cmd_buf: vulkan::CmdBuf,
    fence: Fence,
    resources: Vec<RetainResource>,
    staging_cmd_buf: Option<CmdBuf>,
}

#[derive(Clone)]
pub struct Image(Arc<ImageInner>);

struct ImageInner {
    image: VkImage,
    session: Weak<SessionInner>,
}

#[derive(Clone)]
pub struct Buffer(Arc<BufferInner>);

struct BufferInner {
    buffer: VkBuffer,
    session: Weak<SessionInner>,
}

pub struct PipelineBuilder(vulkan::PipelineBuilder);

pub struct DescriptorSetBuilder(vulkan::DescriptorSetBuilder);

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
    pub fn new(device: vulkan::VkDevice) -> Session {
        let gpu_info = device.query_gpu_info();
        Session(Arc::new(SessionInner {
            device,
            gpu_info,
            cmd_buf_pool: Default::default(),
            pending: Default::default(),
            staging_cmd_buf: Default::default(),
        }))
    }

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
                if let Ok(true) = self.0.device.get_fence_status(pending[i].fence) {
                    let item = pending.swap_remove(i);
                    // TODO: wait is superfluous, can just reset
                    let _ = self.0.device.wait_and_reset(&[item.fence]);
                    let mut pool = self.0.cmd_buf_pool.lock().unwrap();
                    pool.push((item.cmd_buf, item.fence));
                    std::mem::drop(item.resources);
                    if let Some(staging_cmd_buf) = item.staging_cmd_buf {
                        pool.push((staging_cmd_buf.cmd_buf, staging_cmd_buf.fence));
                        std::mem::drop(staging_cmd_buf.resources);
                    }
                } else {
                    i += 1;
                }
            }
        }
    }

    pub unsafe fn run_cmd_buf(
        &self,
        cmd_buf: CmdBuf,
        wait_semaphores: &[Semaphore],
        signal_semaphores: &[Semaphore],
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
            Some(&cmd_buf.fence),
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

    pub fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Buffer, Error> {
        let buffer = self.0.device.create_buffer(size, usage)?;
        Ok(Buffer(Arc::new(BufferInner {
            buffer,
            session: Arc::downgrade(&self.0),
        })))
    }

    /// Create a buffer with initialized data.
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
            .write_buffer(&create_buf.vk_buffer(), contents, 0, size)?;
        if use_staging_buffer {
            let buf = self.create_buffer(size, usage | BufferUsage::COPY_DST)?;
            let mut staging_cmd_buf = self.0.staging_cmd_buf.lock().unwrap();
            if staging_cmd_buf.is_none() {
                let mut cmd_buf = self.cmd_buf()?;
                cmd_buf.begin();
                *staging_cmd_buf = Some(cmd_buf);
            }
            let staging_cmd_buf = staging_cmd_buf.as_mut().unwrap();
            // This will ensure the staging buffer is deallocated. It would be nice to
            staging_cmd_buf.copy_buffer(create_buf.vk_buffer(), buf.vk_buffer());
            staging_cmd_buf.add_resource(create_buf);
            Ok(buf)
        } else {
            Ok(create_buf)
        }
    }

    pub unsafe fn create_image2d(&self, width: u32, height: u32) -> Result<Image, Error> {
        let image = self.0.device.create_image2d(width, height)?;
        Ok(Image(Arc::new(ImageInner {
            image,
            session: Arc::downgrade(&self.0),
        })))
    }

    pub unsafe fn create_semaphore(&self) -> Result<Semaphore, Error> {
        self.0.device.create_semaphore()
    }

    /// This creates a pipeline that operates on some buffers and images.
    ///
    /// The descriptor set layout is just some number of storage buffers and storage images (this might change).
    pub unsafe fn create_simple_compute_pipeline(
        &self,
        code: &[u8],
        n_buffers: u32,
    ) -> Result<Pipeline, Error> {
        self.pipeline_builder()
            .add_buffers(n_buffers)
            .create_compute_pipeline(self, code)
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

    /// Create a query pool for timestamp queries.
    pub fn create_query_pool(&self, n_queries: u32) -> Result<QueryPool, Error> {
        self.0.device.create_query_pool(n_queries)
    }

    pub unsafe fn fetch_query_pool(&self, pool: &QueryPool) -> Result<Vec<f64>, Error> {
        self.0.device.fetch_query_pool(pool)
    }

    pub unsafe fn pipeline_builder(&self) -> PipelineBuilder {
        PipelineBuilder(self.0.device.pipeline_builder())
    }

    pub unsafe fn descriptor_set_builder(&self) -> DescriptorSetBuilder {
        DescriptorSetBuilder(self.0.device.descriptor_set_builder())
    }

    pub unsafe fn create_sampler(&self, params: SamplerParams) -> Result<Sampler, Error> {
        self.0.device.create_sampler(params)
    }

    pub fn gpu_info(&self) -> &GpuInfo {
        &self.0.gpu_info
    }
}

impl CmdBuf {
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
    pub fn wait(mut self) -> Result<(), Error> {
        let item = self.0.take().unwrap();
        if let Some(session) = Weak::upgrade(&self.1) {
            unsafe {
                session.device.wait_and_reset(&[item.fence])?;
            }
            session
                .cmd_buf_pool
                .lock()
                .unwrap()
                .push((item.cmd_buf, item.fence));
            std::mem::drop(item.resources);
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

/// For now, we deref, but for runtime backend switching we'll need to wrap
/// all methods.
impl std::ops::Deref for CmdBuf {
    type Target = vulkan::CmdBuf;
    fn deref(&self) -> &Self::Target {
        &self.cmd_buf
    }
}

impl std::ops::DerefMut for CmdBuf {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cmd_buf
    }
}

impl Image {
    pub fn vk_image(&self) -> &vulkan::Image {
        &self.0.image
    }
}

impl Buffer {
    pub fn vk_buffer(&self) -> &vulkan::Buffer {
        &self.0.buffer
    }

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
    pub unsafe fn read<T: PlainData>(&self, result: &mut Vec<T>) -> Result<(), Error> {
        let size = self.vk_buffer().size;
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

    pub unsafe fn create_compute_pipeline(
        self,
        session: &Session,
        code: &[u8],
    ) -> Result<Pipeline, Error> {
        self.0.create_compute_pipeline(&session.0.device, code)
    }
}

impl DescriptorSetBuilder {
    pub fn add_buffers<'a>(mut self, buffers: impl IntoRefs<'a, Buffer>) -> Self {
        let vk_buffers = buffers
            .into_refs()
            .map(|b| b.vk_buffer())
            .collect::<Vec<_>>();
        self.0.add_buffers(&vk_buffers);
        self
    }

    pub fn add_images<'a>(mut self, images: impl IntoRefs<'a, Image>) -> Self {
        let vk_images = images.into_refs().map(|i| i.vk_image()).collect::<Vec<_>>();
        self.0.add_images(&vk_images);
        self
    }

    pub fn add_textures<'a>(mut self, images: impl IntoRefs<'a, Image>) -> Self {
        let vk_images = images.into_refs().map(|i| i.vk_image()).collect::<Vec<_>>();
        self.0.add_textures(&vk_images);
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

impl<'a, T: Into<RetainResource>> From<&'a T> for RetainResource {
    fn from(resource: &'a T) -> Self {
        resource.clone().into()
    }
}
