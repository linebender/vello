//! A convenience layer on top of raw hal.
//!
//! This layer takes care of some lifetime and synchronization bookkeeping.
//! It is likely that it will also take care of compile time and runtime
//! negotiation of backends (Vulkan, DX12), but right now it's Vulkan-only.

use std::any::Any;
use std::sync::{Arc, Mutex, Weak};

use crate::vulkan;
use crate::DescriptorSetBuilder as DescriptorSetBuilderTrait;
use crate::PipelineBuilder as PipelineBuilderTrait;
use crate::{Device, Error, SamplerParams};

pub type MemFlags = <vulkan::VkDevice as Device>::MemFlags;
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
}

pub struct CmdBuf {
    cmd_buf: vulkan::CmdBuf,
    fence: Fence,
    resources: Vec<Box<dyn Any>>,
    session: Weak<SessionInner>,
}

// Maybe "pending" is a better name?
pub struct SubmittedCmdBuf(Option<SubmittedCmdBufInner>, Weak<SessionInner>);

struct SubmittedCmdBufInner {
    cmd_buf: vulkan::CmdBuf,
    fence: Fence,
    resources: Vec<Box<dyn Any>>,
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

impl Session {
    pub fn new(device: vulkan::VkDevice) -> Session {
        Session(Arc::new(SessionInner {
            device,
            cmd_buf_pool: Default::default(),
            pending: Default::default(),
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
                    self.0
                        .cmd_buf_pool
                        .lock()
                        .unwrap()
                        .push((item.cmd_buf, item.fence));
                    std::mem::drop(item.resources);
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
        self.0.device.run_cmd_buf(
            &cmd_buf.cmd_buf,
            wait_semaphores,
            signal_semaphores,
            Some(&cmd_buf.fence),
        )?;
        Ok(SubmittedCmdBuf(
            Some(SubmittedCmdBufInner {
                cmd_buf: cmd_buf.cmd_buf,
                fence: cmd_buf.fence,
                resources: cmd_buf.resources,
            }),
            cmd_buf.session,
        ))
    }

    pub fn create_buffer(&self, size: u64, mem_flags: MemFlags) -> Result<Buffer, Error> {
        let buffer = self.0.device.create_buffer(size, mem_flags)?;
        Ok(Buffer(Arc::new(BufferInner {
            buffer,
            session: Arc::downgrade(&self.0),
        })))
    }

    pub unsafe fn create_image2d(
        &self,
        width: u32,
        height: u32,
        mem_flags: MemFlags,
    ) -> Result<Image, Error> {
        let image = self.0.device.create_image2d(width, height, mem_flags)?;
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
    pub fn add_resource<T: Clone + 'static>(&mut self, resource: &T) {
        self.resources.push(Box::new(resource.clone()));
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

    pub unsafe fn write<T: Sized>(&mut self, contents: &[T]) -> Result<(), Error> {
        if let Some(session) = Weak::upgrade(&self.0.session) {
            session.device.write_buffer(&self.0.buffer, contents)?;
        }
        // else session lost error?
        Ok(())
    }
    pub unsafe fn read<T: Sized>(&self, result: &mut Vec<T>) -> Result<(), Error> {
        if let Some(session) = Weak::upgrade(&self.0.session) {
            session.device.read_buffer(&self.0.buffer, result)?;
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

// TODO: this will benefit from const generics!
impl<'a, T> IntoRefs<'a, T> for &'a [&'a T; 1] {
    type Iterator = std::iter::Copied<std::slice::Iter<'a, &'a T>>;
    fn into_refs(self) -> Self::Iterator {
        self.into_iter().copied()
    }
}

impl<'a, T> IntoRefs<'a, T> for &'a [&'a T; 2] {
    type Iterator = std::iter::Copied<std::slice::Iter<'a, &'a T>>;
    fn into_refs(self) -> Self::Iterator {
        self.into_iter().copied()
    }
}

impl<'a, T> IntoRefs<'a, T> for &'a [&'a T; 3] {
    type Iterator = std::iter::Copied<std::slice::Iter<'a, &'a T>>;
    fn into_refs(self) -> Self::Iterator {
        self.into_iter().copied()
    }
}

impl<'a, T> IntoRefs<'a, T> for &'a [&'a T; 4] {
    type Iterator = std::iter::Copied<std::slice::Iter<'a, &'a T>>;
    fn into_refs(self) -> Self::Iterator {
        self.into_iter().copied()
    }
}

impl<'a, T> IntoRefs<'a, T> for &'a [&'a T; 5] {
    type Iterator = std::iter::Copied<std::slice::Iter<'a, &'a T>>;
    fn into_refs(self) -> Self::Iterator {
        self.into_iter().copied()
    }
}

impl<'a, T> IntoRefs<'a, T> for &'a [&'a T; 6] {
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
