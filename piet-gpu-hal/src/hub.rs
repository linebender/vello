//! A convenience layer on top of raw hal.
//!
//! This layer takes care of some lifetime and synchronization bookkeeping.
//! It is likely that it will also take care of compile time and runtime
//! negotiation of backends (Vulkan, DX12), but right now it's Vulkan-only.

use std::any::Any;
use std::sync::{Arc, Mutex, Weak};

use crate::vulkan;
use crate::{Device, Error};

pub type MemFlags = <vulkan::VkDevice as Device>::MemFlags;
pub type Semaphore = <vulkan::VkDevice as Device>::Semaphore;
pub type Pipeline = <vulkan::VkDevice as Device>::Pipeline;
pub type DescriptorSet = <vulkan::VkDevice as Device>::DescriptorSet;
pub type QueryPool = <vulkan::VkDevice as Device>::QueryPool;

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

    /// This creates a pipeline that runs over the buffer.
    ///
    /// The descriptor set layout is just some number of storage buffers and storage images (this might change).
    pub unsafe fn create_simple_compute_pipeline(
        &self,
        code: &[u8],
        n_buffers: u32,
        n_images: u32,
    ) -> Result<Pipeline, Error> {
        self.0
            .device
            .create_simple_compute_pipeline(code, n_buffers, n_images)
    }

    /// Create a descriptor set for a simple pipeline that just references buffers and images.
    ///
    /// Note: when we do portability, the signature will change to not reference the Vulkan types
    /// directly.
    pub unsafe fn create_descriptor_set(
        &self,
        pipeline: &Pipeline,
        bufs: &[&vulkan::Buffer],
        images: &[&vulkan::Image],
    ) -> Result<DescriptorSet, Error> {
        self.0.device.create_descriptor_set(pipeline, bufs, images)
    }

    /// Create a query pool for timestamp queries.
    pub fn create_query_pool(&self, n_queries: u32) -> Result<QueryPool, Error> {
        self.0.device.create_query_pool(n_queries)
    }

    pub unsafe fn fetch_query_pool(&self, pool: &QueryPool) -> Result<Vec<f64>, Error> {
        self.0.device.fetch_query_pool(pool)
    }
}

impl CmdBuf {
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
