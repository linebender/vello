//! DX12 implemenation of HAL trait.

mod error;
mod wrappers;

use winapi::shared::dxgi1_3;
use winapi::um::d3d12;

use crate::{Error};

use self::wrappers::{CommandQueue, Device, Factory4, Resource};

pub struct Dx12Instance {
    factory: Factory4,
}

pub struct Dx12Device {
    device: Device,
    command_queue: CommandQueue,
}

pub struct Buffer {
    resource: Resource,
}

pub struct Image {
    resource: Resource,
}

#[derive(Clone, Copy)]
pub enum MemFlags {
    DeviceLocal,
    HostCoherent,
}

pub struct CmdBuf(CommandQueue);

pub struct Pipeline;

pub struct DescriptorSet;

pub struct QueryPool;

pub struct Fence(wrappers::Fence);

pub struct Semaphore;

impl Dx12Instance {
    /// Create a new instance.
    ///
    /// TODO: take a raw window handle.
    /// TODO: can probably be a trait.
    pub fn new() -> Result<Dx12Instance, Error> {
        unsafe {
            #[cfg(debug_assertions)]
            let factory_flags = dxgi1_3::DXGI_CREATE_FACTORY_DEBUG;

            #[cfg(not(debug_assertions))]
            let factory_flags: u32 = 0;

            let factory = Factory4::create(factory_flags)?;
            Ok(Dx12Instance { factory })
        }
    }

    /// Get a device suitable for compute workloads.
    ///
    /// TODO: handle window.
    /// TODO: probably can also be trait'ified.
    pub fn device(&self) -> Result<Dx12Device, Error> {
        unsafe {
            let device = Device::create_device(&self.factory)?;
            let list_type = d3d12::D3D12_COMMAND_LIST_TYPE_DIRECT;
            let command_queue = device.create_command_queue(list_type, 0, d3d12::D3D12_COMMAND_QUEUE_FLAG_NONE, 0)?;
            Ok(Dx12Device { device, command_queue })
        }
    }
}

impl crate::Device for Dx12Device {
    type Buffer = Buffer;

    type Image = Image;

    type MemFlags = MemFlags;

    type Pipeline = Pipeline;

    type DescriptorSet = DescriptorSet;

    type QueryPool = QueryPool;

    type CmdBuf = CmdBuf;

    type Fence = Fence;

    type Semaphore = Semaphore;

    fn create_buffer(&self, size: u64, mem_flags: Self::MemFlags) -> Result<Self::Buffer, Error> {
        todo!()
    }

    unsafe fn destroy_buffer(&self, buffer: &Self::Buffer) -> Result<(), Error> {
        todo!()
    }

    unsafe fn create_image2d(
        &self,
        width: u32,
        height: u32,
        mem_flags: Self::MemFlags,
    ) -> Result<Self::Image, Error> {
        todo!()
    }

    unsafe fn destroy_image(&self, image: &Self::Image) -> Result<(), Error> {
        todo!()
    }

    unsafe fn create_simple_compute_pipeline(
        &self,
        code: &[u8],
        n_buffers: u32,
        n_images: u32,
    ) -> Result<Self::Pipeline, Error> {
        todo!()
    }

    unsafe fn create_descriptor_set(
        &self,
        pipeline: &Self::Pipeline,
        bufs: &[&Self::Buffer],
        images: &[&Self::Image],
    ) -> Result<Self::DescriptorSet, Error> {
        todo!()
    }

    fn create_cmd_buf(&self) -> Result<Self::CmdBuf, Error> {
        todo!()
    }

    fn create_query_pool(&self, n_queries: u32) -> Result<Self::QueryPool, Error> {
        todo!()
    }

    unsafe fn fetch_query_pool(&self, pool: &Self::QueryPool) -> Result<Vec<f64>, Error> {
        todo!()
    }

    unsafe fn run_cmd_buf(
        &self,
        cmd_buf: &Self::CmdBuf,
        wait_semaphores: &[Self::Semaphore],
        signal_semaphores: &[Self::Semaphore],
        fence: Option<&Self::Fence>,
    ) -> Result<(), Error> {
        todo!()
    }

    unsafe fn read_buffer<T: Sized>(
        &self,
        buffer: &Self::Buffer,
        result: &mut Vec<T>,
    ) -> Result<(), Error> {
        todo!()
    }

    unsafe fn write_buffer<T: Sized>(
        &self,
        buffer: &Self::Buffer,
        contents: &[T],
    ) -> Result<(), Error> {
        todo!()
    }

    unsafe fn create_semaphore(&self) -> Result<Self::Semaphore, Error> {
        todo!()
    }

    unsafe fn create_fence(&self, signaled: bool) -> Result<Self::Fence, Error> {
        todo!()
    }

    unsafe fn wait_and_reset(&self, fences: &[Self::Fence]) -> Result<(), Error> {
        todo!()
    }

    unsafe fn get_fence_status(&self, fence: Self::Fence) -> Result<bool, Error> {
        todo!()
    }
}

impl crate::CmdBuf<Dx12Device> for CmdBuf {
    unsafe fn begin(&mut self) {
        todo!()
    }

    unsafe fn finish(&mut self) {
        todo!()
    }

    unsafe fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        descriptor_set: &DescriptorSet,
        size: (u32, u32, u32),
    ) {
        todo!()
    }

    unsafe fn memory_barrier(&mut self) {
        todo!()
    }

    unsafe fn host_barrier(&mut self) {
        todo!()
    }

    unsafe fn image_barrier(
        &mut self,
        image: &Image,
        src_layout: crate::ImageLayout,
        dst_layout: crate::ImageLayout,
    ) {
        todo!()
    }

    unsafe fn clear_buffer(&self, buffer: &Buffer) {
        todo!()
    }

    unsafe fn copy_buffer(&self, src: &Buffer, dst: &Buffer) {
        todo!()
    }

    unsafe fn copy_image_to_buffer(&self, src: &Image, dst: &Buffer) {
        todo!()
    }

    unsafe fn copy_buffer_to_image(&self, src: &Buffer, dst: &Image) {
        todo!()
    }

    unsafe fn blit_image(&self, src: &Image, dst: &Image) {
        todo!()
    }

    unsafe fn reset_query_pool(&mut self, pool: &QueryPool) {
        todo!()
    }

    unsafe fn write_timestamp(&mut self, pool: &QueryPool, query: u32) {
        todo!()
    }
}

impl crate::MemFlags for MemFlags {
    fn device_local() -> Self {
        MemFlags::DeviceLocal
    }

    fn host_coherent() -> Self {
        MemFlags::HostCoherent
    }
}
