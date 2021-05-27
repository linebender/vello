// Copyright 2021 The piet-gpu authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

use crate::{BufferUsage, Error};

pub struct MtlInstance;

pub struct MtlDevice {
    device: metal::Device,
}

pub struct MtlSurface;

pub struct MtlSwapchain;

pub struct Buffer {
    buffer: metal::Buffer,
    pub(crate) size: u64,
}

pub struct Image;

pub struct Pipeline;

pub struct DescriptorSet;

pub struct Fence;

pub struct Semaphore;

pub struct CmdBuf;

pub struct QueryPool;

pub struct PipelineBuilder;

pub struct DescriptorSetBuilder;

impl MtlInstance {
    pub fn new(
        window_handle: Option<&dyn raw_window_handle::HasRawWindowHandle>,
    ) -> Result<(MtlInstance, Option<MtlSurface>), Error> {
        Ok((MtlInstance, None))
    }

    // TODO might do some enumeration of devices

    pub fn device(&self, surface: Option<&MtlSurface>) -> Result<MtlDevice, Error> {
        if let Some(device) = metal::Device::system_default() {
            Ok(MtlDevice { device })
        } else {
            Err("can't create system default Metal device".into())
        }
    }

    pub unsafe fn swapchain(
        &self,
        width: usize,
        height: usize,
        device: &MtlDevice,
        surface: &MtlSurface,
    ) -> Result<MtlSwapchain, Error> {
        todo!()
    }
}

impl crate::Device for MtlDevice {
    type Buffer = Buffer;

    type Image = Image;

    type Pipeline = Pipeline;

    type DescriptorSet = DescriptorSet;

    type QueryPool = QueryPool;

    type CmdBuf = CmdBuf;

    type Fence = Fence;

    type Semaphore = Semaphore;

    type PipelineBuilder = PipelineBuilder;

    type DescriptorSetBuilder = DescriptorSetBuilder;

    type Sampler = ();

    type ShaderSource = str;

    fn query_gpu_info(&self) -> crate::GpuInfo {
        todo!()
    }

    fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Self::Buffer, Error> {
        let options = if usage.contains(BufferUsage::MAP_READ) {
            metal::MTLResourceOptions::StorageModeShared | metal::MTLResourceOptions::CPUCacheModeDefaultCache
        } else if usage.contains(BufferUsage::MAP_WRITE) {
            metal::MTLResourceOptions::StorageModeShared | metal::MTLResourceOptions::CPUCacheModeWriteCombined
        } else {
            metal::MTLResourceOptions::StorageModePrivate
        };
        let buffer = self.device.new_buffer(size, options);
        Ok(Buffer { buffer, size })
    }

    unsafe fn destroy_buffer(&self, buffer: &Self::Buffer) -> Result<(), Error> {
        todo!()
    }

    unsafe fn create_image2d(
        &self,
        width: u32,
        height: u32,
    ) -> Result<Self::Image, Error> {
        todo!()
    }

    unsafe fn destroy_image(&self, image: &Self::Image) -> Result<(), Error> {
        todo!()
    }

    unsafe fn pipeline_builder(&self) -> Self::PipelineBuilder {
        todo!()
    }

    unsafe fn descriptor_set_builder(&self) -> Self::DescriptorSetBuilder {
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

    unsafe fn run_cmd_bufs(
        &self,
        cmd_bufs: &[&Self::CmdBuf],
        wait_semaphores: &[&Self::Semaphore],
        signal_semaphores: &[&Self::Semaphore],
        fence: Option<&Self::Fence>,
    ) -> Result<(), Error> {
        todo!()
    }

    unsafe fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        dst: *mut u8,
        offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        let contents_ptr = buffer.buffer.contents();
        if contents_ptr.is_null() {
            return Err("probably trying to read from private buffer".into());
        }
        std::ptr::copy_nonoverlapping((contents_ptr as *const u8).add(offset as usize), dst, size as usize);
        Ok(())
    }

    unsafe fn write_buffer(
        &self,
        buffer: &Buffer,
        contents: *const u8,
        offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        let contents_ptr = buffer.buffer.contents();
        if contents_ptr.is_null() {
            return Err("probably trying to write to private buffer".into());
        }
        std::ptr::copy_nonoverlapping(contents, (contents_ptr as *mut u8).add(offset as usize), size as usize);
        Ok(())
    }

    unsafe fn create_semaphore(&self) -> Result<Self::Semaphore, Error> {
        todo!()
    }

    unsafe fn create_fence(&self, signaled: bool) -> Result<Self::Fence, Error> {
        todo!()
    }

    unsafe fn wait_and_reset(&self, fences: &[&Self::Fence]) -> Result<(), Error> {
        todo!()
    }

    unsafe fn get_fence_status(&self, fence: &Self::Fence) -> Result<bool, Error> {
        todo!()
    }

    unsafe fn create_sampler(&self, params: crate::SamplerParams) -> Result<Self::Sampler, Error> {
        todo!()
    }
}

impl crate::CmdBuf<MtlDevice> for CmdBuf {
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

    unsafe fn clear_buffer(&self, buffer: &Buffer, size: Option<u64>) {
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

impl crate::PipelineBuilder<MtlDevice> for PipelineBuilder {
    fn add_buffers(&mut self, n_buffers: u32) {
        todo!()
    }

    fn add_images(&mut self, n_images: u32) {
        todo!()
    }

    fn add_textures(&mut self, max_textures: u32) {
        todo!()
    }

    unsafe fn create_compute_pipeline(self, device: &MtlDevice, code: &str) -> Result<Pipeline, Error> {
        todo!()
    }
}

impl crate::DescriptorSetBuilder<MtlDevice> for DescriptorSetBuilder {
    fn add_buffers(&mut self, buffers: &[&Buffer]) {
        todo!()
    }

    fn add_images(&mut self, images: &[&Image]) {
        todo!()
    }

    fn add_textures(&mut self, images: &[&Image]) {
        todo!()
    }

    unsafe fn build(self, device: &MtlDevice, pipeline: &Pipeline) -> Result<DescriptorSet, Error> {
        todo!()
    }
}

impl MtlSwapchain {
    pub unsafe fn next(&mut self) -> Result<(usize, Semaphore), Error> {
        todo!()
    }

    pub unsafe fn image(&self, idx: usize) -> Image {
        todo!()
    }

    pub unsafe fn present(
        &self,
        image_idx: usize,
        semaphores: &[&Semaphore],
    ) -> Result<bool, Error> {
        todo!()
    }    
}