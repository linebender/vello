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

//! A multiplexer module that selects a back-end at runtime.

use smallvec::SmallVec;

use crate::dx12;
use crate::vulkan;
use crate::CmdBuf as CmdBufTrait;
use crate::DescriptorSetBuilder as DescriptorSetBuilderTrait;
use crate::Device as DeviceTrait;
use crate::PipelineBuilder as PipelineBuilderTrait;
use crate::{BufferUsage, Error, GpuInfo, ImageLayout};

mux_enum! {
    /// An instance, selected from multiple backends.
    pub enum Instance {
        Vk(vulkan::VkInstance),
        Dx12(dx12::Dx12Instance),
    }
}

mux_enum! {
    /// A device, selected from multiple backends.
    pub enum Device {
        Vk(vulkan::VkDevice),
        Dx12(dx12::Dx12Device),
    }
}

mux_enum! {
    /// A surface, which can apply to one of multiple backends.
    pub enum Surface {
        Vk(vulkan::VkSurface),
        Dx12(dx12::Dx12Surface),
    }
}

mux_device_enum! { Buffer }
mux_device_enum! { Image }
mux_device_enum! { Fence }
mux_device_enum! { Semaphore }
mux_device_enum! { PipelineBuilder }
mux_device_enum! { Pipeline }
mux_device_enum! { DescriptorSetBuilder }
mux_device_enum! { DescriptorSet }
mux_device_enum! { CmdBuf }
mux_device_enum! { QueryPool }

/// The code for a shader, either as source or intermediate representation.
pub enum ShaderCode<'a> {
    Spv(&'a [u8]),
    Hlsl(&'a str),
}

impl Instance {
    pub fn new(
        window_handle: Option<&dyn raw_window_handle::HasRawWindowHandle>,
    ) -> Result<(Instance, Option<Surface>), Error> {
        mux! {
            #[cfg(vk)]
            {
                let result = vulkan::VkInstance::new(window_handle);
                if let Ok((instance, surface)) = result {
                    return Ok((Instance::Vk(instance), surface.map(Surface::Vk)));
                }
            }
        }
        mux! {
            #[cfg(dx12)]
            {
                let result = dx12::Dx12Instance::new(window_handle);
                if let Ok((instance, surface)) = result {
                    return Ok((Instance::Dx12(instance), surface.map(Surface::Dx12)));
                }
            }
        }
        // TODO plumb creation errors through.
        Err("No suitable instances found".into())
    }

    pub unsafe fn device(&self, surface: Option<&Surface>) -> Result<Device, Error> {
        match self {
            Instance::Vk(i) => i.device(surface.map(Surface::vk)).map(Device::Vk),
            Instance::Dx12(i) => i.device(surface.map(Surface::dx12)).map(Device::Dx12),
        }
    }
}

// This is basically re-exporting the backend device trait, and we could do that,
// but not doing so lets us diverge more easily (at the moment, the divergence is
// missing functionality).
impl Device {
    pub fn query_gpu_info(&self) -> GpuInfo {
        match self {
            Device::Vk(d) => d.query_gpu_info(),
            Device::Dx12(d) => d.query_gpu_info(),
        }
    }

    pub fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Buffer, Error> {
        match self {
            Device::Vk(d) => d.create_buffer(size, usage).map(Buffer::Vk),
            Device::Dx12(d) => d.create_buffer(size, usage).map(Buffer::Dx12),
        }
    }

    pub unsafe fn destroy_buffer(&self, buffer: &Buffer) -> Result<(), Error> {
        match self {
            Device::Vk(d) => d.destroy_buffer(buffer.vk()),
            Device::Dx12(d) => d.destroy_buffer(buffer.dx12()),
        }
    }

    pub unsafe fn create_fence(&self, signaled: bool) -> Result<Fence, Error> {
        match self {
            Device::Vk(d) => d.create_fence(signaled).map(Fence::Vk),
            Device::Dx12(d) => d.create_fence(signaled).map(Fence::Dx12),
        }
    }

    pub unsafe fn wait_and_reset(&self, fences: &[&Fence]) -> Result<(), Error> {
        match self {
            Device::Vk(d) => {
                let fences = fences
                    .iter()
                    .copied()
                    .map(Fence::vk)
                    .collect::<SmallVec<[_; 4]>>();
                d.wait_and_reset(&*fences)
            }
            Device::Dx12(d) => {
                let fences = fences
                    .iter()
                    .copied()
                    .map(Fence::dx12)
                    .collect::<SmallVec<[_; 4]>>();
                d.wait_and_reset(&*fences)
            }
            // Probably need to change device trait to accept &Fence
            _ => todo!(),
        }
    }

    pub unsafe fn pipeline_builder(&self) -> PipelineBuilder {
        match self {
            Device::Vk(d) => PipelineBuilder::Vk(d.pipeline_builder()),
            Device::Dx12(d) => PipelineBuilder::Dx12(d.pipeline_builder()),
        }
    }

    pub unsafe fn descriptor_set_builder(&self) -> DescriptorSetBuilder {
        match self {
            Device::Vk(d) => DescriptorSetBuilder::Vk(d.descriptor_set_builder()),
            Device::Dx12(d) => DescriptorSetBuilder::Dx12(d.descriptor_set_builder()),
        }
    }

    pub fn create_cmd_buf(&self) -> Result<CmdBuf, Error> {
        match self {
            Device::Vk(d) => d.create_cmd_buf().map(CmdBuf::Vk),
            Device::Dx12(d) => d.create_cmd_buf().map(CmdBuf::Dx12),
        }
    }

    pub fn create_query_pool(&self, n_queries: u32) -> Result<QueryPool, Error> {
        match self {
            Device::Vk(d) => d.create_query_pool(n_queries).map(QueryPool::Vk),
            Device::Dx12(d) => d.create_query_pool(n_queries).map(QueryPool::Dx12),
        }
    }

    pub unsafe fn fetch_query_pool(&self, pool: &QueryPool) -> Result<Vec<f64>, Error> {
        match self {
            Device::Vk(d) => d.fetch_query_pool(pool.vk()),
            Device::Dx12(d) => d.fetch_query_pool(pool.dx12()),
        }
    }

    pub unsafe fn run_cmd_bufs(
        &self,
        cmd_bufs: &[&CmdBuf],
        wait_semaphores: &[&Semaphore],
        signal_semaphores: &[&Semaphore],
        fence: Option<&Fence>,
    ) -> Result<(), Error> {
        match self {
            Device::Vk(d) => d.run_cmd_bufs(
                &cmd_bufs
                    .iter()
                    .map(|c| c.vk())
                    .collect::<SmallVec<[_; 4]>>(),
                &wait_semaphores.iter().copied().map(Semaphore::vk).collect::<SmallVec<[_; 4]>>(),
                &signal_semaphores.iter().copied().map(Semaphore::vk).collect::<SmallVec<[_; 4]>>(),
                fence.map(Fence::vk),
            ),
            Device::Dx12(d) => d.run_cmd_bufs(
                &cmd_bufs
                    .iter()
                    .map(|c| c.dx12())
                    .collect::<SmallVec<[_; 4]>>(),
                &wait_semaphores.iter().copied().map(Semaphore::dx12).collect::<SmallVec<[_; 4]>>(),
                &signal_semaphores.iter().copied().map(Semaphore::dx12).collect::<SmallVec<[_; 4]>>(),
                fence.map(Fence::dx12),
            ),
        }
    }

    pub unsafe fn read_buffer(
        &self,
        buffer: &Buffer,
        dst: *mut u8,
        offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        match self {
            Device::Vk(d) => d.read_buffer(buffer.vk(), dst, offset, size),
            Device::Dx12(d) => d.read_buffer(buffer.dx12(), dst, offset, size),
        }
    }

    pub unsafe fn write_buffer(
        &self,
        buffer: &Buffer,
        contents: *const u8,
        offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        match self {
            Device::Vk(d) => d.write_buffer(buffer.vk(), contents, offset, size),
            Device::Dx12(d) => d.write_buffer(buffer.dx12(), contents, offset, size),
        }
    }
}

impl PipelineBuilder {
    pub fn add_buffers(&mut self, n_buffers: u32) {
        match self {
            PipelineBuilder::Vk(x) => x.add_buffers(n_buffers),
            PipelineBuilder::Dx12(x) => x.add_buffers(n_buffers),
        }
    }

    pub fn add_images(&mut self, n_buffers: u32) {
        match self {
            PipelineBuilder::Vk(x) => x.add_images(n_buffers),
            PipelineBuilder::Dx12(x) => x.add_images(n_buffers),
        }
    }

    pub fn add_textures(&mut self, n_buffers: u32) {
        match self {
            PipelineBuilder::Vk(x) => x.add_textures(n_buffers),
            PipelineBuilder::Dx12(x) => x.add_textures(n_buffers),
        }
    }

    pub unsafe fn create_compute_pipeline<'a>(
        self,
        device: &Device,
        code: ShaderCode<'a>,
    ) -> Result<Pipeline, Error> {
        match self {
            PipelineBuilder::Vk(x) => {
                let shader_code = match code {
                    ShaderCode::Spv(spv) => spv,
                    // Panic or return "incompatible shader" error here?
                    _ => panic!("Vulkan backend requires shader code in SPIR-V format"),
                };
                x.create_compute_pipeline(device.vk(), shader_code)
                    .map(Pipeline::Vk)
            }
            PipelineBuilder::Dx12(x) => {
                let shader_code = match code {
                    ShaderCode::Hlsl(hlsl) => hlsl,
                    // Panic or return "incompatible shader" error here?
                    _ => panic!("DX12 backend requires shader code in HLSL format"),
                };
                x.create_compute_pipeline(device.dx12(), shader_code)
                    .map(Pipeline::Dx12)
            }
        }
    }
}

impl DescriptorSetBuilder {
    pub fn add_buffers(&mut self, buffers: &[Buffer]) {
        match self {
            DescriptorSetBuilder::Vk(x) => {
                x.add_buffers(&buffers.iter().map(Buffer::vk).collect::<SmallVec<[_; 8]>>())
            }
            DescriptorSetBuilder::Dx12(x) => x.add_buffers(
                &buffers
                    .iter()
                    .map(Buffer::dx12)
                    .collect::<SmallVec<[_; 8]>>(),
            ),
        }
    }

    pub fn add_images(&mut self, images: &[Image]) {
        match self {
            DescriptorSetBuilder::Vk(x) => {
                x.add_images(&images.iter().map(Image::vk).collect::<SmallVec<[_; 8]>>())
            }
            DescriptorSetBuilder::Dx12(x) => {
                x.add_images(&images.iter().map(Image::dx12).collect::<SmallVec<[_; 8]>>())
            }
        }
    }

    pub fn add_textures(&mut self, images: &[Image]) {
        match self {
            DescriptorSetBuilder::Vk(x) => {
                x.add_textures(&images.iter().map(Image::vk).collect::<SmallVec<[_; 8]>>())
            }
            DescriptorSetBuilder::Dx12(x) => {
                x.add_textures(&images.iter().map(Image::dx12).collect::<SmallVec<[_; 8]>>())
            }
        }
    }

    pub unsafe fn build(
        self,
        device: &Device,
        pipeline: &Pipeline,
    ) -> Result<DescriptorSet, Error> {
        match self {
            DescriptorSetBuilder::Vk(x) => {
                x.build(device.vk(), pipeline.vk()).map(DescriptorSet::Vk)
            }
            DescriptorSetBuilder::Dx12(x) => x
                .build(device.dx12(), pipeline.dx12())
                .map(DescriptorSet::Dx12),
        }
    }
}

impl CmdBuf {
    pub unsafe fn begin(&mut self) {
        match self {
            CmdBuf::Vk(c) => c.begin(),
            CmdBuf::Dx12(c) => c.begin(),
        }
    }

    pub unsafe fn finish(&mut self) {
        match self {
            CmdBuf::Vk(c) => c.finish(),
            CmdBuf::Dx12(c) => c.finish(),
        }
    }

    pub unsafe fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        descriptor_set: &DescriptorSet,
        size: (u32, u32, u32),
    ) {
        match self {
            CmdBuf::Vk(c) => c.dispatch(pipeline.vk(), descriptor_set.vk(), size),
            CmdBuf::Dx12(c) => c.dispatch(pipeline.dx12(), descriptor_set.dx12(), size),
        }
    }

    pub unsafe fn memory_barrier(&mut self) {
        match self {
            CmdBuf::Vk(c) => c.memory_barrier(),
            CmdBuf::Dx12(c) => c.memory_barrier(),
        }
    }

    pub unsafe fn host_barrier(&mut self) {
        match self {
            CmdBuf::Vk(c) => c.host_barrier(),
            CmdBuf::Dx12(c) => c.host_barrier(),
        }
    }

    pub unsafe fn image_barrier(
        &mut self,
        image: &Image,
        src_layout: ImageLayout,
        dst_layout: ImageLayout,
    ) {
        match self {
            CmdBuf::Vk(c) => c.image_barrier(image.vk(), src_layout, dst_layout),
            CmdBuf::Dx12(c) => c.image_barrier(image.dx12(), src_layout, dst_layout),
        }
    }

    pub unsafe fn clear_buffer(&mut self, buffer: &Buffer, size: Option<u64>) {
        match self {
            CmdBuf::Vk(c) => c.clear_buffer(buffer.vk(), size),
            CmdBuf::Dx12(c) => c.clear_buffer(buffer.dx12(), size),
        }
    }

    pub unsafe fn copy_buffer(&mut self, src: &Buffer, dst: &Buffer) {
        match self {
            CmdBuf::Vk(c) => c.copy_buffer(src.vk(), dst.vk()),
            CmdBuf::Dx12(c) => c.copy_buffer(src.dx12(), dst.dx12()),
        }
    }

    pub unsafe fn copy_image_to_buffer(&mut self, src: &Image, dst: &Buffer) {
        match self {
            CmdBuf::Vk(c) => c.copy_image_to_buffer(src.vk(), dst.vk()),
            CmdBuf::Dx12(c) => c.copy_image_to_buffer(src.dx12(), dst.dx12()),
        }
    }

    pub unsafe fn copy_buffer_to_image(&mut self, src: &Buffer, dst: &Image) {
        match self {
            CmdBuf::Vk(c) => c.copy_buffer_to_image(src.vk(), dst.vk()),
            CmdBuf::Dx12(c) => c.copy_buffer_to_image(src.dx12(), dst.dx12()),
        }
    }

    pub unsafe fn blit_image(&mut self, src: &Image, dst: &Image) {
        match self {
            CmdBuf::Vk(c) => c.blit_image(src.vk(), dst.vk()),
            CmdBuf::Dx12(c) => c.blit_image(src.dx12(), dst.dx12()),
        }
    }

    pub unsafe fn reset_query_pool(&mut self, pool: &QueryPool) {
        match self {
            CmdBuf::Vk(c) => c.reset_query_pool(pool.vk()),
            CmdBuf::Dx12(c) => c.reset_query_pool(pool.dx12()),
        }
    }

    pub unsafe fn write_timestamp(&mut self, pool: &QueryPool, query: u32) {
        match self {
            CmdBuf::Vk(c) => c.write_timestamp(pool.vk(), query),
            CmdBuf::Dx12(c) => c.write_timestamp(pool.dx12(), query),
        }
    }

    pub unsafe fn finish_timestamps(&mut self, pool: &QueryPool) {
        match self {
            CmdBuf::Vk(c) => c.finish_timestamps(pool.vk()),
            CmdBuf::Dx12(c) => c.finish_timestamps(pool.dx12()),
        }
    }
}
