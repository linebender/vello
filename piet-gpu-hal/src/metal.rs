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

use cocoa_foundation::foundation::NSInteger;
use objc::rc::autoreleasepool;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

use metal::MTLFeatureSet;

use crate::{BufferUsage, Error, GpuInfo};

pub struct MtlInstance;

pub struct MtlDevice {
    device: metal::Device,
    cmd_queue: metal::CommandQueue,
    gpu_info: GpuInfo,
}

pub struct MtlSurface;

pub struct MtlSwapchain;

#[derive(Clone)]
pub struct Buffer {
    buffer: metal::Buffer,
    pub(crate) size: u64,
}

pub struct Image;

pub struct Fence;

pub struct Semaphore;

pub struct CmdBuf {
    cmd_buf: metal::CommandBuffer,
}

pub struct QueryPool;

pub struct PipelineBuilder;

pub struct Pipeline(metal::ComputePipelineState);

#[derive(Default)]
pub struct DescriptorSetBuilder(DescriptorSet);

#[derive(Default)]
pub struct DescriptorSet {
    buffers: Vec<Buffer>,
}

impl MtlInstance {
    pub fn new(
        _window_handle: Option<&dyn raw_window_handle::HasRawWindowHandle>,
    ) -> Result<(MtlInstance, Option<MtlSurface>), Error> {
        Ok((MtlInstance, None))
    }

    // TODO might do some enumeration of devices

    pub fn device(&self, _surface: Option<&MtlSurface>) -> Result<MtlDevice, Error> {
        if let Some(device) = metal::Device::system_default() {
            let cmd_queue = device.new_command_queue();
            let is_mac = device.supports_feature_set(MTLFeatureSet::macOS_GPUFamily1_v1);
            let is_ios = device.supports_feature_set(MTLFeatureSet::iOS_GPUFamily1_v1);
            let version = NSOperatingSystemVersion::get();

            let use_staging_buffers =
                if (is_mac && version.at_least(10, 15)) || (is_ios && version.at_least(13, 0)) {
                    !device.has_unified_memory()
                } else {
                    !device.is_low_power()
                };
            // TODO: these are conservative; we need to derive these from
            // supports_feature_set queries.
            let gpu_info = GpuInfo {
                has_descriptor_indexing: false,
                has_subgroups: false,
                subgroup_size: None,
                has_memory_model: false,
                use_staging_buffers: use_staging_buffers,
            };
            Ok(MtlDevice { device, cmd_queue, gpu_info })
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
        self.gpu_info.clone()
    }

    fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Self::Buffer, Error> {
        let options = if usage.contains(BufferUsage::MAP_READ) {
            metal::MTLResourceOptions::StorageModeShared
                | metal::MTLResourceOptions::CPUCacheModeDefaultCache
        } else if usage.contains(BufferUsage::MAP_WRITE) {
            metal::MTLResourceOptions::StorageModeShared
                | metal::MTLResourceOptions::CPUCacheModeWriteCombined
        } else {
            metal::MTLResourceOptions::StorageModePrivate
        };
        let buffer = self.device.new_buffer(size, options);
        Ok(Buffer { buffer, size })
    }

    unsafe fn destroy_buffer(&self, buffer: &Self::Buffer) -> Result<(), Error> {
        // This defers dropping until the buffer object is dropped. We probably need
        // to rethink buffer lifetime if descriptor sets can retain references.
        Ok(())
    }

    unsafe fn create_image2d(&self, width: u32, height: u32) -> Result<Self::Image, Error> {
        todo!()
    }

    unsafe fn destroy_image(&self, image: &Self::Image) -> Result<(), Error> {
        todo!()
    }

    unsafe fn pipeline_builder(&self) -> Self::PipelineBuilder {
        PipelineBuilder
    }

    unsafe fn descriptor_set_builder(&self) -> Self::DescriptorSetBuilder {
        DescriptorSetBuilder::default()
    }

    fn create_cmd_buf(&self) -> Result<Self::CmdBuf, Error> {
        // consider new_command_buffer_with_unretained_references for performance
        let cmd_buf = self.cmd_queue.new_command_buffer();
        let cmd_buf = autoreleasepool(|| cmd_buf.to_owned());
        Ok(CmdBuf { cmd_buf })
    }

    fn create_query_pool(&self, n_queries: u32) -> Result<Self::QueryPool, Error> {
        // TODO
        Ok(QueryPool)
    }

    unsafe fn fetch_query_pool(&self, pool: &Self::QueryPool) -> Result<Vec<f64>, Error> {
        // TODO
        Ok(Vec::new())
    }

    unsafe fn run_cmd_bufs(
        &self,
        cmd_bufs: &[&Self::CmdBuf],
        wait_semaphores: &[&Self::Semaphore],
        signal_semaphores: &[&Self::Semaphore],
        fence: Option<&mut Self::Fence>,
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
        std::ptr::copy_nonoverlapping(
            (contents_ptr as *const u8).add(offset as usize),
            dst,
            size as usize,
        );
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
        std::ptr::copy_nonoverlapping(
            contents,
            (contents_ptr as *mut u8).add(offset as usize),
            size as usize,
        );
        Ok(())
    }

    unsafe fn create_semaphore(&self) -> Result<Self::Semaphore, Error> {
        todo!()
    }

    unsafe fn create_fence(&self, signaled: bool) -> Result<Self::Fence, Error> {
        todo!()
    }

    unsafe fn wait_and_reset(&self, fences: &[&mut Self::Fence]) -> Result<(), Error> {
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
        let encoder = self.cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.0);
        let mut ix = 0;
        for buffer in &descriptor_set.buffers {
            encoder.set_buffer(ix, Some(&buffer.buffer), 0);
            ix += 1;
        }
        // TODO: set images
        let work_group_count = metal::MTLSize {
            width: size.0 as u64,
            height: size.1 as u64,
            depth: size.2 as u64,
        };
        // TODO: we need to pass this in explicitly. In gfx-hal, this is parsed from
        // the spv before translation.
        let work_group_size = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(work_group_count, work_group_size);
        encoder.end_encoding();
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
    fn add_buffers(&mut self, _n_buffers: u32) {
        // My understanding is that Metal infers the pipeline layout from
        // the source.
    }

    fn add_images(&mut self, _n_images: u32) {
    }

    fn add_textures(&mut self, _max_textures: u32) {
    }

    unsafe fn create_compute_pipeline(
        self,
        device: &MtlDevice,
        code: &str,
    ) -> Result<Pipeline, Error> {
        let options = metal::CompileOptions::new();
        // Probably want to set MSL version here.
        let library = device.device.new_library_with_source(code, &options)?;
        // This seems to be the default name from spirv-cross, but we may need to tweak.
        let function = library.get_function("main0", None)?;
        let pipeline = device.device.new_compute_pipeline_state_with_function(&function)?;
        Ok(Pipeline(pipeline))
    }
}

impl crate::DescriptorSetBuilder<MtlDevice> for DescriptorSetBuilder {
    fn add_buffers(&mut self, buffers: &[&Buffer]) {
        self.0.buffers.extend(buffers.iter().copied().cloned());
    }

    fn add_images(&mut self, images: &[&Image]) {
        todo!()
    }

    fn add_textures(&mut self, images: &[&Image]) {
        todo!()
    }

    unsafe fn build(self, device: &MtlDevice, pipeline: &Pipeline) -> Result<DescriptorSet, Error> {
        Ok(self.0)
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

#[repr(C)]
struct NSOperatingSystemVersion {
    major: NSInteger,
    minor: NSInteger,
    patch: NSInteger,
}

impl NSOperatingSystemVersion {
    fn get() -> NSOperatingSystemVersion {
        unsafe {
            let process_info: *mut Object = msg_send![class!(NSProcessInfo), processInfo];
            msg_send![process_info, operatingSystemVersion]
        }
    }

    fn at_least(&self, major: u32, minor: u32) -> bool {
        let major = major as NSInteger;
        let minor = minor as NSInteger;
        self.major > major || (self.major == major && self.minor >= minor)
    }
}
