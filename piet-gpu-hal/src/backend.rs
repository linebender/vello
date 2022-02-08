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

//! The generic trait for backends to implement.

use crate::{
    BindType, BufferUsage, Error, GpuInfo, ImageFormat, ImageLayout, MapMode, SamplerParams,
};

pub trait Device: Sized {
    type Buffer: 'static;
    type Image;
    type Pipeline;
    type DescriptorSet;
    type QueryPool;
    type CmdBuf: CmdBuf<Self>;
    type Fence;
    type Semaphore;
    type DescriptorSetBuilder: DescriptorSetBuilder<Self>;
    type Sampler;
    type ShaderSource: ?Sized;

    /// Query the GPU info.
    ///
    /// This method may be expensive, so the hub should call it once and retain
    /// the info.
    fn query_gpu_info(&self) -> GpuInfo;

    fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Self::Buffer, Error>;

    /// Destroy a buffer.
    ///
    /// The same safety requirements hold as in Vulkan: the buffer cannot be used
    /// after this call, and all commands referencing this buffer must have completed.
    ///
    /// Maybe doesn't need result return?
    unsafe fn destroy_buffer(&self, buffer: &Self::Buffer) -> Result<(), Error>;

    unsafe fn create_image2d(
        &self,
        width: u32,
        height: u32,
        format: ImageFormat,
    ) -> Result<Self::Image, Error>;

    /// Destroy an image.
    ///
    /// The same safety requirements hold as in Vulkan: the image cannot be used
    /// after this call, and all commands referencing this image must have completed.
    ///
    /// Use this only with images we created, not for swapchain images.
    ///
    /// Maybe doesn't need result return?
    unsafe fn destroy_image(&self, image: &Self::Image) -> Result<(), Error>;

    /// Build a compute pipeline.
    ///
    /// A pipeline is a bit of shader IR plus a signature for what kinds of resources
    /// it expects.
    unsafe fn create_compute_pipeline(
        &self,
        code: &Self::ShaderSource,
        bind_types: &[BindType],
    ) -> Result<Self::Pipeline, Error>;

    /// Start building a descriptor set.
    ///
    /// A descriptor set is a binding of resources for a given pipeline.
    unsafe fn descriptor_set_builder(&self) -> Self::DescriptorSetBuilder;

    /// Create a descriptor set for a given pipeline, binding buffers and images.
    ///
    /// This is provided as a convenience but will probably go away, as the functionality
    /// is subsumed by the builder.
    unsafe fn create_descriptor_set(
        &self,
        pipeline: &Self::Pipeline,
        bufs: &[&Self::Buffer],
        images: &[&Self::Image],
    ) -> Result<Self::DescriptorSet, Error> {
        let mut builder = self.descriptor_set_builder();
        builder.add_buffers(bufs);
        builder.add_images(images);
        builder.build(self, pipeline)
    }

    fn create_cmd_buf(&self) -> Result<Self::CmdBuf, Error>;

    /// If the command buffer was submitted, it must complete before this is called.
    unsafe fn destroy_cmd_buf(&self, cmd_buf: Self::CmdBuf) -> Result<(), Error>;

    fn create_query_pool(&self, n_queries: u32) -> Result<Self::QueryPool, Error>;

    /// Get results from query pool, destroying it in the process.
    ///
    /// The returned vector is one less than the number of queries; the first is used as
    /// a baseline.
    ///
    /// # Safety
    /// All submitted commands that refer to this query pool must have completed.
    unsafe fn fetch_query_pool(&self, pool: &Self::QueryPool) -> Result<Vec<f64>, Error>;

    unsafe fn run_cmd_bufs(
        &self,
        cmd_buf: &[&Self::CmdBuf],
        wait_semaphores: &[&Self::Semaphore],
        signal_semaphores: &[&Self::Semaphore],
        fence: Option<&mut Self::Fence>,
    ) -> Result<(), Error>;

    /// Map the buffer into addressable memory.
    ///
    /// # Safety
    ///
    /// The buffer must be valid to access. The offset + size much be within the
    /// buffer's allocation. The buffer must not already be mapped. Of course,
    /// the usual safety rules apply to the returned pointer.
    unsafe fn map_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        size: u64,
        mode: MapMode,
    ) -> Result<*mut u8, Error>;

    /// Map the buffer into addressable memory.
    ///
    /// # Safety
    ///
    /// The buffer must be mapped. The parameters must be the same as the map
    /// call.
    unsafe fn unmap_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        size: u64,
        mode: MapMode,
    ) -> Result<(), Error>;

    unsafe fn create_semaphore(&self) -> Result<Self::Semaphore, Error>;
    unsafe fn create_fence(&self, signaled: bool) -> Result<Self::Fence, Error>;
    unsafe fn destroy_fence(&self, fence: Self::Fence) -> Result<(), Error>;
    unsafe fn wait_and_reset(&self, fences: Vec<&mut Self::Fence>) -> Result<(), Error>;
    unsafe fn get_fence_status(&self, fence: &mut Self::Fence) -> Result<bool, Error>;

    unsafe fn create_sampler(&self, params: SamplerParams) -> Result<Self::Sampler, Error>;
}

pub trait CmdBuf<D: Device> {
    unsafe fn begin(&mut self);

    unsafe fn finish(&mut self);

    /// Return true if the command buffer is suitable for reuse.
    unsafe fn reset(&mut self) -> bool;

    unsafe fn dispatch(
        &mut self,
        pipeline: &D::Pipeline,
        descriptor_set: &D::DescriptorSet,
        workgroup_count: (u32, u32, u32),
        workgroup_size: (u32, u32, u32),
    );

    /// Insert an execution and memory barrier.
    ///
    /// Compute kernels (and other actions) after this barrier may read from buffers
    /// that were written before this barrier.
    unsafe fn memory_barrier(&mut self);

    /// Insert a barrier for host access to buffers.
    ///
    /// The host may read buffers written before this barrier, after the fence for
    /// the command buffer is signaled.
    ///
    /// See http://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/
    /// ("Host memory reads") for an explanation of this barrier.
    unsafe fn host_barrier(&mut self);

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
    unsafe fn clear_buffer(&self, buffer: &D::Buffer, size: Option<u64>);

    unsafe fn copy_buffer(&self, src: &D::Buffer, dst: &D::Buffer);

    unsafe fn copy_image_to_buffer(&self, src: &D::Image, dst: &D::Buffer);

    unsafe fn copy_buffer_to_image(&self, src: &D::Buffer, dst: &D::Image);

    // low portability, dx12 doesn't support it natively
    unsafe fn blit_image(&self, src: &D::Image, dst: &D::Image);

    /// Reset the query pool.
    ///
    /// The query pool must be reset before each use, to avoid validation errors.
    /// This is annoying, and we could tweak the API to make it implicit, doing
    /// the reset before the first timestamp write.
    unsafe fn reset_query_pool(&mut self, pool: &D::QueryPool);

    unsafe fn write_timestamp(&mut self, pool: &D::QueryPool, query: u32);

    /// Prepare the timestamps for reading. This isn't required on Vulkan but
    /// is required on (at least) DX12.
    unsafe fn finish_timestamps(&mut self, _pool: &D::QueryPool) {}

    /// Begin a labeled section for debugging and profiling purposes.
    unsafe fn begin_debug_label(&mut self, label: &str) {}

    /// End a section opened by `begin_debug_label`.
    unsafe fn end_debug_label(&mut self) {}
}

/// A builder for descriptor sets with more complex layouts.
///
/// Note: the order needs to match the pipeline building, and it also needs to
/// be buffers, then images, then textures.
pub trait DescriptorSetBuilder<D: Device> {
    fn add_buffers(&mut self, buffers: &[&D::Buffer]);
    /// Add an array of storage images.
    ///
    /// The images need to be in `ImageLayout::General` layout.
    fn add_images(&mut self, images: &[&D::Image]);
    /// Add an array of textures.
    ///
    /// The images need to be in `ImageLayout::ShaderRead` layout.
    ///
    /// The same sampler is used for all textures, which is not very sophisticated;
    /// we should have a way to vary the sampler.
    fn add_textures(&mut self, images: &[&D::Image]);
    unsafe fn build(self, device: &D, pipeline: &D::Pipeline) -> Result<D::DescriptorSet, Error>;
}
