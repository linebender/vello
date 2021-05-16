//! DX12 implemenation of HAL trait.

mod error;
mod wrappers;

use std::{cell::Cell, convert::TryInto, mem, ptr};

use winapi::shared::dxgi1_3;
use winapi::um::d3d12;

use crate::Error;

use self::wrappers::{
    CommandAllocator, CommandQueue, Device, Factory4, GraphicsCommandList, Resource, ShaderByteCode,
};

pub struct Dx12Instance {
    factory: Factory4,
}

pub struct Dx12Device {
    device: Device,
    command_allocator: CommandAllocator,
    command_queue: CommandQueue,
}

#[derive(Clone)]
pub struct Buffer {
    resource: Resource,
    size: u64,
}

pub struct Image {
    resource: Resource,
}

// TODO: this doesn't make an upload/download distinction. Probably
// we want to move toward webgpu-style usage flags, ie map_read and
// map_write are the main ones that affect buffer creation.
#[derive(Clone, Copy)]
pub enum MemFlags {
    DeviceLocal,
    HostCoherent,
}

pub struct CmdBuf(GraphicsCommandList);

pub struct Pipeline {
    pipeline_state: wrappers::PipelineState,
    root_signature: wrappers::RootSignature,
}

// Right now, each descriptor set gets its own heap, but we'll move
// to a more sophisticated allocation scheme, probably using the
// gpu-descriptor crate.
pub struct DescriptorSet(wrappers::DescriptorHeap);

pub struct QueryPool;

pub struct Fence {
    fence: wrappers::Fence,
    event: wrappers::Event,
    // This could as well be an atomic, if we needed to cross threads.
    val: Cell<u64>,
}

pub struct Semaphore;

#[derive(Default)]
pub struct PipelineBuilder {
    ranges: Vec<d3d12::D3D12_DESCRIPTOR_RANGE>,
    n_uav: u32,
    // TODO: add counters for other resource types
}

// TODO
#[derive(Default)]
pub struct DescriptorSetBuilder {
    buffers: Vec<Buffer>,
}

impl Dx12Instance {
    /// Create a new instance.
    ///
    /// TODO: take a raw window handle.
    /// TODO: can probably be a trait.
    pub fn new() -> Result<Dx12Instance, Error> {
        unsafe {
            #[cfg(debug_assertions)]
            if let Err(e) = wrappers::enable_debug_layer() {
                // Maybe a better logging solution?
                println!("{}", e);
            }

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
            let command_queue = device.create_command_queue(
                list_type,
                0,
                d3d12::D3D12_COMMAND_QUEUE_FLAG_NONE,
                0,
            )?;
            let command_allocator = device.create_command_allocator(list_type)?;
            Ok(Dx12Device {
                device,
                command_queue,
                command_allocator,
            })
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

    type PipelineBuilder = PipelineBuilder;

    type DescriptorSetBuilder = DescriptorSetBuilder;

    type Sampler = ();

    // Currently this is HLSL source, but we'll probably change it to IR.
    type ShaderSource = str;

    fn create_buffer(&self, size: u64, mem_flags: Self::MemFlags) -> Result<Self::Buffer, Error> {
        unsafe {
            let resource = match mem_flags {
                MemFlags::DeviceLocal => self
                    .device
                    .create_gpu_only_byte_addressed_buffer(size.try_into()?)?,
                MemFlags::HostCoherent => self
                    .device
                    .create_uploadable_byte_addressed_buffer(size.try_into()?)?,
            };
            Ok(Buffer { resource, size })
        }
    }

    unsafe fn destroy_buffer(&self, buffer: &Self::Buffer) -> Result<(), Error> {
        buffer.resource.destroy();
        Ok(())
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

    fn create_cmd_buf(&self) -> Result<Self::CmdBuf, Error> {
        let list_type = d3d12::D3D12_COMMAND_LIST_TYPE_DIRECT;
        let node_mask = 0;
        unsafe {
            let cmd_buf = self.device.create_graphics_command_list(
                list_type,
                &self.command_allocator,
                None,
                node_mask,
            )?;
            Ok(CmdBuf(cmd_buf))
        }
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
        // TODO: handle semaphores
        self.command_queue
            .execute_command_lists(&[cmd_buf.0.as_raw_list()]);
        if let Some(fence) = fence {
            let val = fence.val.get() + 1;
            fence.val.set(val);
            self.command_queue.signal(&fence.fence, val)?;
            fence.fence.set_event_on_completion(&fence.event, val)?;
        }
        Ok(())
    }

    unsafe fn read_buffer<T: Sized>(
        &self,
        buffer: &Self::Buffer,
        result: &mut Vec<T>,
    ) -> Result<(), Error> {
        let len = buffer.size as usize / std::mem::size_of::<T>();
        if len > result.len() {
            result.reserve(len - result.len());
        }
        buffer.resource.read_resource(result.as_mut_ptr(), len)?;
        result.set_len(len);
        Ok(())
    }

    unsafe fn write_buffer<T: Sized>(
        &self,
        buffer: &Self::Buffer,
        contents: &[T],
    ) -> Result<(), Error> {
        let len = buffer.size as usize / std::mem::size_of::<T>();
        buffer.resource.write_resource(len, contents.as_ptr())?;
        Ok(())
    }

    unsafe fn create_semaphore(&self) -> Result<Self::Semaphore, Error> {
        todo!()
    }

    unsafe fn create_fence(&self, signaled: bool) -> Result<Self::Fence, Error> {
        let fence = self.device.create_fence(0)?;
        let event = wrappers::Event::create(false, signaled)?;
        let val = Cell::new(0);
        Ok(Fence { fence, event, val })
    }

    unsafe fn wait_and_reset(&self, fences: &[Self::Fence]) -> Result<(), Error> {
        for fence in fences {
            // TODO: probably handle errors here.
            let _status = fence.event.wait(winapi::um::winbase::INFINITE);
        }
        Ok(())
    }

    unsafe fn get_fence_status(&self, fence: Self::Fence) -> Result<bool, Error> {
        let fence_val = fence.fence.get_value();
        Ok(fence_val == fence.val.get())
    }

    fn query_gpu_info(&self) -> crate::GpuInfo {
        todo!()
    }

    unsafe fn pipeline_builder(&self) -> Self::PipelineBuilder {
        PipelineBuilder::default()
    }

    unsafe fn descriptor_set_builder(&self) -> Self::DescriptorSetBuilder {
        DescriptorSetBuilder::default()
    }

    unsafe fn create_sampler(&self, params: crate::SamplerParams) -> Result<Self::Sampler, Error> {
        todo!()
    }
}

impl crate::CmdBuf<Dx12Device> for CmdBuf {
    unsafe fn begin(&mut self) {}

    unsafe fn finish(&mut self) {
        let _ = self.0.close();
    }

    unsafe fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        descriptor_set: &DescriptorSet,
        size: (u32, u32, u32),
    ) {
        self.0.set_pipeline_state(&pipeline.pipeline_state);
        self.0
            .set_compute_pipeline_root_signature(&pipeline.root_signature);
        self.0.set_descriptor_heaps(&[&descriptor_set.0]);
        self.0.set_compute_root_descriptor_table(
            0,
            descriptor_set.0.get_gpu_descriptor_handle_at_offset(0),
        );
        self.0.dispatch(size.0, size.1, size.2);
    }

    unsafe fn memory_barrier(&mut self) {
        // See comments in CommandBuffer::pipeline_barrier in gfx-hal dx12 backend.
        // The "proper" way to do this would be to name the actual buffers participating
        // in the barrier. But it seems like this is a reasonable way to create a
        // global barrier.
        let bar = wrappers::create_uav_resource_barrier(ptr::null_mut());
        self.0.resource_barrier(&[bar]);
    }

    unsafe fn host_barrier(&mut self) {
        // TODO: anything special here?
        self.memory_barrier();
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
        let size = src.size.min(dst.size);
        self.0.copy_buffer(&dst.resource, 0, &src.resource, 0, size);
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

impl crate::PipelineBuilder<Dx12Device> for PipelineBuilder {
    fn add_buffers(&mut self, n_buffers: u32) {
        if n_buffers != 0 {
            self.ranges.push(d3d12::D3D12_DESCRIPTOR_RANGE {
                RangeType: d3d12::D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                NumDescriptors: n_buffers,
                BaseShaderRegister: self.n_uav,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: d3d12::D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            });
            self.n_uav += n_buffers;
        }
    }

    fn add_images(&mut self, n_images: u32) {
        if n_images != 0 {
            todo!()
        }
    }

    fn add_textures(&mut self, max_textures: u32) {
        todo!()
    }

    unsafe fn create_compute_pipeline(
        self,
        device: &Dx12Device,
        code: &str,
    ) -> Result<Pipeline, Error> {
        #[cfg(debug_assertions)]
        let flags = winapi::um::d3dcompiler::D3DCOMPILE_DEBUG
            | winapi::um::d3dcompiler::D3DCOMPILE_SKIP_OPTIMIZATION;
        #[cfg(not(debug_assertions))]
        let flags = 0;
        let shader_blob = ShaderByteCode::compile(code, "cs_5_1", "main", flags)?;
        let shader = ShaderByteCode::from_blob(shader_blob);
        let mut root_parameter = d3d12::D3D12_ROOT_PARAMETER {
            ParameterType: d3d12::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            ShaderVisibility: d3d12::D3D12_SHADER_VISIBILITY_ALL,
            ..mem::zeroed()
        };
        *root_parameter.u.DescriptorTable_mut() = d3d12::D3D12_ROOT_DESCRIPTOR_TABLE {
            NumDescriptorRanges: self.ranges.len().try_into()?,
            pDescriptorRanges: self.ranges.as_ptr(),
        };
        let root_signature_desc = d3d12::D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: 1,
            pParameters: &root_parameter,
            NumStaticSamplers: 0,
            pStaticSamplers: ptr::null(),
            Flags: d3d12::D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };
        let root_signature_blob = wrappers::RootSignature::serialize_description(
            &root_signature_desc,
            d3d12::D3D_ROOT_SIGNATURE_VERSION_1,
        )?;
        let root_signature = device
            .device
            .create_root_signature(0, root_signature_blob)?;
        let desc = d3d12::D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: root_signature.0.as_raw(),
            CS: shader.bytecode,
            NodeMask: 0,
            CachedPSO: d3d12::D3D12_CACHED_PIPELINE_STATE {
                pCachedBlob: ptr::null(),
                CachedBlobSizeInBytes: 0,
            },
            Flags: d3d12::D3D12_PIPELINE_STATE_FLAG_NONE,
        };
        let pipeline_state = device.device.create_compute_pipeline_state(&desc)?;
        Ok(Pipeline {
            pipeline_state,
            root_signature,
        })
    }
}

impl crate::DescriptorSetBuilder<Dx12Device> for DescriptorSetBuilder {
    fn add_buffers(&mut self, buffers: &[&Buffer]) {
        // Note: we could get rid of the clone here (which is an AddRef)
        // and store a raw pointer, as it's a safety precondition that
        // the resources are kept alive til build.
        self.buffers.extend(buffers.iter().copied().cloned());
    }

    fn add_images(&mut self, images: &[&Image]) {
        if !images.is_empty() {
            todo!()
        }
    }

    fn add_textures(&mut self, images: &[&Image]) {
        todo!()
    }

    unsafe fn build(
        self,
        device: &Dx12Device,
        _pipeline: &Pipeline,
    ) -> Result<DescriptorSet, Error> {
        let n_descriptors = self.buffers.len();
        let heap_desc = d3d12::D3D12_DESCRIPTOR_HEAP_DESC {
            Type: d3d12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            NumDescriptors: n_descriptors.try_into()?,
            Flags: d3d12::D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        };
        let heap = device.device.create_descriptor_heap(&heap_desc)?;
        let mut ix = 0;
        for buffer in self.buffers {
            device
                .device
                .create_byte_addressed_buffer_unordered_access_view(
                    &buffer.resource,
                    heap.get_cpu_descriptor_handle_at_offset(ix),
                    0,
                    (buffer.size / 4).try_into()?,
                );
            ix += 1;
        }
        Ok(DescriptorSet(heap))
    }
}
