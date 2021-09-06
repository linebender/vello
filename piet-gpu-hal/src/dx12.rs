//! DX12 implemenation of HAL trait.

mod error;
mod wrappers;

use std::sync::{Arc, Mutex, Weak};
use std::{cell::Cell, convert::TryInto, mem, ptr};

use winapi::shared::minwindef::TRUE;
use winapi::shared::{dxgi, dxgi1_2, dxgi1_3, dxgitype};
use winapi::um::d3d12;

use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};

use smallvec::SmallVec;

use crate::{BufferUsage, Error, GpuInfo, ImageLayout, WorkgroupLimits};

use self::wrappers::{CommandAllocator, CommandQueue, Device, Factory4, Resource, ShaderByteCode};

pub struct Dx12Instance {
    factory: Factory4,
}

pub struct Dx12Surface {
    hwnd: winapi::shared::windef::HWND,
}

pub struct Dx12Swapchain {
    swapchain: wrappers::SwapChain3,
    size: (u32, u32),
}

pub struct Dx12Device {
    device: Device,
    free_allocators: Arc<Mutex<Vec<CommandAllocator>>>,
    command_queue: CommandQueue,
    ts_freq: u64,
    gpu_info: GpuInfo,
    memory_arch: MemoryArchitecture,
}

#[derive(Clone)]
pub struct Buffer {
    resource: Resource,
    pub size: u64,
}

#[derive(Clone)]
pub struct Image {
    resource: Resource,
    size: (u32, u32),
}

pub struct CmdBuf {
    c: wrappers::GraphicsCommandList,
    allocator: Option<CommandAllocator>,
    // One for resetting, one to put back into the allocator pool
    allocator_clone: CommandAllocator,
    free_allocators: Weak<Mutex<Vec<CommandAllocator>>>,
}

pub struct Pipeline {
    pipeline_state: wrappers::PipelineState,
    root_signature: wrappers::RootSignature,
}

// Right now, each descriptor set gets its own heap, but we'll move
// to a more sophisticated allocation scheme, probably using the
// gpu-descriptor crate.
pub struct DescriptorSet(wrappers::DescriptorHeap);

pub struct QueryPool {
    heap: wrappers::QueryHeap,
    buf: Buffer,
    n_queries: u32,
}

pub struct Fence {
    fence: wrappers::Fence,
    event: wrappers::Event,
    // This could as well be an atomic, if we needed to cross threads.
    val: Cell<u64>,
}

/// This will probably be renamed "PresentSem" or similar. I believe no
/// semaphore is needed for presentation on DX12.
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
    images: Vec<Image>,
}

#[derive(PartialEq, Eq)]
enum MemoryArchitecture {
    /// Integrated graphics
    CacheCoherentUMA,
    /// Unified memory with no cache coherence (does this happen?)
    UMA,
    /// Discrete graphics
    NUMA,
}

impl Dx12Instance {
    /// Create a new instance.
    ///
    /// TODO: take a raw window handle.
    /// TODO: can probably be a trait.
    pub fn new(
        window_handle: Option<&dyn HasRawWindowHandle>,
    ) -> Result<(Dx12Instance, Option<Dx12Surface>), Error> {
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

            let mut surface = None;
            if let Some(window_handle) = window_handle {
                let window_handle = window_handle.raw_window_handle();
                if let RawWindowHandle::Windows(w) = window_handle {
                    let hwnd = w.hwnd as *mut _;
                    surface = Some(Dx12Surface { hwnd });
                }
            }
            Ok((Dx12Instance { factory }, surface))
        }
    }

    /// Get a device suitable for compute workloads.
    ///
    /// TODO: handle window.
    /// TODO: probably can also be trait'ified.
    pub fn device(&self, surface: Option<&Dx12Surface>) -> Result<Dx12Device, Error> {
        unsafe {
            let device = Device::create_device(&self.factory)?;
            let list_type = d3d12::D3D12_COMMAND_LIST_TYPE_DIRECT;
            let command_queue = device.create_command_queue(
                list_type,
                0,
                d3d12::D3D12_COMMAND_QUEUE_FLAG_NONE,
                0,
            )?;

            let ts_freq = command_queue.get_timestamp_frequency()?;
            let features_architecture = device.get_features_architecture()?;
            let uma = features_architecture.UMA == TRUE;
            let cc_uma = features_architecture.CacheCoherentUMA == TRUE;
            let memory_arch = match (uma, cc_uma) {
                (true, true) => MemoryArchitecture::CacheCoherentUMA,
                (true, false) => MemoryArchitecture::UMA,
                _ => MemoryArchitecture::NUMA,
            };
            let use_staging_buffers = memory_arch == MemoryArchitecture::NUMA;
            // These values are appropriate for Shader Model 5. When we open up
            // DXIL, fix this with proper dynamic queries.
            let gpu_info = GpuInfo {
                has_descriptor_indexing: false,
                has_subgroups: false,
                subgroup_size: None,
                workgroup_limits: WorkgroupLimits {
                    max_size: [1024, 1024, 64],
                    max_invocations: 1024,
                },
                has_memory_model: false,
                use_staging_buffers,
            };
            let free_allocators = Default::default();
            Ok(Dx12Device {
                device,
                command_queue,
                free_allocators,
                ts_freq,
                memory_arch,
                gpu_info,
            })
        }
    }

    pub unsafe fn swapchain(
        &self,
        width: usize,
        height: usize,
        device: &Dx12Device,
        surface: &Dx12Surface,
    ) -> Result<Dx12Swapchain, Error> {
        const FRAME_COUNT: u32 = 2;
        let desc = dxgi1_2::DXGI_SWAP_CHAIN_DESC1 {
            Width: width as u32,
            Height: height as u32,
            AlphaMode: dxgi1_2::DXGI_ALPHA_MODE_IGNORE,
            BufferCount: FRAME_COUNT,
            Format: winapi::shared::dxgiformat::DXGI_FORMAT_R8G8B8A8_UNORM,
            Flags: 0,
            BufferUsage: dxgitype::DXGI_USAGE_RENDER_TARGET_OUTPUT,
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Scaling: dxgi1_2::DXGI_SCALING_STRETCH,
            Stereo: winapi::shared::minwindef::FALSE,
            SwapEffect: dxgi::DXGI_SWAP_EFFECT_FLIP_DISCARD,
        };
        let swapchain =
            self.factory
                .create_swapchain_for_hwnd(&device.command_queue, surface.hwnd, desc)?;
        let size = (width as u32, height as u32);
        Ok(Dx12Swapchain { swapchain, size })
    }
}

impl crate::backend::Device for Dx12Device {
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

    // Currently this is HLSL source, but we'll probably change it to IR.
    type ShaderSource = str;

    fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Self::Buffer, Error> {
        // TODO: consider supporting BufferUsage::QUERY_RESOLVE here rather than
        // having a separate function.
        unsafe {
            let page_property = self.memory_arch.page_property(usage);
            let memory_pool = self.memory_arch.memory_pool(usage);
            //TODO: consider flag D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS?
            let flags = d3d12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            let resource = self.device.create_buffer(
                size.try_into()?,
                d3d12::D3D12_HEAP_TYPE_CUSTOM,
                page_property,
                memory_pool,
                d3d12::D3D12_RESOURCE_STATE_COMMON,
                flags,
            )?;
            Ok(Buffer { resource, size })
        }
    }

    unsafe fn destroy_buffer(&self, buffer: &Self::Buffer) -> Result<(), Error> {
        buffer.resource.destroy();
        Ok(())
    }

    unsafe fn create_image2d(&self, width: u32, height: u32) -> Result<Self::Image, Error> {
        let format = winapi::shared::dxgiformat::DXGI_FORMAT_R8G8B8A8_UNORM;
        let resource = self
            .device
            .create_texture2d_buffer(width.into(), height, format, true)?;
        let size = (width, height);
        Ok(Image { resource, size })
    }

    unsafe fn destroy_image(&self, image: &Self::Image) -> Result<(), Error> {
        image.resource.destroy();
        Ok(())
    }

    fn create_cmd_buf(&self) -> Result<Self::CmdBuf, Error> {
        let list_type = d3d12::D3D12_COMMAND_LIST_TYPE_DIRECT;
        let allocator = self.free_allocators.lock().unwrap().pop();
        let allocator = if let Some(allocator) = allocator {
            allocator
        } else {
            unsafe { self.device.create_command_allocator(list_type)? }
        };
        let node_mask = 0;
        unsafe {
            let c = self
                .device
                .create_graphics_command_list(list_type, &allocator, None, node_mask)?;
            let free_allocators = Arc::downgrade(&self.free_allocators);
            Ok(CmdBuf {
                c,
                allocator: Some(allocator.clone()),
                allocator_clone: allocator,
                free_allocators,
            })
        }
    }

    unsafe fn destroy_cmd_buf(&self, _cmd_buf: Self::CmdBuf) -> Result<(), Error> {
        Ok(())
    }

    fn create_query_pool(&self, n_queries: u32) -> Result<Self::QueryPool, Error> {
        unsafe {
            let heap = self
                .device
                .create_query_heap(d3d12::D3D12_QUERY_HEAP_TYPE_TIMESTAMP, n_queries)?;
            let buf = self.create_readback_buffer((n_queries * 8) as u64)?;
            Ok(QueryPool {
                heap,
                buf,
                n_queries,
            })
        }
    }

    unsafe fn fetch_query_pool(&self, pool: &Self::QueryPool) -> Result<Vec<f64>, Error> {
        let mut buf = vec![0u64; pool.n_queries as usize];
        self.read_buffer(
            &pool.buf,
            buf.as_mut_ptr() as *mut u8,
            0,
            mem::size_of_val(buf.as_slice()) as u64,
        )?;
        let ts0 = buf[0];
        let tsp = (self.ts_freq as f64).recip();
        let result = buf[1..]
            .iter()
            .map(|ts| ts.wrapping_sub(ts0) as f64 * tsp)
            .collect();
        Ok(result)
    }

    unsafe fn run_cmd_bufs(
        &self,
        cmd_bufs: &[&Self::CmdBuf],
        _wait_semaphores: &[&Self::Semaphore],
        _signal_semaphores: &[&Self::Semaphore],
        fence: Option<&mut Self::Fence>,
    ) -> Result<(), Error> {
        // TODO: handle semaphores
        let lists = cmd_bufs
            .iter()
            .map(|c| c.c.as_raw_command_list())
            .collect::<SmallVec<[_; 4]>>();
        self.command_queue.execute_command_lists(&lists);
        for c in cmd_bufs {
            c.c.reset(&c.allocator_clone, None);
        }
        if let Some(fence) = fence {
            let val = fence.val.get() + 1;
            fence.val.set(val);
            self.command_queue.signal(&fence.fence, val)?;
            fence.fence.set_event_on_completion(&fence.event, val)?;
        }
        Ok(())
    }

    unsafe fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        dst: *mut u8,
        offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        buffer
            .resource
            .read_resource(dst, offset as usize, size as usize)?;
        Ok(())
    }

    unsafe fn write_buffer(
        &self,
        buffer: &Self::Buffer,
        contents: *const u8,
        offset: u64,
        size: u64,
    ) -> Result<(), Error> {
        buffer
            .resource
            .write_resource(contents, offset as usize, size as usize)?;
        Ok(())
    }

    unsafe fn create_semaphore(&self) -> Result<Self::Semaphore, Error> {
        Ok(Semaphore)
    }

    unsafe fn create_fence(&self, signaled: bool) -> Result<Self::Fence, Error> {
        let fence = self.device.create_fence(0)?;
        let event = wrappers::Event::create(false, signaled)?;
        let val = Cell::new(0);
        Ok(Fence { fence, event, val })
    }

    unsafe fn destroy_fence(&self, _fence: Self::Fence) -> Result<(), Error> {
        Ok(())
    }

    unsafe fn wait_and_reset(&self, fences: Vec<&mut Self::Fence>) -> Result<(), Error> {
        for fence in fences {
            // TODO: probably handle errors here.
            let _status = fence.event.wait(winapi::um::winbase::INFINITE);
        }
        Ok(())
    }

    unsafe fn get_fence_status(&self, fence: &mut Self::Fence) -> Result<bool, Error> {
        let fence_val = fence.fence.get_value();
        Ok(fence_val == fence.val.get())
    }

    fn query_gpu_info(&self) -> crate::GpuInfo {
        self.gpu_info.clone()
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

impl Dx12Device {
    fn create_readback_buffer(&self, size: u64) -> Result<Buffer, Error> {
        unsafe {
            let resource = self.device.create_buffer(
                size.try_into()?,
                d3d12::D3D12_HEAP_TYPE_READBACK,
                d3d12::D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                d3d12::D3D12_MEMORY_POOL_UNKNOWN,
                d3d12::D3D12_RESOURCE_STATE_COPY_DEST,
                d3d12::D3D12_RESOURCE_FLAG_NONE,
            )?;
            Ok(Buffer { resource, size })
        }
    }
}

impl crate::backend::CmdBuf<Dx12Device> for CmdBuf {
    unsafe fn begin(&mut self) {}

    unsafe fn finish(&mut self) {
        let _ = self.c.close();
        // This is a bit of a mess. Returning the allocator to the free pool
        // makes sense if the command list will be dropped, but not if it will
        // be reused. Probably need to implement some logic on drop.
        if let Some(free_allocators) = self.free_allocators.upgrade() {
            free_allocators
                .lock()
                .unwrap()
                .push(self.allocator.take().unwrap());
        }
    }

    unsafe fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        descriptor_set: &DescriptorSet,
        workgroup_count: (u32, u32, u32),
        _workgroup_size: (u32, u32, u32),
    ) {
        self.c.set_pipeline_state(&pipeline.pipeline_state);
        self.c
            .set_compute_pipeline_root_signature(&pipeline.root_signature);
        self.c.set_descriptor_heaps(&[&descriptor_set.0]);
        self.c.set_compute_root_descriptor_table(
            0,
            descriptor_set.0.get_gpu_descriptor_handle_at_offset(0),
        );
        self.c
            .dispatch(workgroup_count.0, workgroup_count.1, workgroup_count.2);
    }

    unsafe fn memory_barrier(&mut self) {
        // See comments in CommandBuffer::pipeline_barrier in gfx-hal dx12 backend.
        // The "proper" way to do this would be to name the actual buffers participating
        // in the barrier. But it seems like this is a reasonable way to create a
        // global barrier.
        let bar = wrappers::create_uav_resource_barrier(ptr::null_mut());
        self.c.resource_barrier(&[bar]);
    }

    unsafe fn host_barrier(&mut self) {
        // My understanding is that a host barrier is not needed, but am still hunting
        // down an authoritative source for that. Among other things, the docs for
        // Map suggest that it does the needed visibility operation.
        //
        // https://docs.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12resource-map
    }

    unsafe fn image_barrier(
        &mut self,
        image: &Image,
        src_layout: crate::ImageLayout,
        dst_layout: crate::ImageLayout,
    ) {
        let src_state = resource_state_for_image_layout(src_layout);
        let dst_state = resource_state_for_image_layout(dst_layout);
        if src_state != dst_state {
            let bar = wrappers::create_transition_resource_barrier(
                image.resource.get_mut(),
                src_state,
                dst_state,
            );
            self.c.resource_barrier(&[bar]);
        }
        // Always do a memory barrier in case of UAV image access. We probably
        // want to make these barriers more precise.
        self.memory_barrier();
    }

    unsafe fn clear_buffer(&self, buffer: &Buffer, size: Option<u64>) {
        // Open question: do we call ClearUnorderedAccessViewUint or dispatch a
        // compute shader? Either way we will need descriptors here.
        todo!()
    }

    unsafe fn copy_buffer(&self, src: &Buffer, dst: &Buffer) {
        // TODO: consider using copy_resource here (if sizes match)
        let size = src.size.min(dst.size);
        self.c.copy_buffer(&dst.resource, 0, &src.resource, 0, size);
    }

    unsafe fn copy_image_to_buffer(&self, src: &Image, dst: &Buffer) {
        self.c
            .copy_texture_to_buffer(&src.resource, &dst.resource, src.size.0, src.size.1);
    }

    unsafe fn copy_buffer_to_image(&self, src: &Buffer, dst: &Image) {
        self.c
            .copy_buffer_to_texture(&src.resource, &dst.resource, dst.size.0, dst.size.1);
    }

    unsafe fn blit_image(&self, src: &Image, dst: &Image) {
        self.c.copy_resource(&src.resource, &dst.resource);
    }

    unsafe fn reset_query_pool(&mut self, _pool: &QueryPool) {}

    unsafe fn write_timestamp(&mut self, pool: &QueryPool, query: u32) {
        self.c.end_timing_query(&pool.heap, query);
    }

    unsafe fn finish_timestamps(&mut self, pool: &QueryPool) {
        self.c
            .resolve_timing_query_data(&pool.heap, 0, pool.n_queries, &pool.buf.resource, 0);
    }
}

impl crate::backend::PipelineBuilder<Dx12Device> for PipelineBuilder {
    fn add_buffers(&mut self, n_buffers: u32) {
        // Note: if the buffer is readonly, then it needs to be bound
        // as an SRV, not a UAV. I think that requires distinguishing
        // readonly and read-write cases in pipeline and descriptor set
        // creation. For now we punt.
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
        // These are UAV images, so the descriptor type is the same as buffers.
        self.add_buffers(n_images);
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

impl crate::backend::DescriptorSetBuilder<Dx12Device> for DescriptorSetBuilder {
    fn add_buffers(&mut self, buffers: &[&Buffer]) {
        // Note: we could get rid of the clone here (which is an AddRef)
        // and store a raw pointer, as it's a safety precondition that
        // the resources are kept alive til build.
        self.buffers.extend(buffers.iter().copied().cloned());
    }

    fn add_images(&mut self, images: &[&Image]) {
        self.images.extend(images.iter().copied().cloned());
    }

    fn add_textures(&mut self, images: &[&Image]) {
        todo!()
    }

    unsafe fn build(
        self,
        device: &Dx12Device,
        _pipeline: &Pipeline,
    ) -> Result<DescriptorSet, Error> {
        let n_descriptors = self.buffers.len() + self.images.len();
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
        for image in self.images {
            device.device.create_unordered_access_view(
                &image.resource,
                heap.get_cpu_descriptor_handle_at_offset(ix),
            );
            ix += 1;
        }
        Ok(DescriptorSet(heap))
    }
}

impl MemoryArchitecture {
    // See https://msdn.microsoft.com/de-de/library/windows/desktop/dn788678(v=vs.85).aspx

    fn page_property(&self, usage: BufferUsage) -> d3d12::D3D12_CPU_PAGE_PROPERTY {
        if usage.contains(BufferUsage::MAP_READ) {
            d3d12::D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
        } else if usage.contains(BufferUsage::MAP_WRITE) {
            if *self == MemoryArchitecture::CacheCoherentUMA {
                d3d12::D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
            } else {
                d3d12::D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE
            }
        } else {
            d3d12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE
        }
    }

    fn memory_pool(&self, usage: BufferUsage) -> d3d12::D3D12_MEMORY_POOL {
        if *self == MemoryArchitecture::NUMA
            && !usage.intersects(BufferUsage::MAP_READ | BufferUsage::MAP_WRITE)
        {
            d3d12::D3D12_MEMORY_POOL_L1
        } else {
            d3d12::D3D12_MEMORY_POOL_L0
        }
    }
}

fn resource_state_for_image_layout(layout: ImageLayout) -> d3d12::D3D12_RESOURCE_STATES {
    match layout {
        ImageLayout::Undefined => d3d12::D3D12_RESOURCE_STATE_COMMON,
        ImageLayout::Present => d3d12::D3D12_RESOURCE_STATE_PRESENT,
        ImageLayout::BlitSrc => d3d12::D3D12_RESOURCE_STATE_COPY_SOURCE,
        ImageLayout::BlitDst => d3d12::D3D12_RESOURCE_STATE_COPY_DEST,
        ImageLayout::General => d3d12::D3D12_RESOURCE_STATE_COMMON,
        ImageLayout::ShaderRead => d3d12::D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    }
}

impl Dx12Swapchain {
    pub unsafe fn next(&mut self) -> Result<(usize, Semaphore), Error> {
        let idx = self.swapchain.get_current_back_buffer_index();
        Ok((idx as usize, Semaphore))
    }

    pub unsafe fn image(&self, idx: usize) -> Image {
        let buffer = self.swapchain.get_buffer(idx as u32);
        Image {
            resource: buffer,
            size: self.size,
        }
    }

    pub unsafe fn present(
        &self,
        _image_idx: usize,
        _semaphores: &[&Semaphore],
    ) -> Result<bool, Error> {
        self.swapchain.present(1, 0)?;
        Ok(false)
    }
}
