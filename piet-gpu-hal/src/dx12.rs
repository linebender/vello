//! DX12 implemenation of HAL trait.

mod descriptor;
mod error;
mod wrappers;

use std::{
    cell::Cell,
    convert::{TryFrom, TryInto},
    mem, ptr,
    sync::{Arc, Mutex},
};

#[allow(unused)]
use winapi::shared::dxgi1_3; // for error reporting in debug mode
use winapi::shared::minwindef::TRUE;
use winapi::shared::{dxgi, dxgi1_2, dxgitype};
use winapi::um::d3d12;

use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};

use smallvec::SmallVec;

use crate::{
    BindType, BufferUsage, ComputePassDescriptor, Error, GpuInfo, ImageFormat, ImageLayout,
    MapMode, WorkgroupLimits,
};

use self::{
    descriptor::{CpuHeapRefOwned, DescriptorPool, GpuHeapRefOwned},
    wrappers::{
        CommandAllocator, CommandQueue, DescriptorHeap, Device, Factory4, Resource, ShaderByteCode,
    },
};

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
    command_queue: CommandQueue,
    ts_freq: u64,
    gpu_info: GpuInfo,
    memory_arch: MemoryArchitecture,
    descriptor_pool: Mutex<DescriptorPool>,
}

#[derive(Clone)]
pub struct Buffer {
    resource: Resource,
    pub size: u64,
    // Always present except for query readback buffer.
    cpu_ref: Option<Arc<CpuHeapRefOwned>>,
    // Present when created with CLEAR usage. Heap is here for
    // the same reason it's in DescriptorSet, and might be removed
    // when CmdBuf has access to the descriptor pool.
    gpu_ref: Option<(Arc<GpuHeapRefOwned>, DescriptorHeap)>,
}

#[derive(Clone)]
pub struct Image {
    resource: Resource,
    // Present except for swapchain images.
    cpu_ref: Option<Arc<CpuHeapRefOwned>>,
    size: (u32, u32),
}

pub struct CmdBuf {
    c: wrappers::GraphicsCommandList,
    allocator: CommandAllocator,
    needs_reset: bool,
    end_query: Option<(wrappers::QueryHeap, u32)>,
}

pub struct Pipeline {
    pipeline_state: wrappers::PipelineState,
    root_signature: wrappers::RootSignature,
}

pub struct DescriptorSet {
    gpu_ref: GpuHeapRefOwned,
    // Note: the heap is only needed here so CmdBuf::dispatch can get
    // use it easily. If CmdBuf had a reference to the Device (or just
    // the descriptor pool), we could get rid of this.
    heap: DescriptorHeap,
}

pub struct QueryPool {
    heap: wrappers::QueryHeap,
    // Maybe this should just be a Resource, not a full Buffer.
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
pub struct DescriptorSetBuilder {
    handles: SmallVec<[d3d12::D3D12_CPU_DESCRIPTOR_HANDLE; 16]>,
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

    /// Create a surface for the specified window handle.
    pub fn surface(
        &self,
        window_handle: &dyn HasRawWindowHandle,
    ) -> Result<Dx12Surface, Error> {
        if let RawWindowHandle::Windows(w) = window_handle.raw_window_handle() {
            let hwnd = w.hwnd as *mut _;
            Ok(Dx12Surface { hwnd })
        } else {
            Err("can't create surface for window handle".into())
        }
    }

    /// Get a device suitable for compute workloads.
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
            let descriptor_pool = Default::default();
            Ok(Dx12Device {
                device,
                command_queue,
                ts_freq,
                memory_arch,
                gpu_info,
                descriptor_pool,
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

    type DescriptorSetBuilder = DescriptorSetBuilder;

    type Sampler = ();

    // Currently due to type inflexibility this is hardcoded to either HLSL or
    // DXIL, but it would be nice to be able to handle both at runtime.
    type ShaderSource = [u8];

    fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Self::Buffer, Error> {
        // TODO: consider supporting BufferUsage::QUERY_RESOLVE here rather than
        // having a separate function.
        unsafe {
            let page_property = self.memory_arch.page_property(usage);
            let memory_pool = self.memory_arch.memory_pool(usage);
            //TODO: consider flag D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS?
            let flags = d3d12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            let resource = self.device.create_buffer(
                size,
                d3d12::D3D12_HEAP_TYPE_CUSTOM,
                page_property,
                memory_pool,
                d3d12::D3D12_RESOURCE_STATE_COMMON,
                flags,
            )?;
            let mut descriptor_pool = self.descriptor_pool.lock().unwrap();
            let cpu_ref = Arc::new(descriptor_pool.alloc_cpu(&self.device)?);
            let cpu_handle = descriptor_pool.cpu_handle(&cpu_ref);
            self.device
                .create_byte_addressed_buffer_unordered_access_view(
                    &resource,
                    cpu_handle,
                    0,
                    (size / 4).try_into()?,
                );
            let gpu_ref = if usage.contains(BufferUsage::CLEAR) {
                let gpu_ref = Arc::new(descriptor_pool.alloc_gpu(&self.device, 1)?);
                let gpu_handle = descriptor_pool.cpu_handle_of_gpu(&gpu_ref, 0);
                self.device.copy_descriptors(
                    &[gpu_handle],
                    &[1],
                    &[cpu_handle],
                    &[1],
                    d3d12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                );
                let heap = descriptor_pool.gpu_heap(&gpu_ref).to_owned();
                Some((gpu_ref, heap))
            } else {
                None
            };
            Ok(Buffer {
                resource,
                size,
                cpu_ref: Some(cpu_ref),
                gpu_ref,
            })
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
        format: ImageFormat,
    ) -> Result<Self::Image, Error> {
        let format = match format {
            ImageFormat::A8 => winapi::shared::dxgiformat::DXGI_FORMAT_R8_UNORM,
            ImageFormat::Rgba8 | ImageFormat::Surface => winapi::shared::dxgiformat::DXGI_FORMAT_R8G8B8A8_UNORM,
        };
        let resource = self
            .device
            .create_texture2d_buffer(width.into(), height, format, true)?;

        let mut descriptor_pool = self.descriptor_pool.lock().unwrap();
        let cpu_ref = Arc::new(descriptor_pool.alloc_cpu(&self.device)?);
        let cpu_handle = descriptor_pool.cpu_handle(&cpu_ref);
        self.device
            .create_unordered_access_view(&resource, cpu_handle);
        let size = (width, height);
        Ok(Image {
            resource,
            cpu_ref: Some(cpu_ref),
            size,
        })
    }

    unsafe fn destroy_image(&self, image: &Self::Image) -> Result<(), Error> {
        image.resource.destroy();
        Ok(())
    }

    fn create_cmd_buf(&self) -> Result<Self::CmdBuf, Error> {
        let list_type = d3d12::D3D12_COMMAND_LIST_TYPE_DIRECT;
        let allocator = unsafe { self.device.create_command_allocator(list_type)? };
        let node_mask = 0;
        unsafe {
            let c = self
                .device
                .create_graphics_command_list(list_type, &allocator, None, node_mask)?;
            Ok(CmdBuf {
                c,
                allocator,
                needs_reset: false,
                end_query: None,
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
        let size = mem::size_of_val(buf.as_slice());
        let mapped = self.map_buffer(&pool.buf, 0, size as u64, MapMode::Read)?;
        std::ptr::copy_nonoverlapping(mapped, buf.as_mut_ptr() as *mut u8, size);
        self.unmap_buffer(&pool.buf, 0, size as u64, MapMode::Read)?;
        let tsp = (self.ts_freq as f64).recip();
        let result = buf.iter().map(|ts| *ts as f64 * tsp).collect();
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
        if let Some(fence) = fence {
            let val = fence.val.get() + 1;
            fence.val.set(val);
            self.command_queue.signal(&fence.fence, val)?;
            fence.fence.set_event_on_completion(&fence.event, val)?;
        }
        Ok(())
    }

    unsafe fn map_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        size: u64,
        mode: MapMode,
    ) -> Result<*mut u8, Error> {
        let mapped = buffer.resource.map_buffer(offset, size, mode)?;
        Ok(mapped)
    }

    unsafe fn unmap_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        size: u64,
        mode: MapMode,
    ) -> Result<(), Error> {
        buffer.resource.unmap_buffer(offset, size, mode)?;
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

    unsafe fn create_compute_pipeline(
        &self,
        code: &Self::ShaderSource,
        bind_types: &[BindType],
    ) -> Result<Pipeline, Error> {
        if u32::try_from(bind_types.len()).is_err() {
            panic!("bind type length overflow");
        }
        let mut ranges = Vec::new();
        let mut i = 0;
        fn map_range_type(bind_type: BindType) -> d3d12::D3D12_DESCRIPTOR_RANGE_TYPE {
            match bind_type {
                BindType::Buffer | BindType::Image | BindType::ImageRead => {
                    d3d12::D3D12_DESCRIPTOR_RANGE_TYPE_UAV
                }
                BindType::BufReadOnly => d3d12::D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            }
        }
        while i < bind_types.len() {
            let range_type = map_range_type(bind_types[i]);
            let mut end = i + 1;
            while end < bind_types.len() && map_range_type(bind_types[end]) == range_type {
                end += 1;
            }
            let n_descriptors = (end - i) as u32;
            ranges.push(d3d12::D3D12_DESCRIPTOR_RANGE {
                RangeType: range_type,
                NumDescriptors: n_descriptors,
                BaseShaderRegister: i as u32,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: d3d12::D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            });
            i = end;
        }

        // We could always have ShaderSource as [u8] even when it's HLSL, and use the
        // magic number to distinguish. In any case, for now it's hardcoded as one or
        // the other.
        /*
        // HLSL code path
        #[cfg(debug_assertions)]
        let flags = winapi::um::d3dcompiler::D3DCOMPILE_DEBUG
            | winapi::um::d3dcompiler::D3DCOMPILE_SKIP_OPTIMIZATION;
        #[cfg(not(debug_assertions))]
        let flags = 0;
        let shader_blob = ShaderByteCode::compile(code, "cs_5_1", "main", flags)?;
        let shader = ShaderByteCode::from_blob(shader_blob);
        */

        // DXIL code path
        let shader = ShaderByteCode::from_slice(code);

        let mut root_parameter = d3d12::D3D12_ROOT_PARAMETER {
            ParameterType: d3d12::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            ShaderVisibility: d3d12::D3D12_SHADER_VISIBILITY_ALL,
            ..mem::zeroed()
        };
        *root_parameter.u.DescriptorTable_mut() = d3d12::D3D12_ROOT_DESCRIPTOR_TABLE {
            NumDescriptorRanges: ranges.len() as u32,
            pDescriptorRanges: ranges.as_ptr(),
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
        let root_signature = self.device.create_root_signature(0, root_signature_blob)?;
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
        let pipeline_state = self.device.create_compute_pipeline_state(&desc)?;

        Ok(Pipeline {
            pipeline_state,
            root_signature,
        })
    }

    unsafe fn descriptor_set_builder(&self) -> Self::DescriptorSetBuilder {
        DescriptorSetBuilder::default()
    }

    unsafe fn update_buffer_descriptor(
        &self,
        ds: &mut Self::DescriptorSet,
        index: u32,
        buf: &Self::Buffer,
    ) {
        let src_cpu_ref = buf.cpu_ref.as_ref().unwrap().handle();
        ds.gpu_ref
            .copy_one_descriptor(&self.device, src_cpu_ref, index);
    }

    unsafe fn update_image_descriptor(
        &self,
        ds: &mut Self::DescriptorSet,
        index: u32,
        image: &Self::Image,
    ) {
        let src_cpu_ref = image.cpu_ref.as_ref().unwrap().handle();
        ds.gpu_ref
            .copy_one_descriptor(&self.device, src_cpu_ref, index);
    }

    unsafe fn create_sampler(&self, _params: crate::SamplerParams) -> Result<Self::Sampler, Error> {
        todo!()
    }
}

impl Dx12Device {
    fn create_readback_buffer(&self, size: u64) -> Result<Buffer, Error> {
        unsafe {
            let resource = self.device.create_buffer(
                size,
                d3d12::D3D12_HEAP_TYPE_READBACK,
                d3d12::D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                d3d12::D3D12_MEMORY_POOL_UNKNOWN,
                d3d12::D3D12_RESOURCE_STATE_COPY_DEST,
                d3d12::D3D12_RESOURCE_FLAG_NONE,
            )?;
            let cpu_ref = None;
            let gpu_ref = None;
            Ok(Buffer {
                resource,
                size,
                cpu_ref,
                gpu_ref,
            })
        }
    }
}

impl crate::backend::CmdBuf<Dx12Device> for CmdBuf {
    unsafe fn begin(&mut self) {
        if self.needs_reset {}
    }

    unsafe fn finish(&mut self) {
        let _ = self.c.close();
        self.needs_reset = true;
    }

    unsafe fn flush(&mut self) {}

    unsafe fn reset(&mut self) -> bool {
        self.allocator.reset().is_ok() && self.c.reset(&self.allocator, None).is_ok()
    }

    unsafe fn begin_compute_pass(&mut self, desc: &ComputePassDescriptor) {
        if let Some((pool, start, end)) = &desc.timer_queries {
            #[allow(irrefutable_let_patterns)]
            if let crate::hub::QueryPool::Dx12(pool) = pool {
                self.write_timestamp(pool, *start);
                self.end_query = Some((pool.heap.clone(), *end));
            }
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
        // TODO: persist heap ix and only set if changed.
        self.c.set_descriptor_heaps(&[&descriptor_set.heap]);
        self.c
            .set_compute_root_descriptor_table(0, descriptor_set.gpu_ref.gpu_handle());
        self.c
            .dispatch(workgroup_count.0, workgroup_count.1, workgroup_count.2);
    }

    unsafe fn end_compute_pass(&mut self) {
        if let Some((heap, end)) = self.end_query.take() {
            self.c.end_timing_query(&heap, end);
        }
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

    unsafe fn clear_buffer(&mut self, buffer: &Buffer, size: Option<u64>) {
        let cpu_ref = buffer.cpu_ref.as_ref().unwrap();
        let (gpu_ref, heap) = buffer
            .gpu_ref
            .as_ref()
            .expect("Need to set CLEAR usage on buffer");
        // Same TODO as dispatch: track and only set if changed.
        self.c.set_descriptor_heaps(&[heap]);
        // Discussion question: would compute shader be faster? Should measure.
        self.c.clear_uav(
            gpu_ref.gpu_handle(),
            cpu_ref.handle(),
            &buffer.resource,
            0,
            size,
        );
    }

    unsafe fn copy_buffer(&mut self, src: &Buffer, dst: &Buffer) {
        // TODO: consider using copy_resource here (if sizes match)
        let size = src.size.min(dst.size);
        self.c.copy_buffer(&dst.resource, 0, &src.resource, 0, size);
    }

    unsafe fn copy_image_to_buffer(&mut self, src: &Image, dst: &Buffer) {
        self.c
            .copy_texture_to_buffer(&src.resource, &dst.resource, src.size.0, src.size.1);
    }

    unsafe fn copy_buffer_to_image(&mut self, src: &Buffer, dst: &Image) {
        self.c
            .copy_buffer_to_texture(&src.resource, &dst.resource, dst.size.0, dst.size.1);
    }

    unsafe fn blit_image(&mut self, src: &Image, dst: &Image) {
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

impl crate::backend::DescriptorSetBuilder<Dx12Device> for DescriptorSetBuilder {
    fn add_buffers(&mut self, buffers: &[&Buffer]) {
        for buf in buffers {
            self.handles.push(buf.cpu_ref.as_ref().unwrap().handle());
        }
    }

    fn add_images(&mut self, images: &[&Image]) {
        for img in images {
            self.handles.push(img.cpu_ref.as_ref().unwrap().handle());
        }
    }

    fn add_textures(&mut self, images: &[&Image]) {
        for img in images {
            self.handles.push(img.cpu_ref.as_ref().unwrap().handle());
        }
    }

    unsafe fn build(
        self,
        device: &Dx12Device,
        _pipeline: &Pipeline,
    ) -> Result<DescriptorSet, Error> {
        let mut descriptor_pool = device.descriptor_pool.lock().unwrap();
        let n_descriptors = self.handles.len().try_into()?;
        let gpu_ref = descriptor_pool.alloc_gpu(&device.device, n_descriptors)?;
        gpu_ref.copy_descriptors(&device.device, &self.handles);
        let heap = descriptor_pool.gpu_heap(&gpu_ref).to_owned();
        Ok(DescriptorSet { gpu_ref, heap })
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
            cpu_ref: None,
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
