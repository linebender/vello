// Copyright © 2019 piet-gpu developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::dx12::error::{self, error_if_failed_else_unit, explain_error, Error};
use crate::MapMode;
use smallvec::SmallVec;
use std::convert::{TryFrom, TryInto};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::{ffi, mem, ptr};
use winapi::shared::{dxgi, dxgi1_2, dxgi1_3, dxgi1_4, dxgiformat, dxgitype, minwindef, windef};
use winapi::um::d3dcommon::ID3DBlob;
use winapi::um::{
    d3d12, d3d12sdklayers, d3dcommon, d3dcompiler, dxgidebug, handleapi, synchapi, winnt,
};
use winapi::Interface;
use wio::com::ComPtr;

// everything is ripped from d3d12-rs, but wio::com::ComPtr, and winapi are used more directly

#[derive(Clone)]
pub struct Heap(pub ComPtr<d3d12::ID3D12Heap>);

pub struct Resource {
    // Note: the use of AtomicPtr is to support explicit destruction,
    // similar to Vulkan.
    ptr: AtomicPtr<d3d12::ID3D12Resource>,
}

#[derive(Clone)]
pub struct Adapter1(pub ComPtr<dxgi::IDXGIAdapter1>);
#[derive(Clone)]
pub struct Factory2(pub ComPtr<dxgi1_2::IDXGIFactory2>);
#[derive(Clone)]
pub struct Factory4(pub ComPtr<dxgi1_4::IDXGIFactory4>);
#[derive(Clone)]
pub struct SwapChain3(pub ComPtr<dxgi1_4::IDXGISwapChain3>);

#[derive(Clone)]
pub struct Device(pub ComPtr<d3d12::ID3D12Device>);

#[derive(Clone)]
pub struct CommandQueue(pub ComPtr<d3d12::ID3D12CommandQueue>);

#[derive(Clone)]
pub struct CommandAllocator(pub ComPtr<d3d12::ID3D12CommandAllocator>);

pub type CpuDescriptor = d3d12::D3D12_CPU_DESCRIPTOR_HANDLE;
pub type GpuDescriptor = d3d12::D3D12_GPU_DESCRIPTOR_HANDLE;

#[derive(Clone)]
pub struct DescriptorHeap(ComPtr<d3d12::ID3D12DescriptorHeap>);

#[derive(Clone)]
pub struct RootSignature(pub ComPtr<d3d12::ID3D12RootSignature>);

#[derive(Clone)]
pub struct CommandSignature(pub ComPtr<d3d12::ID3D12CommandSignature>);
#[derive(Clone)]
pub struct GraphicsCommandList(pub ComPtr<d3d12::ID3D12GraphicsCommandList>);

pub struct Event(pub winnt::HANDLE);
#[derive(Clone)]
pub struct Fence(pub ComPtr<d3d12::ID3D12Fence>);

#[derive(Clone)]
pub struct PipelineState(pub ComPtr<d3d12::ID3D12PipelineState>);

#[derive(Clone)]
pub struct CachedPSO(d3d12::D3D12_CACHED_PIPELINE_STATE);

#[derive(Clone)]
pub struct Blob(pub ComPtr<d3dcommon::ID3DBlob>);

#[derive(Clone)]
pub struct ShaderByteCode {
    pub bytecode: d3d12::D3D12_SHADER_BYTECODE,
}

#[derive(Clone)]
pub struct QueryHeap(pub ComPtr<d3d12::ID3D12QueryHeap>);

impl Resource {
    pub unsafe fn new(ptr: *mut d3d12::ID3D12Resource) -> Resource {
        Resource {
            ptr: AtomicPtr::new(ptr),
        }
    }

    pub fn get(&self) -> *const d3d12::ID3D12Resource {
        self.get_mut()
    }

    pub fn get_mut(&self) -> *mut d3d12::ID3D12Resource {
        self.ptr.load(Ordering::Relaxed)
    }

    // Safety: call only single-threaded.
    pub unsafe fn destroy(&self) {
        (*self.get()).Release();
        self.ptr.store(ptr::null_mut(), Ordering::Relaxed);
    }

    pub unsafe fn map_buffer(
        &self,
        offset: u64,
        size: u64,
        mode: MapMode,
    ) -> Result<*mut u8, Error> {
        let mut mapped_memory: *mut u8 = ptr::null_mut();
        let (begin, end) = match mode {
            MapMode::Read => (offset as usize, (offset + size) as usize),
            MapMode::Write => (0, 0),
        };
        let range = d3d12::D3D12_RANGE {
            Begin: begin,
            End: end,
        };
        explain_error(
            (*self.get()).Map(0, &range, &mut mapped_memory as *mut _ as *mut _),
            "could not map GPU mem to CPU mem",
        )?;
        Ok(mapped_memory.add(offset as usize))
    }

    pub unsafe fn unmap_buffer(&self, offset: u64, size: u64, mode: MapMode) -> Result<(), Error> {
        let (begin, end) = match mode {
            MapMode::Read => (0, 0),
            MapMode::Write => (offset as usize, (offset + size) as usize),
        };
        let range = d3d12::D3D12_RANGE {
            Begin: begin,
            End: end,
        };
        (*self.get()).Unmap(0, &range);
        Ok(())
    }
}

impl Drop for Resource {
    fn drop(&mut self) {
        unsafe {
            let ptr = self.get();
            if !ptr.is_null() {
                (*ptr).Release();
            }
        }
    }
}

impl Clone for Resource {
    fn clone(&self) -> Self {
        unsafe {
            let ptr = self.get_mut();
            (*ptr).AddRef();
            Resource {
                ptr: AtomicPtr::new(ptr),
            }
        }
    }
}

impl Factory4 {
    pub unsafe fn create(flags: minwindef::UINT) -> Result<Factory4, Error> {
        let mut factory = ptr::null_mut();

        explain_error(
            dxgi1_3::CreateDXGIFactory2(
                flags,
                &dxgi1_4::IDXGIFactory4::uuidof(),
                &mut factory as *mut _ as *mut _,
            ),
            "error creating DXGI factory",
        )?;

        Ok(Factory4(ComPtr::from_raw(factory)))
    }

    pub unsafe fn enumerate_adapters(&self, id: u32) -> Result<Adapter1, Error> {
        let mut adapter = ptr::null_mut();
        error_if_failed_else_unit(self.0.EnumAdapters1(id, &mut adapter))?;
        let mut desc = mem::zeroed();
        (*adapter).GetDesc(&mut desc);
        //println!("desc: {:?}", desc.Description);
        Ok(Adapter1(ComPtr::from_raw(adapter)))
    }

    pub unsafe fn create_swapchain_for_hwnd(
        &self,
        command_queue: &CommandQueue,
        hwnd: windef::HWND,
        desc: dxgi1_2::DXGI_SWAP_CHAIN_DESC1,
    ) -> Result<SwapChain3, Error> {
        let mut swap_chain = ptr::null_mut();
        explain_error(
            self.0.CreateSwapChainForHwnd(
                command_queue.0.as_raw() as *mut _,
                hwnd,
                &desc,
                ptr::null(),
                ptr::null_mut(),
                &mut swap_chain as *mut _ as *mut _,
            ),
            "could not creation swapchain for hwnd",
        )?;

        Ok(SwapChain3(ComPtr::from_raw(swap_chain)))
    }
}

impl CommandQueue {
    pub unsafe fn signal(&self, fence: &Fence, value: u64) -> Result<(), Error> {
        explain_error(
            self.0.Signal(fence.0.as_raw(), value),
            "error setting signal",
        )
    }

    pub unsafe fn execute_command_lists(&self, command_lists: &[*mut d3d12::ID3D12CommandList]) {
        let num_command_lists = command_lists.len().try_into().unwrap();
        self.0
            .ExecuteCommandLists(num_command_lists, command_lists.as_ptr());
    }

    pub unsafe fn get_timestamp_frequency(&self) -> Result<u64, Error> {
        let mut result: u64 = 0;

        explain_error(
            self.0.GetTimestampFrequency(&mut result),
            "could not get timestamp frequency",
        )?;

        Ok(result)
    }
}

impl SwapChain3 {
    pub unsafe fn get_buffer(&self, id: u32) -> Resource {
        let mut resource = ptr::null_mut();
        error::error_if_failed_else_unit(self.0.GetBuffer(
            id,
            &d3d12::ID3D12Resource::uuidof(),
            &mut resource as *mut _ as *mut _,
        ))
        .expect("SwapChain3 could not get buffer");

        Resource::new(resource)
    }

    pub unsafe fn get_current_back_buffer_index(&self) -> u32 {
        self.0.GetCurrentBackBufferIndex()
    }

    pub unsafe fn present(&self, interval: u32, flags: u32) -> Result<(), Error> {
        error::error_if_failed_else_unit(self.0.Present1(
            interval,
            flags,
            &dxgi1_2::DXGI_PRESENT_PARAMETERS { ..mem::zeroed() } as *const _,
        ))
    }
}

impl Blob {
    #[allow(unused)]
    pub unsafe fn print_to_console(blob: &Blob) {
        println!("==SHADER COMPILE MESSAGES==");
        let message = {
            let pointer = blob.0.GetBufferPointer();
            let size = blob.0.GetBufferSize();
            let slice = std::slice::from_raw_parts(pointer as *const u8, size as usize);
            String::from_utf8_lossy(slice).into_owned()
        };
        println!("{}", message);
        println!("===========================");
    }
}

impl Device {
    pub unsafe fn create_device(factory4: &Factory4) -> Result<Device, Error> {
        let mut id = 0;

        loop {
            // This always returns DXGI_ERROR_NOT_FOUND if no suitable adapter is found.
            // Might be slightly more useful to retain the error from the attempt to create.
            let adapter = factory4.enumerate_adapters(id)?;

            if let Ok(device) =
                Self::create_using_adapter(&adapter, d3dcommon::D3D_FEATURE_LEVEL_12_0)
            {
                return Ok(device);
            }
            id += 1;
        }
    }

    pub unsafe fn create_using_adapter(
        adapter: &Adapter1,
        feature_level: d3dcommon::D3D_FEATURE_LEVEL,
    ) -> Result<Device, Error> {
        let mut device = ptr::null_mut();
        error_if_failed_else_unit(d3d12::D3D12CreateDevice(
            adapter.0.as_raw() as *mut _,
            feature_level,
            &d3d12::ID3D12Device::uuidof(),
            &mut device as *mut _ as *mut _,
        ))?;

        Ok(Device(ComPtr::from_raw(device)))
    }

    pub unsafe fn create_command_allocator(
        &self,
        list_type: d3d12::D3D12_COMMAND_LIST_TYPE,
    ) -> Result<CommandAllocator, Error> {
        let mut allocator = ptr::null_mut();
        explain_error(
            self.0.CreateCommandAllocator(
                list_type,
                &d3d12::ID3D12CommandAllocator::uuidof(),
                &mut allocator as *mut _ as *mut _,
            ),
            "device could not create command allocator",
        )?;

        Ok(CommandAllocator(ComPtr::from_raw(allocator)))
    }

    pub unsafe fn create_command_queue(
        &self,
        list_type: d3d12::D3D12_COMMAND_LIST_TYPE,
        priority: minwindef::INT,
        flags: d3d12::D3D12_COMMAND_QUEUE_FLAGS,
        node_mask: minwindef::UINT,
    ) -> Result<CommandQueue, Error> {
        let desc = d3d12::D3D12_COMMAND_QUEUE_DESC {
            Type: list_type,
            Priority: priority,
            Flags: flags,
            NodeMask: node_mask,
        };

        let mut cmd_q = ptr::null_mut();
        explain_error(
            self.0.CreateCommandQueue(
                &desc,
                &d3d12::ID3D12CommandQueue::uuidof(),
                &mut cmd_q as *mut _ as *mut _,
            ),
            "device could not create command queue",
        )?;

        Ok(CommandQueue(ComPtr::from_raw(cmd_q)))
    }

    pub unsafe fn create_descriptor_heap(
        &self,
        heap_description: &d3d12::D3D12_DESCRIPTOR_HEAP_DESC,
    ) -> Result<DescriptorHeap, Error> {
        let mut heap = ptr::null_mut();
        explain_error(
            self.0.CreateDescriptorHeap(
                heap_description,
                &d3d12::ID3D12DescriptorHeap::uuidof(),
                &mut heap as *mut _ as *mut _,
            ),
            "device could not create descriptor heap",
        )?;

        Ok(DescriptorHeap(ComPtr::from_raw(heap)))
    }

    pub unsafe fn get_descriptor_increment_size(
        &self,
        heap_type: d3d12::D3D12_DESCRIPTOR_HEAP_TYPE,
    ) -> u32 {
        self.0.GetDescriptorHandleIncrementSize(heap_type)
    }

    pub unsafe fn copy_descriptors(
        &self,
        dst_starts: &[d3d12::D3D12_CPU_DESCRIPTOR_HANDLE],
        dst_sizes: &[u32],
        src_starts: &[d3d12::D3D12_CPU_DESCRIPTOR_HANDLE],
        src_sizes: &[u32],
        descriptor_heap_type: d3d12::D3D12_DESCRIPTOR_HEAP_TYPE,
    ) {
        debug_assert_eq!(dst_starts.len(), dst_sizes.len());
        debug_assert_eq!(src_starts.len(), src_sizes.len());
        debug_assert_eq!(
            src_sizes.iter().copied().sum::<u32>(),
            dst_sizes.iter().copied().sum()
        );
        self.0.CopyDescriptors(
            dst_starts.len().try_into().unwrap(),
            dst_starts.as_ptr(),
            dst_sizes.as_ptr(),
            src_starts.len().try_into().unwrap(),
            src_starts.as_ptr(),
            src_sizes.as_ptr(),
            descriptor_heap_type,
        );
    }

    pub unsafe fn copy_one_descriptor(
        &self,
        dst: d3d12::D3D12_CPU_DESCRIPTOR_HANDLE,
        src: d3d12::D3D12_CPU_DESCRIPTOR_HANDLE,
        descriptor_heap_type: d3d12::D3D12_DESCRIPTOR_HEAP_TYPE,
    ) {
        self.0
            .CopyDescriptorsSimple(1, dst, src, descriptor_heap_type);
    }

    pub unsafe fn create_compute_pipeline_state(
        &self,
        compute_pipeline_desc: &d3d12::D3D12_COMPUTE_PIPELINE_STATE_DESC,
    ) -> Result<PipelineState, Error> {
        let mut pipeline_state = ptr::null_mut();

        explain_error(
            self.0.CreateComputePipelineState(
                compute_pipeline_desc as *const _,
                &d3d12::ID3D12PipelineState::uuidof(),
                &mut pipeline_state as *mut _ as *mut _,
            ),
            "device could not create compute pipeline state",
        )?;

        Ok(PipelineState(ComPtr::from_raw(pipeline_state)))
    }

    pub unsafe fn create_root_signature(
        &self,
        node_mask: minwindef::UINT,
        blob: Blob,
    ) -> Result<RootSignature, Error> {
        let mut signature = ptr::null_mut();
        explain_error(
            self.0.CreateRootSignature(
                node_mask,
                blob.0.GetBufferPointer(),
                blob.0.GetBufferSize(),
                &d3d12::ID3D12RootSignature::uuidof(),
                &mut signature as *mut _ as *mut _,
            ),
            "device could not create root signature",
        )?;

        Ok(RootSignature(ComPtr::from_raw(signature)))
    }

    pub unsafe fn create_graphics_command_list(
        &self,
        list_type: d3d12::D3D12_COMMAND_LIST_TYPE,
        allocator: &CommandAllocator,
        initial_ps: Option<&PipelineState>,
        node_mask: minwindef::UINT,
    ) -> Result<GraphicsCommandList, Error> {
        let mut command_list = ptr::null_mut();
        let p_initial_state = initial_ps.map(|p| p.0.as_raw()).unwrap_or(ptr::null_mut());
        explain_error(
            self.0.CreateCommandList(
                node_mask,
                list_type,
                allocator.0.as_raw(),
                p_initial_state,
                &d3d12::ID3D12GraphicsCommandList::uuidof(),
                &mut command_list as *mut _ as *mut _,
            ),
            "device could not create graphics command list",
        )?;

        Ok(GraphicsCommandList(ComPtr::from_raw(command_list)))
    }

    pub unsafe fn create_byte_addressed_buffer_unordered_access_view(
        &self,
        resource: &Resource,
        descriptor: CpuDescriptor,
        first_element: u64,
        num_elements: u32,
    ) {
        // shouldn't flags be dxgiformat::DXGI_FORMAT_R32_TYPELESS?
        let mut uav_desc = d3d12::D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: dxgiformat::DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: d3d12::D3D12_UAV_DIMENSION_BUFFER,
            ..mem::zeroed()
        };
        *uav_desc.u.Buffer_mut() = d3d12::D3D12_BUFFER_UAV {
            FirstElement: first_element,
            NumElements: num_elements,
            // shouldn't StructureByteStride be 0?
            StructureByteStride: 0,
            CounterOffsetInBytes: 0,
            // shouldn't flags be d3d12::D3D12_BUFFER_UAV_FLAG_RAW?
            Flags: d3d12::D3D12_BUFFER_UAV_FLAG_RAW,
        };
        self.0
            .CreateUnorderedAccessView(resource.get_mut(), ptr::null_mut(), &uav_desc, descriptor)
    }

    pub unsafe fn create_unordered_access_view(
        &self,
        resource: &Resource,
        descriptor: CpuDescriptor,
    ) {
        self.0.CreateUnorderedAccessView(
            resource.get_mut(),
            ptr::null_mut(),
            ptr::null(),
            descriptor,
        )
    }

    pub unsafe fn create_fence(&self, initial: u64) -> Result<Fence, Error> {
        let mut fence = ptr::null_mut();
        explain_error(
            self.0.CreateFence(
                initial,
                d3d12::D3D12_FENCE_FLAG_NONE,
                &d3d12::ID3D12Fence::uuidof(),
                &mut fence as *mut _ as *mut _,
            ),
            "device could not create fence",
        )?;

        Ok(Fence(ComPtr::from_raw(fence)))
    }

    pub unsafe fn create_committed_resource(
        &self,
        heap_properties: &d3d12::D3D12_HEAP_PROPERTIES,
        flags: d3d12::D3D12_HEAP_FLAGS,
        resource_description: &d3d12::D3D12_RESOURCE_DESC,
        initial_resource_state: d3d12::D3D12_RESOURCE_STATES,
        optimized_clear_value: *const d3d12::D3D12_CLEAR_VALUE,
    ) -> Result<Resource, Error> {
        let mut resource = ptr::null_mut();

        explain_error(
            self.0.CreateCommittedResource(
                heap_properties as *const _,
                flags,
                resource_description as *const _,
                initial_resource_state,
                optimized_clear_value,
                &d3d12::ID3D12Resource::uuidof(),
                &mut resource as *mut _ as *mut _,
            ),
            "device could not create committed resource",
        )?;

        Ok(Resource::new(resource))
    }

    pub unsafe fn create_query_heap(
        &self,
        heap_type: d3d12::D3D12_QUERY_HEAP_TYPE,
        num_expected_queries: u32,
    ) -> Result<QueryHeap, Error> {
        let query_heap_desc = d3d12::D3D12_QUERY_HEAP_DESC {
            Type: heap_type,
            Count: num_expected_queries,
            NodeMask: 0,
        };

        let mut query_heap = ptr::null_mut();

        explain_error(
            self.0.CreateQueryHeap(
                &query_heap_desc as *const _,
                &d3d12::ID3D12QueryHeap::uuidof(),
                &mut query_heap as *mut _ as *mut _,
            ),
            "could not create query heap",
        )?;

        Ok(QueryHeap(ComPtr::from_raw(query_heap)))
    }

    pub unsafe fn create_buffer(
        &self,
        buffer_size_in_bytes: u64,
        heap_type: d3d12::D3D12_HEAP_TYPE,
        cpu_page: d3d12::D3D12_CPU_PAGE_PROPERTY,
        memory_pool_preference: d3d12::D3D12_MEMORY_POOL,
        init_resource_state: d3d12::D3D12_RESOURCE_STATES,
        resource_flags: d3d12::D3D12_RESOURCE_FLAGS,
    ) -> Result<Resource, Error> {
        let heap_properties = d3d12::D3D12_HEAP_PROPERTIES {
            Type: heap_type,
            CPUPageProperty: cpu_page,
            MemoryPoolPreference: memory_pool_preference,
            //we don't care about multi-adapter operation, so these next two will be zero
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };
        let resource_description = d3d12::D3D12_RESOURCE_DESC {
            Dimension: d3d12::D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: buffer_size_in_bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: d3d12::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: resource_flags,
            ..mem::zeroed()
        };

        let buffer = self.create_committed_resource(
            &heap_properties,
            d3d12::D3D12_HEAP_FLAG_NONE,
            &resource_description,
            init_resource_state,
            ptr::null(),
        )?;

        Ok(buffer)
    }

    pub unsafe fn create_texture2d_buffer(
        &self,
        width: u64,
        height: u32,
        format: dxgiformat::DXGI_FORMAT,
        allow_unordered_access: bool,
    ) -> Result<Resource, Error> {
        // Images are always created device-local.
        let heap_properties = d3d12::D3D12_HEAP_PROPERTIES {
            Type: d3d12::D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: d3d12::D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: d3d12::D3D12_MEMORY_POOL_UNKNOWN,
            //we don't care about multi-adapter operation, so these next two will be zero
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let (flags, initial_resource_state) = {
            if allow_unordered_access {
                (
                    d3d12::D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    d3d12::D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )
            } else {
                (
                    d3d12::D3D12_RESOURCE_FLAG_NONE,
                    d3d12::D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )
            }
        };

        let resource_description = d3d12::D3D12_RESOURCE_DESC {
            Dimension: d3d12::D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: width,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: d3d12::D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: flags,
            Format: format,
            ..mem::zeroed()
        };

        let buffer = self.create_committed_resource(
            &heap_properties,
            //TODO: is this heap flag ok?
            d3d12::D3D12_HEAP_FLAG_NONE,
            &resource_description,
            initial_resource_state,
            ptr::null(),
        )?;

        Ok(buffer)
    }

    pub unsafe fn get_features_architecture(
        &self,
    ) -> Result<d3d12::D3D12_FEATURE_DATA_ARCHITECTURE, Error> {
        let mut features_architecture = mem::zeroed();
        explain_error(
            self.0.CheckFeatureSupport(
                d3d12::D3D12_FEATURE_ARCHITECTURE,
                &mut features_architecture as *mut _ as *mut _,
                mem::size_of::<d3d12::D3D12_FEATURE_DATA_ARCHITECTURE>() as u32,
            ),
            "error querying feature architecture",
        )?;
        Ok(features_architecture)
    }
}

impl DescriptorHeap {
    pub unsafe fn get_cpu_descriptor_handle_for_heap_start(&self) -> CpuDescriptor {
        self.0.GetCPUDescriptorHandleForHeapStart()
    }

    pub unsafe fn get_gpu_descriptor_handle_for_heap_start(&self) -> GpuDescriptor {
        self.0.GetGPUDescriptorHandleForHeapStart()
    }
}

impl RootSignature {
    pub unsafe fn serialize_description(
        desc: &d3d12::D3D12_ROOT_SIGNATURE_DESC,
        version: d3d12::D3D_ROOT_SIGNATURE_VERSION,
    ) -> Result<Blob, Error> {
        let mut blob = ptr::null_mut();
        let mut error_blob_ptr = ptr::null_mut();

        let hresult =
            d3d12::D3D12SerializeRootSignature(desc, version, &mut blob, &mut error_blob_ptr);

        #[cfg(debug_assertions)]
        {
            let error_blob = if error_blob_ptr.is_null() {
                None
            } else {
                Some(Blob(ComPtr::from_raw(error_blob_ptr)))
            };
            if let Some(error_blob) = &error_blob {
                Blob::print_to_console(error_blob);
            }
        }

        explain_error(hresult, "could not serialize root signature description")?;

        Ok(Blob(ComPtr::from_raw(blob)))
    }
}

impl ShaderByteCode {
    // `blob` may not be null.
    // TODO: this is not super elegant, maybe want to move the get
    // operations closer to where they're used.
    #[allow(unused)]
    pub unsafe fn from_blob(blob: Blob) -> ShaderByteCode {
        ShaderByteCode {
            bytecode: d3d12::D3D12_SHADER_BYTECODE {
                BytecodeLength: blob.0.GetBufferSize(),
                pShaderBytecode: blob.0.GetBufferPointer(),
            },
        }
    }

    /// Compile a shader from raw HLSL.
    ///
    /// * `target`: example format: `ps_5_1`.
    #[allow(unused)]
    pub unsafe fn compile(
        source: &str,
        target: &str,
        entry: &str,
        flags: minwindef::DWORD,
    ) -> Result<Blob, Error> {
        let mut shader_blob_ptr: *mut ID3DBlob = ptr::null_mut();
        //TODO: use error blob properly
        let mut error_blob_ptr: *mut ID3DBlob = ptr::null_mut();

        let target = ffi::CString::new(target)
            .expect("could not convert target format string into ffi::CString");
        let entry = ffi::CString::new(entry)
            .expect("could not convert entry name String into ffi::CString");

        let hresult = d3dcompiler::D3DCompile(
            source.as_ptr() as *const _,
            source.len(),
            ptr::null(),
            ptr::null(),
            d3dcompiler::D3D_COMPILE_STANDARD_FILE_INCLUDE,
            entry.as_ptr(),
            target.as_ptr(),
            flags,
            0,
            &mut shader_blob_ptr,
            &mut error_blob_ptr,
        );

        let error_blob = if error_blob_ptr.is_null() {
            None
        } else {
            Some(Blob(ComPtr::from_raw(error_blob_ptr)))
        };
        #[cfg(debug_assertions)]
        {
            if let Some(error_blob) = &error_blob {
                Blob::print_to_console(error_blob);
            }
        }

        // TODO: we can put the shader compilation error into the returned error.
        explain_error(hresult, "shader compilation failed")?;

        Ok(Blob(ComPtr::from_raw(shader_blob_ptr)))
    }

    /// Create bytecode from a slice.
    ///
    /// # Safety
    ///
    /// This call elides the lifetime from the slice. The caller is responsible
    /// for making sure the reference remains valid for the lifetime of this
    /// object.
    #[allow(unused)]
    pub unsafe fn from_slice(bytecode: &[u8]) -> ShaderByteCode {
        ShaderByteCode {
            bytecode: d3d12::D3D12_SHADER_BYTECODE {
                BytecodeLength: bytecode.len(),
                pShaderBytecode: bytecode.as_ptr() as *const _,
            },
        }
    }
}

impl Fence {
    pub unsafe fn set_event_on_completion(&self, event: &Event, value: u64) -> Result<(), Error> {
        explain_error(
            self.0.SetEventOnCompletion(value, event.0),
            "error setting event completion",
        )
    }

    pub unsafe fn get_value(&self) -> u64 {
        self.0.GetCompletedValue()
    }
}

impl Event {
    pub unsafe fn create(manual_reset: bool, initial_state: bool) -> Result<Self, Error> {
        let handle = synchapi::CreateEventA(
            ptr::null_mut(),
            manual_reset as _,
            initial_state as _,
            ptr::null(),
        );
        if handle.is_null() {
            // TODO: should probably call GetLastError here
            Err(Error::Hresult(-1))
        } else {
            Ok(Event(handle))
        }
    }

    /// Wait for the event, or a timeout.
    ///
    /// If the timeout is `winapi::um::winbase::INFINITE`, it will wait until the
    /// event is signaled.
    ///
    /// The return value is defined here:
    /// https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject
    pub unsafe fn wait(&self, timeout_ms: u32) -> u32 {
        synchapi::WaitForSingleObject(self.0, timeout_ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            handleapi::CloseHandle(self.0);
        }
    }
}

impl CommandAllocator {
    pub unsafe fn reset(&self) -> Result<(), Error> {
        error::error_if_failed_else_unit(self.0.Reset())
    }
}

impl GraphicsCommandList {
    pub unsafe fn as_raw_command_list(&self) -> *mut d3d12::ID3D12CommandList {
        self.0.as_raw() as *mut d3d12::ID3D12CommandList
    }

    pub unsafe fn close(&self) -> Result<(), Error> {
        explain_error(self.0.Close(), "error closing command list")
    }

    pub unsafe fn reset(
        &self,
        allocator: &CommandAllocator,
        initial_pso: Option<&PipelineState>,
    ) -> Result<(), Error> {
        let p_initial_state = initial_pso.map(|p| p.0.as_raw()).unwrap_or(ptr::null_mut());
        error::error_if_failed_else_unit(self.0.Reset(allocator.0.as_raw(), p_initial_state))
    }

    pub unsafe fn set_compute_pipeline_root_signature(&self, signature: &RootSignature) {
        self.0.SetComputeRootSignature(signature.0.as_raw());
    }

    pub unsafe fn resource_barrier(&self, resource_barriers: &[d3d12::D3D12_RESOURCE_BARRIER]) {
        self.0.ResourceBarrier(
            resource_barriers
                .len()
                .try_into()
                .expect("Waaaaaay too many barriers"),
            resource_barriers.as_ptr(),
        );
    }

    pub unsafe fn dispatch(&self, count_x: u32, count_y: u32, count_z: u32) {
        self.0.Dispatch(count_x, count_y, count_z);
    }

    pub unsafe fn set_pipeline_state(&self, pipeline_state: &PipelineState) {
        self.0.SetPipelineState(pipeline_state.0.as_raw());
    }

    pub unsafe fn set_compute_root_descriptor_table(
        &self,
        root_parameter_index: u32,
        base_descriptor: d3d12::D3D12_GPU_DESCRIPTOR_HANDLE,
    ) {
        self.0
            .SetComputeRootDescriptorTable(root_parameter_index, base_descriptor);
    }

    pub unsafe fn set_descriptor_heaps(&self, descriptor_heaps: &[&DescriptorHeap]) {
        let mut descriptor_heap_pointers: SmallVec<[_; 4]> =
            descriptor_heaps.iter().map(|dh| dh.0.as_raw()).collect();
        self.0.SetDescriptorHeaps(
            u32::try_from(descriptor_heap_pointers.len())
                .expect("could not safely convert descriptor_heap_pointers.len() into u32"),
            descriptor_heap_pointers.as_mut_ptr(),
        );
    }

    pub unsafe fn end_timing_query(&self, query_heap: &QueryHeap, index: u32) {
        self.0.EndQuery(
            query_heap.0.as_raw(),
            d3d12::D3D12_QUERY_TYPE_TIMESTAMP,
            index,
        );
    }

    pub unsafe fn resolve_timing_query_data(
        &self,
        query_heap: &QueryHeap,
        start_index: u32,
        num_queries: u32,
        destination_buffer: &Resource,
        aligned_destination_buffer_offset: u64,
    ) {
        self.0.ResolveQueryData(
            query_heap.0.as_raw() as *mut _,
            d3d12::D3D12_QUERY_TYPE_TIMESTAMP,
            start_index,
            num_queries,
            destination_buffer.get_mut(),
            aligned_destination_buffer_offset,
        );
    }

    pub unsafe fn clear_uav(
        &self,
        gpu_handle: d3d12::D3D12_GPU_DESCRIPTOR_HANDLE,
        cpu_handle: d3d12::D3D12_CPU_DESCRIPTOR_HANDLE,
        resource: &Resource,
        value: u32,
        size: Option<u64>,
    ) {
        // In testing, only the first value seems to be used, but just in case...
        let values = [value, value, value, value];
        let mut rect = d3d12::D3D12_RECT {
            left: 0,
            right: 0,
            top: 0,
            bottom: 1,
        };
        let (num_rects, p_rects) = if let Some(size) = size {
            rect.right = (size / 4).try_into().unwrap();
            (1, &rect as *const _)
        } else {
            (0, std::ptr::null())
        };
        self.0.ClearUnorderedAccessViewUint(
            gpu_handle,
            cpu_handle,
            resource.get_mut(),
            &values,
            num_rects,
            p_rects,
        );
    }

    /// Copy an entire resource (buffer or image)
    pub unsafe fn copy_resource(&self, src: &Resource, dst: &Resource) {
        self.0.CopyResource(dst.get_mut(), src.get_mut());
    }

    pub unsafe fn copy_buffer(
        &self,
        dst_buf: &Resource,
        dst_offset: u64,
        src_buf: &Resource,
        src_offset: u64,
        size: u64,
    ) {
        self.0.CopyBufferRegion(
            dst_buf.get_mut(),
            dst_offset,
            src_buf.get_mut(),
            src_offset,
            size,
        );
    }

    pub unsafe fn copy_buffer_to_texture(
        &self,
        buffer: &Resource,
        texture: &Resource,
        width: u32,
        height: u32,
    ) {
        let mut src = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: buffer.get_mut(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            ..mem::zeroed()
        };
        let row_pitch = width * 4;
        assert!(
            row_pitch % d3d12::D3D12_TEXTURE_DATA_PITCH_ALIGNMENT == 0,
            "TODO: handle unaligned row pitch"
        );
        let footprint = d3d12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: d3d12::D3D12_SUBRESOURCE_FOOTPRINT {
                Format: dxgiformat::DXGI_FORMAT_R8G8B8A8_UNORM,
                Width: width,
                Height: height,
                Depth: 1,
                RowPitch: row_pitch,
            },
        };
        *src.u.PlacedFootprint_mut() = footprint;

        let mut dst = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: texture.get_mut(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            ..mem::zeroed()
        };
        *dst.u.SubresourceIndex_mut() = 0;

        self.0.CopyTextureRegion(&dst, 0, 0, 0, &src, ptr::null());
    }

    pub unsafe fn copy_texture_to_buffer(
        &self,
        texture: &Resource,
        buffer: &Resource,
        width: u32,
        height: u32,
    ) {
        let mut src = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: texture.get_mut(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            ..mem::zeroed()
        };
        *src.u.SubresourceIndex_mut() = 0;

        let mut dst = d3d12::D3D12_TEXTURE_COPY_LOCATION {
            pResource: buffer.get_mut(),
            Type: d3d12::D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            ..mem::zeroed()
        };
        let row_pitch = width * 4;
        assert!(
            row_pitch % d3d12::D3D12_TEXTURE_DATA_PITCH_ALIGNMENT == 0,
            "TODO: handle unaligned row pitch"
        );
        let footprint = d3d12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: d3d12::D3D12_SUBRESOURCE_FOOTPRINT {
                Format: dxgiformat::DXGI_FORMAT_R8G8B8A8_UNORM,
                Width: width,
                Height: height,
                Depth: 1,
                RowPitch: row_pitch,
            },
        };
        *dst.u.PlacedFootprint_mut() = footprint;

        self.0.CopyTextureRegion(&dst, 0, 0, 0, &src, ptr::null());
    }
}

pub unsafe fn create_uav_resource_barrier(
    resource: *mut d3d12::ID3D12Resource,
) -> d3d12::D3D12_RESOURCE_BARRIER {
    let uav = d3d12::D3D12_RESOURCE_UAV_BARRIER {
        pResource: resource,
    };

    let mut resource_barrier: d3d12::D3D12_RESOURCE_BARRIER = mem::zeroed();
    resource_barrier.Type = d3d12::D3D12_RESOURCE_BARRIER_TYPE_UAV;
    resource_barrier.Flags = d3d12::D3D12_RESOURCE_BARRIER_FLAG_NONE;
    *resource_barrier.u.UAV_mut() = uav;

    resource_barrier
}

pub unsafe fn create_transition_resource_barrier(
    resource: *mut d3d12::ID3D12Resource,
    state_before: d3d12::D3D12_RESOURCE_STATES,
    state_after: d3d12::D3D12_RESOURCE_STATES,
) -> d3d12::D3D12_RESOURCE_BARRIER {
    let transition = d3d12::D3D12_RESOURCE_TRANSITION_BARRIER {
        pResource: resource,
        Subresource: d3d12::D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        StateBefore: state_before,
        StateAfter: state_after,
    };

    let mut resource_barrier: d3d12::D3D12_RESOURCE_BARRIER = mem::zeroed();
    resource_barrier.Type = d3d12::D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    resource_barrier.Flags = d3d12::D3D12_RESOURCE_BARRIER_FLAG_NONE;
    *resource_barrier.u.Transition_mut() = transition;

    resource_barrier
}

#[allow(unused)]
pub unsafe fn enable_debug_layer() -> Result<(), Error> {
    let mut debug_controller: *mut d3d12sdklayers::ID3D12Debug1 = ptr::null_mut();
    explain_error(
        d3d12::D3D12GetDebugInterface(
            &d3d12sdklayers::ID3D12Debug1::uuidof(),
            &mut debug_controller as *mut _ as *mut _,
        ),
        "could not create debug controller",
    )?;

    let debug_controller = ComPtr::from_raw(debug_controller);
    debug_controller.EnableDebugLayer();

    let mut queue = ptr::null_mut();
    let hr = dxgi1_3::DXGIGetDebugInterface1(
        0,
        &dxgidebug::IDXGIInfoQueue::uuidof(),
        &mut queue as *mut _ as *mut _,
    );

    explain_error(hr, "failed to enable debug layer")?;

    debug_controller.SetEnableGPUBasedValidation(minwindef::TRUE);
    Ok(())
}
