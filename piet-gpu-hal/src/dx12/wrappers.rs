// Copyright © 2019 piet-gpu developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::dx12::error::{self, error_if_failed_else_unit, explain_error, Error};
use std::convert::{TryFrom, TryInto};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::{ffi, mem, path::Path, ptr};
use winapi::shared::{
    dxgi, dxgi1_2, dxgi1_3, dxgi1_4, dxgiformat, dxgitype, minwindef, windef, winerror,
};
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

pub struct VertexBufferView(pub ComPtr<d3d12::D3D12_VERTEX_BUFFER_VIEW>);

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
pub struct DescriptorHeap {
    pub heap_type: d3d12::D3D12_DESCRIPTOR_HEAP_TYPE,
    pub increment_size: u32,
    pub heap: ComPtr<d3d12::ID3D12DescriptorHeap>,
}

pub type TextureAddressMode = [d3d12::D3D12_TEXTURE_ADDRESS_MODE; 3];

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
    blob: Option<Blob>,
}

pub struct DebugController(pub d3d12sdklayers::ID3D12Debug);

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

    pub unsafe fn write_resource(
        &self,
        data: *const u8,
        offset: usize,
        size: usize,
    ) -> Result<(), Error> {
        let mut mapped_memory: *mut u8 = ptr::null_mut();
        let zero_range = d3d12::D3D12_RANGE { ..mem::zeroed() };
        let range = d3d12::D3D12_RANGE {
            Begin: offset,
            End: offset + size,
        };
        explain_error(
            (*self.get()).Map(0, &zero_range, &mut mapped_memory as *mut _ as *mut _),
            "could not map GPU mem to CPU mem",
        )?;

        ptr::copy_nonoverlapping(data, mapped_memory.add(offset), size);
        (*self.get()).Unmap(0, &range);
        Ok(())
    }

    pub unsafe fn read_resource(
        &self,
        dst: *mut u8,
        offset: usize,
        size: usize,
    ) -> Result<(), Error> {
        let mut mapped_memory: *mut u8 = ptr::null_mut();
        let range = d3d12::D3D12_RANGE {
            Begin: offset,
            End: offset + size,
        };
        let zero_range = d3d12::D3D12_RANGE { ..mem::zeroed() };
        explain_error(
            (*self.get()).Map(0, &range, &mut mapped_memory as *mut _ as *mut _),
            "could not map GPU mem to CPU mem",
        )?;
        ptr::copy_nonoverlapping(mapped_memory.add(offset), dst, size);
        (*self.get()).Unmap(0, &zero_range);
        Ok(())
    }

    pub unsafe fn get_gpu_virtual_address(&self) -> d3d12::D3D12_GPU_VIRTUAL_ADDRESS {
        (*self.get()).GetGPUVirtualAddress()
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
        println!("desc: {:?}", desc.Description);
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

        Ok(DescriptorHeap {
            heap_type: heap_description.Type,
            increment_size: self.get_descriptor_increment_size(heap_description.Type),
            heap: ComPtr::from_raw(heap),
        })
    }

    pub unsafe fn get_descriptor_increment_size(
        &self,
        heap_type: d3d12::D3D12_DESCRIPTOR_HEAP_TYPE,
    ) -> u32 {
        self.0.GetDescriptorHandleIncrementSize(heap_type)
    }

    pub unsafe fn create_graphics_pipeline_state(
        &self,
        graphics_pipeline_desc: &d3d12::D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    ) -> PipelineState {
        let mut pipeline_state = ptr::null_mut();

        error::error_if_failed_else_unit(self.0.CreateGraphicsPipelineState(
            graphics_pipeline_desc as *const _,
            &d3d12::ID3D12PipelineState::uuidof(),
            &mut pipeline_state as *mut _ as *mut _,
        ))
        .expect("device could not create graphics pipeline state");

        PipelineState(ComPtr::from_raw(pipeline_state))
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

    // This is for indirect command submission and we probably won't use it.
    pub unsafe fn create_command_signature(
        &self,
        root_signature: RootSignature,
        arguments: &[d3d12::D3D12_INDIRECT_ARGUMENT_DESC],
        stride: u32,
        node_mask: minwindef::UINT,
    ) -> CommandSignature {
        let mut signature = ptr::null_mut();
        let desc = d3d12::D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: stride,
            NumArgumentDescs: arguments.len() as _,
            pArgumentDescs: arguments.as_ptr() as *const _,
            NodeMask: node_mask,
        };

        error::error_if_failed_else_unit(self.0.CreateCommandSignature(
            &desc,
            root_signature.0.as_raw(),
            &d3d12::ID3D12CommandSignature::uuidof(),
            &mut signature as *mut _ as *mut _,
        ))
        .expect("device could not create command signature");

        CommandSignature(ComPtr::from_raw(signature))
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

    pub unsafe fn create_constant_buffer_view(
        &self,
        resource: &Resource,
        descriptor: CpuDescriptor,
        size_in_bytes: u32,
    ) {
        let cbv_desc = d3d12::D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: resource.get_gpu_virtual_address(),
            SizeInBytes: size_in_bytes,
        };
        self.0
            .CreateConstantBufferView(&cbv_desc as *const _, descriptor);
    }

    pub unsafe fn create_byte_addressed_buffer_shader_resource_view(
        &self,
        resource: &Resource,
        descriptor: CpuDescriptor,
        first_element: u64,
        num_elements: u32,
    ) {
        let mut srv_desc = d3d12::D3D12_SHADER_RESOURCE_VIEW_DESC {
            // shouldn't flags be dxgiformat::DXGI_FORMAT_R32_TYPELESS?
            Format: dxgiformat::DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: d3d12::D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: 0x1688,
            ..mem::zeroed()
        };
        *srv_desc.u.Buffer_mut() = d3d12::D3D12_BUFFER_SRV {
            FirstElement: first_element,
            NumElements: num_elements,
            // shouldn't StructureByteStride be 0?
            StructureByteStride: 0,
            // shouldn't flags be d3d12::D3D12_BUFFER_SRV_FLAG_RAW?
            Flags: d3d12::D3D12_BUFFER_SRV_FLAG_RAW,
        };
        self.0
            .CreateShaderResourceView(resource.get_mut(), &srv_desc as *const _, descriptor);
    }

    pub unsafe fn create_structured_buffer_shader_resource_view(
        &self,
        resource: &Resource,
        descriptor: CpuDescriptor,
        first_element: u64,
        num_elements: u32,
        structure_byte_stride: u32,
    ) {
        let mut srv_desc = d3d12::D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: dxgiformat::DXGI_FORMAT_UNKNOWN,
            ViewDimension: d3d12::D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: 0x1688,
            ..mem::zeroed()
        };
        *srv_desc.u.Buffer_mut() = d3d12::D3D12_BUFFER_SRV {
            FirstElement: first_element,
            NumElements: num_elements,
            StructureByteStride: structure_byte_stride,
            Flags: d3d12::D3D12_BUFFER_SRV_FLAG_NONE,
        };
        self.0
            .CreateShaderResourceView(resource.get_mut(), &srv_desc as *const _, descriptor);
    }

    pub unsafe fn create_texture2d_shader_resource_view(
        &self,
        resource: &Resource,
        format: dxgiformat::DXGI_FORMAT,
        descriptor: CpuDescriptor,
    ) {
        let mut srv_desc = d3d12::D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: d3d12::D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: 0x1688,
            ..mem::zeroed()
        };
        *srv_desc.u.Texture2D_mut() = d3d12::D3D12_TEX2D_SRV {
            MostDetailedMip: 0,
            MipLevels: 1,
            PlaneSlice: 0,
            ResourceMinLODClamp: 0.0,
        };
        self.0
            .CreateShaderResourceView(resource.get_mut(), &srv_desc as *const _, descriptor);
    }

    pub unsafe fn create_render_target_view(
        &self,
        resource: &Resource,
        desc: *const d3d12::D3D12_RENDER_TARGET_VIEW_DESC,
        descriptor: CpuDescriptor,
    ) {
        self.0
            .CreateRenderTargetView(resource.get_mut(), desc, descriptor);
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

    pub unsafe fn destroy_fence(&self, fence: &Fence) -> Result<(), Error> {
        Ok(())
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

    // based on: https://github.com/microsoft/DirectX-Graphics-Samples/blob/682051ddbe4be820195fffed0bfbdbbde8611a90/Libraries/D3DX12/d3dx12.h#L1875
    pub unsafe fn get_required_intermediate_buffer_size(
        &self,
        dest_resource: Resource,
        first_subresource: u32,
        num_subresources: u32,
    ) -> u64 {
        let desc: d3d12::D3D12_RESOURCE_DESC = (*dest_resource.get()).GetDesc();

        let mut required_size: *mut u64 = ptr::null_mut();
        self.0.GetCopyableFootprints(
            &desc as *const _,
            first_subresource,
            num_subresources,
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut required_size as *mut _ as *mut _,
        );

        *required_size
    }

    pub unsafe fn get_copyable_footprint(
        &self,
        first_subresource: u32,
        num_subresources: usize,
        base_offset: u64,
        dest_resource: &Resource,
    ) -> (
        Vec<d3d12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
        Vec<u32>,
        Vec<u64>,
        u64,
    ) {
        let desc: d3d12::D3D12_RESOURCE_DESC = (*dest_resource.get()).GetDesc();

        let mut layouts: Vec<d3d12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT> =
            Vec::with_capacity(num_subresources);

        let mut num_rows: Vec<u32> = Vec::with_capacity(num_subresources);

        let mut row_size_in_bytes: Vec<u64> = Vec::with_capacity(num_subresources);

        let mut total_size: u64 = 0;

        self.0.GetCopyableFootprints(
            &desc as *const _,
            first_subresource,
            u32::try_from(num_subresources)
                .expect("could not safely convert num_subresources into u32"),
            base_offset,
            layouts.as_mut_ptr(),
            num_rows.as_mut_ptr(),
            row_size_in_bytes.as_mut_ptr(),
            &mut total_size as *mut _,
        );

        layouts.set_len(num_subresources);
        num_rows.set_len(num_subresources);
        row_size_in_bytes.set_len(num_subresources);

        (layouts, num_rows, row_size_in_bytes, total_size)
    }

    pub unsafe fn create_buffer(
        &self,
        buffer_size_in_bytes: u32,
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
            Width: buffer_size_in_bytes as u64,
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

    pub unsafe fn get_removal_reason(&self) -> Error {
        Error::Hresult(self.0.GetDeviceRemovedReason())
    }
}

pub struct SubresourceData {
    pub data: Vec<u8>,
    pub row_size: isize,
    pub column_size: isize,
}

impl SubresourceData {
    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn as_d3d12_subresource_data(&self) -> d3d12::D3D12_SUBRESOURCE_DATA {
        assert_eq!(self.row_size % 256, 0);

        d3d12::D3D12_SUBRESOURCE_DATA {
            pData: self.data.as_ptr() as *const _,
            RowPitch: self.row_size,
            SlicePitch: self.column_size,
        }
    }
}

impl CommandAllocator {
    pub unsafe fn reset(&self) -> Result<(), Error> {
        explain_error(self.0.Reset(), "error resetting command allocator")
    }
}

impl DescriptorHeap {
    unsafe fn get_cpu_descriptor_handle_for_heap_start(&self) -> CpuDescriptor {
        self.heap.GetCPUDescriptorHandleForHeapStart()
    }

    unsafe fn get_gpu_descriptor_handle_for_heap_start(&self) -> GpuDescriptor {
        self.heap.GetGPUDescriptorHandleForHeapStart()
    }

    pub unsafe fn get_cpu_descriptor_handle_at_offset(&self, offset: u32) -> CpuDescriptor {
        let mut descriptor = self.get_cpu_descriptor_handle_for_heap_start();
        descriptor.ptr += (offset as usize) * (self.increment_size as usize);

        descriptor
    }

    pub unsafe fn get_gpu_descriptor_handle_at_offset(&self, offset: u32) -> GpuDescriptor {
        let mut descriptor = self.get_gpu_descriptor_handle_for_heap_start();
        descriptor.ptr += (offset as u64) * (self.increment_size as u64);

        descriptor
    }
}

#[repr(transparent)]
pub struct DescriptorRange(d3d12::D3D12_DESCRIPTOR_RANGE);
impl DescriptorRange {}

impl RootSignature {
    pub unsafe fn serialize_description(
        desc: &d3d12::D3D12_ROOT_SIGNATURE_DESC,
        version: d3d12::D3D_ROOT_SIGNATURE_VERSION,
    ) -> Result<Blob, Error> {
        let mut blob = ptr::null_mut();
        let mut error_blob_ptr = ptr::null_mut();

        let hresult =
            d3d12::D3D12SerializeRootSignature(desc, version, &mut blob, &mut error_blob_ptr);

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

        explain_error(hresult, "could not serialize root signature description")?;

        Ok(Blob(ComPtr::from_raw(blob)))
    }
}

impl ShaderByteCode {
    // empty byte code
    pub unsafe fn empty() -> ShaderByteCode {
        ShaderByteCode {
            bytecode: d3d12::D3D12_SHADER_BYTECODE {
                BytecodeLength: 0,
                pShaderBytecode: ptr::null(),
            },
            blob: None,
        }
    }

    // `blob` may not be null.
    // TODO: this is not super elegant, maybe want to move the get
    // operations closer to where they're used.
    pub unsafe fn from_blob(blob: Blob) -> ShaderByteCode {
        ShaderByteCode {
            bytecode: d3d12::D3D12_SHADER_BYTECODE {
                BytecodeLength: blob.0.GetBufferSize(),
                pShaderBytecode: blob.0.GetBufferPointer(),
            },
            blob: Some(blob),
        }
    }

    /// Compile a shader from raw HLSL.
    ///
    /// * `target`: example format: `ps_5_1`.
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

    pub unsafe fn compile_from_file(
        file_path: &Path,
        target: &str,
        entry: &str,
        flags: minwindef::DWORD,
    ) -> Result<Blob, Error> {
        let file_open_error = format!("could not open shader source file for entry: {}", &entry);
        let source = std::fs::read_to_string(file_path).expect(&file_open_error);

        ShaderByteCode::compile(&source, target, entry, flags)
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

    pub unsafe fn signal(&self, value: u64) -> winerror::HRESULT {
        self.0.Signal(value)
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

    // TODO: probably remove, yagni
    pub unsafe fn wait_ex(&self, timeout_ms: u32, alertable: bool) -> u32 {
        synchapi::WaitForSingleObjectEx(self.0, timeout_ms, alertable as _)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            handleapi::CloseHandle(self.0);
        }
    }
}

impl GraphicsCommandList {
    pub unsafe fn as_raw_command_list(&self) -> *mut d3d12::ID3D12CommandList {
        self.0.as_raw() as *mut d3d12::ID3D12CommandList
    }

    pub unsafe fn close(&self) -> Result<(), Error> {
        explain_error(self.0.Close(), "error closing command list")
    }

    pub unsafe fn reset(&self, allocator: &CommandAllocator, initial_pso: Option<&PipelineState>) {
        let p_initial_state = initial_pso.map(|p| p.0.as_raw()).unwrap_or(ptr::null_mut());
        error::error_if_failed_else_unit(self.0.Reset(allocator.0.as_raw(), p_initial_state))
            .expect("could not reset command list");
    }

    pub unsafe fn set_compute_pipeline_root_signature(&self, signature: &RootSignature) {
        self.0.SetComputeRootSignature(signature.0.as_raw());
    }

    pub unsafe fn set_graphics_pipeline_root_signature(&self, signature: &RootSignature) {
        self.0.SetGraphicsRootSignature(signature.0.as_raw());
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

    pub unsafe fn set_viewport(&self, viewport: &d3d12::D3D12_VIEWPORT) {
        self.0.RSSetViewports(1, viewport as *const _);
    }

    pub unsafe fn set_scissor_rect(&self, scissor_rect: &d3d12::D3D12_RECT) {
        self.0.RSSetScissorRects(1, scissor_rect as *const _);
    }

    pub unsafe fn dispatch(&self, count_x: u32, count_y: u32, count_z: u32) {
        self.0.Dispatch(count_x, count_y, count_z);
    }

    pub unsafe fn draw_instanced(
        &self,
        num_vertices: u32,
        num_instances: u32,
        start_vertex: u32,
        start_instance: u32,
    ) {
        self.0
            .DrawInstanced(num_vertices, num_instances, start_vertex, start_instance);
    }

    pub unsafe fn set_pipeline_state(&self, pipeline_state: &PipelineState) {
        self.0.SetPipelineState(pipeline_state.0.as_raw());
    }

    pub unsafe fn set_compute_root_unordered_access_view(
        &self,
        root_parameter_index: u32,
        buffer_location: d3d12::D3D12_GPU_VIRTUAL_ADDRESS,
    ) {
        self.0
            .SetComputeRootUnorderedAccessView(root_parameter_index, buffer_location);
    }

    pub unsafe fn set_compute_root_descriptor_table(
        &self,
        root_parameter_index: u32,
        base_descriptor: d3d12::D3D12_GPU_DESCRIPTOR_HANDLE,
    ) {
        self.0
            .SetComputeRootDescriptorTable(root_parameter_index, base_descriptor);
    }

    pub unsafe fn set_graphics_root_shader_resource_view(
        &self,
        root_parameter_index: u32,
        buffer_location: d3d12::D3D12_GPU_VIRTUAL_ADDRESS,
    ) {
        self.0
            .SetGraphicsRootShaderResourceView(root_parameter_index, buffer_location);
    }

    pub unsafe fn set_graphics_root_descriptor_table(
        &self,
        root_parameter_index: u32,
        base_descriptor: d3d12::D3D12_GPU_DESCRIPTOR_HANDLE,
    ) {
        self.0
            .SetGraphicsRootDescriptorTable(root_parameter_index, base_descriptor);
    }

    pub unsafe fn set_render_target(
        &self,
        render_target_descriptor: d3d12::D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        self.0.OMSetRenderTargets(
            1,
            &render_target_descriptor as *const _,
            false as _,
            ptr::null(),
        );
    }

    pub unsafe fn clear_render_target_view(
        &self,
        render_target_descriptor: d3d12::D3D12_CPU_DESCRIPTOR_HANDLE,
        clear_color: &[f32; 4],
    ) {
        self.0.ClearRenderTargetView(
            render_target_descriptor,
            clear_color as *const _,
            0,
            ptr::null(),
        );
    }

    pub unsafe fn set_primitive_topology(
        &self,
        primitive_topology: d3dcommon::D3D_PRIMITIVE_TOPOLOGY,
    ) {
        self.0.IASetPrimitiveTopology(primitive_topology);
    }

    pub unsafe fn set_vertex_buffer(
        &self,
        start_slot: u32,
        num_views: u32,
        vertex_buffer_view: &d3d12::D3D12_VERTEX_BUFFER_VIEW,
    ) {
        self.0
            .IASetVertexBuffers(start_slot, num_views, vertex_buffer_view as *const _);
    }

    pub unsafe fn set_descriptor_heaps(&self, descriptor_heaps: &[&DescriptorHeap]) {
        let mut descriptor_heap_pointers: Vec<_> =
            descriptor_heaps.iter().map(|dh| dh.heap.as_raw()).collect();
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

pub fn default_render_target_blend_desc() -> d3d12::D3D12_RENDER_TARGET_BLEND_DESC {
    d3d12::D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: minwindef::FALSE,
        LogicOpEnable: minwindef::FALSE,
        SrcBlend: d3d12::D3D12_BLEND_ONE,
        DestBlend: d3d12::D3D12_BLEND_ZERO,
        // enum variant 0
        BlendOp: d3d12::D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: d3d12::D3D12_BLEND_ONE,
        DestBlendAlpha: d3d12::D3D12_BLEND_ZERO,
        BlendOpAlpha: d3d12::D3D12_BLEND_OP_ADD,
        // enum variant 0
        LogicOp: d3d12::D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: d3d12::D3D12_COLOR_WRITE_ENABLE_ALL as u8,
    }
}

pub fn default_blend_desc() -> d3d12::D3D12_BLEND_DESC {
    // see default description here: https://docs.microsoft.com/en-us/windows/win32/direct3d12/cd3dx12-blend-desc
    d3d12::D3D12_BLEND_DESC {
        AlphaToCoverageEnable: minwindef::FALSE,
        IndependentBlendEnable: minwindef::FALSE,
        RenderTarget: [
            default_render_target_blend_desc(),
            default_render_target_blend_desc(),
            default_render_target_blend_desc(),
            default_render_target_blend_desc(),
            default_render_target_blend_desc(),
            default_render_target_blend_desc(),
            default_render_target_blend_desc(),
            default_render_target_blend_desc(),
        ],
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

pub unsafe fn enable_debug_layer() -> Result<(), Error> {
    println!("enabling debug layer.");

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

pub struct InputElementDesc {
    pub semantic_name: String,
    pub semantic_index: u32,
    pub format: dxgiformat::DXGI_FORMAT,
    pub input_slot: u32,
    pub aligned_byte_offset: u32,
    pub input_slot_class: d3d12::D3D12_INPUT_CLASSIFICATION,
    pub instance_data_step_rate: u32,
}

impl InputElementDesc {
    pub fn as_winapi_struct(&self) -> d3d12::D3D12_INPUT_ELEMENT_DESC {
        d3d12::D3D12_INPUT_ELEMENT_DESC {
            SemanticName: std::ffi::CString::new(self.semantic_name.as_str())
                .unwrap()
                .into_raw() as *const _,
            SemanticIndex: self.semantic_index,
            Format: self.format,
            InputSlot: self.input_slot,
            AlignedByteOffset: self.aligned_byte_offset,
            InputSlotClass: self.input_slot_class,
            InstanceDataStepRate: self.instance_data_step_rate,
        }
    }
}
