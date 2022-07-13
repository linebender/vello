// Copyright © 2021 piet-gpu developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those

//! Descriptor management.

use std::{
    convert::TryInto,
    ops::Deref,
    sync::{Arc, Mutex, Weak},
};

use smallvec::SmallVec;
use winapi::um::d3d12::{
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
    D3D12_GPU_DESCRIPTOR_HANDLE,
};

use crate::{bestfit::BestFit, Error};

use super::wrappers::{DescriptorHeap, Device};

const CPU_CHUNK_SIZE: u32 = 256;
const GPU_CHUNK_SIZE: u32 = 4096;

#[derive(Default)]
pub struct DescriptorPool {
    cpu_visible: Vec<CpuHeap>,
    gpu_visible: Vec<GpuHeap>,
    free_list: Arc<Mutex<DescriptorFreeList>>,
}

#[derive(Default)]
pub struct DescriptorFreeList {
    cpu_free: Vec<Vec<u32>>,
    gpu_free: Vec<BestFit>,
}

struct CpuHeap {
    // Retained for lifetime reasons.
    #[allow(unused)]
    dx12_heap: DescriptorHeap,
    cpu_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    increment_size: u32,
}

pub struct CpuHeapRef {
    heap_ix: usize,
    offset: u32,
}

/// An owned reference to the CPU heap.
///
/// When dropped, the corresponding heap range will be freed.
pub struct CpuHeapRefOwned {
    heap_ref: CpuHeapRef,
    handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    free_list: Weak<Mutex<DescriptorFreeList>>,
}

/// A shader-visible descriptor heap.
struct GpuHeap {
    dx12_heap: DescriptorHeap,
    cpu_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    gpu_handle: D3D12_GPU_DESCRIPTOR_HANDLE,
    increment_size: u32,
}

pub struct GpuHeapRef {
    heap_ix: usize,
    offset: u32,
    n: u32,
}

/// An owned reference to the GPU heap.
///
/// When dropped, the corresponding heap range will be freed.
pub struct GpuHeapRefOwned {
    heap_ref: GpuHeapRef,
    cpu_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    gpu_handle: D3D12_GPU_DESCRIPTOR_HANDLE,
    increment_size: u32,
    free_list: Weak<Mutex<DescriptorFreeList>>,
}

impl DescriptorPool {
    pub fn alloc_cpu(&mut self, device: &Device) -> Result<CpuHeapRefOwned, Error> {
        let free_list = &self.free_list;
        let mk_owned = |heap_ref, handle| CpuHeapRefOwned {
            heap_ref,
            handle,
            free_list: Arc::downgrade(free_list),
        };
        let mut free_list = free_list.lock().unwrap();
        for (heap_ix, free) in free_list.cpu_free.iter_mut().enumerate() {
            if let Some(offset) = free.pop() {
                let handle = self.cpu_visible[heap_ix].cpu_handle(offset);
                return Ok(mk_owned(CpuHeapRef { heap_ix, offset }, handle));
            }
        }
        unsafe {
            let heap_type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: heap_type,
                NumDescriptors: CPU_CHUNK_SIZE,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            };
            let dx12_heap = device.create_descriptor_heap(&desc)?;
            let mut free = (0..CPU_CHUNK_SIZE).rev().collect::<Vec<_>>();
            let offset = free.pop().unwrap();
            debug_assert_eq!(offset, 0);
            let heap_ref = CpuHeapRef {
                heap_ix: self.cpu_visible.len(),
                offset,
            };
            let cpu_handle = dx12_heap.get_cpu_descriptor_handle_for_heap_start();
            let increment_size = device.get_descriptor_increment_size(heap_type);
            let heap = CpuHeap {
                dx12_heap,
                cpu_handle,
                increment_size,
            };
            self.cpu_visible.push(heap);
            free_list.cpu_free.push(free);
            Ok(mk_owned(heap_ref, cpu_handle))
        }
    }

    pub fn cpu_handle(&self, cpu_ref: &CpuHeapRef) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.cpu_visible[cpu_ref.heap_ix].cpu_handle(cpu_ref.offset)
    }

    pub fn alloc_gpu(&mut self, device: &Device, n: u32) -> Result<GpuHeapRefOwned, Error> {
        let free_list = &self.free_list;
        let heap_type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let increment_size = unsafe { device.get_descriptor_increment_size(heap_type) };
        let mk_owned = |heap_ref, cpu_handle, gpu_handle| GpuHeapRefOwned {
            heap_ref,
            cpu_handle,
            gpu_handle,
            increment_size,
            free_list: Arc::downgrade(free_list),
        };
        let mut free_list = free_list.lock().unwrap();
        for (heap_ix, free) in free_list.gpu_free.iter_mut().enumerate() {
            if let Some(offset) = free.alloc(n) {
                let heap = &self.gpu_visible[heap_ix];
                let cpu_handle = heap.cpu_handle(offset);
                let gpu_handle = heap.gpu_handle(offset);
                return Ok(mk_owned(
                    GpuHeapRef { heap_ix, offset, n },
                    cpu_handle,
                    gpu_handle,
                ));
            }
        }
        unsafe {
            let size = n.max(GPU_CHUNK_SIZE).next_power_of_two();
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: heap_type,
                NumDescriptors: size,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            let dx12_heap = device.create_descriptor_heap(&desc)?;
            let heap_ix = self.gpu_visible.len();
            let mut free = BestFit::new(size);
            let offset = free.alloc(n).unwrap();
            // We assume the first allocation is at 0, to avoid recomputing offsets.
            debug_assert_eq!(offset, 0);
            let cpu_handle = dx12_heap.get_cpu_descriptor_handle_for_heap_start();
            let gpu_handle = dx12_heap.get_gpu_descriptor_handle_for_heap_start();
            let increment_size = device.get_descriptor_increment_size(heap_type);
            let heap = GpuHeap {
                dx12_heap,
                cpu_handle,
                gpu_handle,
                increment_size,
            };
            self.gpu_visible.push(heap);
            free_list.gpu_free.push(free);
            Ok(mk_owned(
                GpuHeapRef { heap_ix, offset, n },
                cpu_handle,
                gpu_handle,
            ))
        }
    }

    pub fn cpu_handle_of_gpu(
        &self,
        gpu_ref: &GpuHeapRef,
        offset: u32,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        debug_assert!(offset < gpu_ref.n);
        let dx12_heap = &self.gpu_visible[gpu_ref.heap_ix];
        dx12_heap.cpu_handle(gpu_ref.offset + offset)
    }

    pub fn gpu_heap(&self, gpu_ref: &GpuHeapRef) -> &DescriptorHeap {
        &self.gpu_visible[gpu_ref.heap_ix].dx12_heap
    }
}

impl DescriptorFreeList {
    fn free_cpu(&mut self, cpu_ref: &CpuHeapRef) {
        self.cpu_free[cpu_ref.heap_ix].push(cpu_ref.offset);
    }

    fn free_gpu(&mut self, gpu_ref: &GpuHeapRef) {
        self.gpu_free[gpu_ref.heap_ix].free(gpu_ref.offset, gpu_ref.n);
    }
}

impl Drop for CpuHeapRefOwned {
    fn drop(&mut self) {
        if let Some(a) = self.free_list.upgrade() {
            a.lock().unwrap().free_cpu(&self.heap_ref)
        }
    }
}

impl CpuHeapRefOwned {
    pub fn handle(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.handle
    }
}

impl GpuHeapRefOwned {
    pub fn gpu_handle(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        self.gpu_handle
    }

    pub unsafe fn copy_descriptors(&self, device: &Device, src: &[D3D12_CPU_DESCRIPTOR_HANDLE]) {
        // TODO: optimize a bit (use simple variant where appropriate)
        let n = src.len().try_into().unwrap();
        let sizes = (0..n).map(|_| 1).collect::<SmallVec<[u32; 16]>>();
        device.copy_descriptors(
            &[self.cpu_handle],
            &[n],
            src,
            &sizes,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        );
    }

    pub unsafe fn copy_one_descriptor(
        &self,
        device: &Device,
        src: D3D12_CPU_DESCRIPTOR_HANDLE,
        index: u32,
    ) {
        let mut dst = self.cpu_handle;
        dst.ptr += (index * self.increment_size) as usize;
        device.copy_one_descriptor(dst, src, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
}

impl Deref for CpuHeapRefOwned {
    type Target = CpuHeapRef;

    fn deref(&self) -> &Self::Target {
        &self.heap_ref
    }
}

impl Drop for GpuHeapRefOwned {
    fn drop(&mut self) {
        if let Some(a) = self.free_list.upgrade() {
            a.lock().unwrap().free_gpu(&self.heap_ref)
        }
    }
}

impl Deref for GpuHeapRefOwned {
    type Target = GpuHeapRef;

    fn deref(&self) -> &Self::Target {
        &self.heap_ref
    }
}

impl CpuHeap {
    fn cpu_handle(&self, offset: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let mut handle = self.cpu_handle;
        handle.ptr += (offset as usize) * (self.increment_size as usize);
        handle
    }
}

impl GpuHeap {
    fn cpu_handle(&self, offset: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let mut handle = self.cpu_handle;
        handle.ptr += (offset as usize) * (self.increment_size as usize);
        handle
    }

    fn gpu_handle(&self, offset: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let mut handle = self.gpu_handle;
        handle.ptr += (offset as u64) * (self.increment_size as u64);
        handle
    }
}
