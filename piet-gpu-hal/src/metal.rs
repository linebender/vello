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

mod util;

use std::mem;
use std::sync::{Arc, Mutex};

use cocoa_foundation::base::id;
use cocoa_foundation::foundation::{NSInteger, NSUInteger};
use objc::rc::autoreleasepool;
use objc::runtime::{Object, BOOL, YES};
use objc::{class, msg_send, sel, sel_impl};

use metal::{CGFloat, MTLFeatureSet};

use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};

use crate::{BufferUsage, Error, GpuInfo, WorkgroupLimits};

use util::*;

pub struct MtlInstance;

pub struct MtlDevice {
    device: metal::Device,
    cmd_queue: Arc<Mutex<metal::CommandQueue>>,
    gpu_info: GpuInfo,
}

pub struct MtlSurface {
    layer: metal::MetalLayer,
}

pub struct MtlSwapchain {
    layer: metal::MetalLayer,
    cmd_queue: Arc<Mutex<metal::CommandQueue>>,
    drawable: Mutex<Option<metal::MetalDrawable>>,
    n_drawables: usize,
    drawable_ix: usize,
}

#[derive(Clone)]
pub struct Buffer {
    buffer: metal::Buffer,
    pub(crate) size: u64,
}

#[derive(Clone)]
pub struct Image {
    texture: metal::Texture,
    width: u32,
    height: u32,
}

// This is the way gfx-hal does it, but a more Vulkan-like strategy would be
// to have a semaphore that gets signaled from the command buffer's completion
// handler.
pub enum Fence {
    Idle,
    CmdBufPending(metal::CommandBuffer),
}

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
    images: Vec<Image>,
}

impl MtlInstance {
    pub fn new(
        window_handle: Option<&dyn HasRawWindowHandle>,
    ) -> Result<(MtlInstance, Option<MtlSurface>), Error> {
        let mut surface = None;
        if let Some(window_handle) = window_handle {
            let window_handle = window_handle.raw_window_handle();
            if let RawWindowHandle::MacOS(w) = window_handle {
                unsafe {
                    surface = Self::make_surface(w.ns_view as id, w.ns_window as id);
                }
            }
        }

        Ok((MtlInstance, surface))
    }

    unsafe fn make_surface(ns_view: id, ns_window: id) -> Option<MtlSurface> {
        let ca_ml_class = class!(CAMetalLayer);
        let is_ca_ml: BOOL = msg_send![ns_view, isKindOfClass: ca_ml_class];
        if is_ca_ml == YES {
            todo!("create surface from layer")
        }
        let layer: id = msg_send![ns_view, layer];
        let use_current = !layer.is_null() && {
            let result: BOOL = msg_send![layer, isKindOfClass: ca_ml_class];
            result == YES
        };
        let metal_layer = if use_current {
            mem::transmute::<_, &metal::MetalLayerRef>(layer).to_owned()
        } else {
            let metal_layer: metal::MetalLayer = msg_send![ca_ml_class, new];
            let () = msg_send![ns_view, setLayer: metal_layer.as_ref()];
            let () = msg_send![ns_view, setWantsLayer: YES];
            let bounds: CGRect = msg_send![ns_view, bounds];
            let () = msg_send![metal_layer, setFrame: bounds];

            if !ns_window.is_null() {
                let scale_factor: CGFloat = msg_send![ns_window, backingScaleFactor];
                let () = msg_send![metal_layer, setContentsScale: scale_factor];
            }
            // gfx-hal sets a delegate here
            metal_layer
        };
        let () = msg_send![metal_layer, setContentsGravity: kCAGravityTopLeft];
        Some(MtlSurface { layer: metal_layer })
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
                // The workgroup limits are taken from the minimum of a desktop installation;
                // we don't support iOS right now, but in case of testing on those devices it might
                // need to change these (or just queried properly).
                workgroup_limits: WorkgroupLimits {
                    max_size: [1024, 1024, 64],
                    max_invocations: 1024,
                },
                has_memory_model: false,
                use_staging_buffers,
            };
            Ok(MtlDevice {
                device,
                cmd_queue: Arc::new(Mutex::new(cmd_queue)),
                gpu_info,
            })
        } else {
            Err("can't create system default Metal device".into())
        }
    }

    pub unsafe fn swapchain(
        &self,
        _width: usize,
        _height: usize,
        device: &MtlDevice,
        surface: &MtlSurface,
    ) -> Result<MtlSwapchain, Error> {
        surface.layer.set_device(&device.device);
        let n_drawables = surface.layer.maximum_drawable_count() as usize;
        Ok(MtlSwapchain {
            layer: surface.layer.to_owned(),
            cmd_queue: device.cmd_queue.clone(),
            drawable: Default::default(),
            n_drawables,
            drawable_ix: 0,
        })
    }
}

impl crate::backend::Device for MtlDevice {
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

    unsafe fn destroy_buffer(&self, _buffer: &Self::Buffer) -> Result<(), Error> {
        // This defers dropping until the buffer object is dropped. We probably need
        // to rethink buffer lifetime if descriptor sets can retain references.
        Ok(())
    }

    unsafe fn create_image2d(&self, width: u32, height: u32) -> Result<Self::Image, Error> {
        let desc = metal::TextureDescriptor::new();
        desc.set_width(width as u64);
        desc.set_height(height as u64);
        // These are defaults so don't need to be explicitly set.
        //desc.set_depth(1);
        //desc.set_mipmap_level_count(1);
        //desc.set_pixel_format(metal::MTLPixelFormat::RGBA8Unorm);
        desc.set_usage(metal::MTLTextureUsage::ShaderRead | metal::MTLTextureUsage::ShaderWrite);
        let texture = self.device.new_texture(&desc);
        Ok(Image {
            texture,
            width,
            height,
        })
    }

    unsafe fn destroy_image(&self, _image: &Self::Image) -> Result<(), Error> {
        todo!()
    }

    unsafe fn pipeline_builder(&self) -> Self::PipelineBuilder {
        PipelineBuilder
    }

    unsafe fn descriptor_set_builder(&self) -> Self::DescriptorSetBuilder {
        DescriptorSetBuilder::default()
    }

    fn create_cmd_buf(&self) -> Result<Self::CmdBuf, Error> {
        let cmd_queue = self.cmd_queue.lock().unwrap();
        // consider new_command_buffer_with_unretained_references for performance
        let cmd_buf = cmd_queue.new_command_buffer();
        let cmd_buf = autoreleasepool(|| cmd_buf.to_owned());
        Ok(CmdBuf { cmd_buf })
    }

    unsafe fn destroy_cmd_buf(&self, _cmd_buf: Self::CmdBuf) -> Result<(), Error> {
        Ok(())
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
        _wait_semaphores: &[&Self::Semaphore],
        _signal_semaphores: &[&Self::Semaphore],
        fence: Option<&mut Self::Fence>,
    ) -> Result<(), Error> {
        for cmd_buf in cmd_bufs {
            cmd_buf.cmd_buf.commit();
        }
        if let Some(last_cmd_buf) = cmd_bufs.last() {
            if let Some(fence) = fence {
                *fence = Fence::CmdBufPending(last_cmd_buf.cmd_buf.to_owned());
            }
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
        Ok(Semaphore)
    }

    unsafe fn create_fence(&self, _signaled: bool) -> Result<Self::Fence, Error> {
        // Doesn't handle signaled case. Maybe the fences should have more
        // limited functionality than, say, Vulkan.
        Ok(Fence::Idle)
    }

    unsafe fn destroy_fence(&self, _fence: Self::Fence) -> Result<(), Error> {
        Ok(())
    }

    unsafe fn wait_and_reset(&self, fences: Vec<&mut Self::Fence>) -> Result<(), Error> {
        for fence in fences {
            match fence {
                Fence::Idle => (),
                Fence::CmdBufPending(cmd_buf) => {
                    cmd_buf.wait_until_completed();
                    // TODO: this would be a good place to check errors, currently
                    // dropped on the floor.
                    *fence = Fence::Idle;
                }
            }
        }
        Ok(())
    }

    unsafe fn get_fence_status(&self, fence: &mut Self::Fence) -> Result<bool, Error> {
        match fence {
            Fence::Idle => Ok(true),
            Fence::CmdBufPending(cmd_buf) => {
                Ok(cmd_buf.status() == metal::MTLCommandBufferStatus::Completed)
            }
        }
    }

    unsafe fn create_sampler(&self, params: crate::SamplerParams) -> Result<Self::Sampler, Error> {
        todo!()
    }
}

impl crate::backend::CmdBuf<MtlDevice> for CmdBuf {
    unsafe fn begin(&mut self) {}

    unsafe fn finish(&mut self) {}

    unsafe fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        descriptor_set: &DescriptorSet,
        workgroup_count: (u32, u32, u32),
        workgroup_size: (u32, u32, u32),
    ) {
        let encoder = self.cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.0);
        let mut buf_ix = 0;
        for buffer in &descriptor_set.buffers {
            encoder.set_buffer(buf_ix, Some(&buffer.buffer), 0);
            buf_ix += 1;
        }
        let mut img_ix = 0;
        for image in &descriptor_set.images {
            encoder.set_texture(img_ix, Some(&image.texture));
            img_ix += 1;
        }
        let workgroup_count = metal::MTLSize {
            width: workgroup_count.0 as u64,
            height: workgroup_count.1 as u64,
            depth: workgroup_count.2 as u64,
        };
        let workgroup_size = metal::MTLSize {
            width: workgroup_size.0 as u64,
            height: workgroup_size.1 as u64,
            depth: workgroup_size.2 as u64,
        };
        encoder.dispatch_thread_groups(workgroup_count, workgroup_size);
        encoder.end_encoding();
    }

    unsafe fn memory_barrier(&mut self) {
        // We'll probably move to explicit barriers, but for now rely on
        // Metal's own tracking.
    }

    unsafe fn host_barrier(&mut self) {}

    unsafe fn image_barrier(
        &mut self,
        _image: &Image,
        _src_layout: crate::ImageLayout,
        _dst_layout: crate::ImageLayout,
    ) {
        // I think these are being tracked.
    }

    unsafe fn clear_buffer(&self, buffer: &Buffer, size: Option<u64>) {
        todo!()
    }

    unsafe fn copy_buffer(&self, src: &Buffer, dst: &Buffer) {
        let encoder = self.cmd_buf.new_blit_command_encoder();
        let size = src.size.min(dst.size);
        encoder.copy_from_buffer(&src.buffer, 0, &dst.buffer, 0, size);
        encoder.end_encoding();
    }

    unsafe fn copy_image_to_buffer(&self, src: &Image, dst: &Buffer) {
        let encoder = self.cmd_buf.new_blit_command_encoder();
        assert_eq!(dst.size, (src.width as u64) * (src.height as u64) * 4);
        let bytes_per_row = (src.width * 4) as NSUInteger;
        let src_size = metal::MTLSize {
            width: src.width as NSUInteger,
            height: src.height as NSUInteger,
            depth: 1,
        };
        let origin = metal::MTLOrigin { x: 0, y: 0, z: 0 };
        encoder.copy_from_texture_to_buffer(
            &src.texture,
            0,
            0,
            origin,
            src_size,
            &dst.buffer,
            0,
            bytes_per_row,
            bytes_per_row * src.height as NSUInteger,
            metal::MTLBlitOption::empty(),
        );
        encoder.end_encoding();
    }

    unsafe fn copy_buffer_to_image(&self, src: &Buffer, dst: &Image) {
        let encoder = self.cmd_buf.new_blit_command_encoder();
        assert_eq!(src.size, (dst.width as u64) * (dst.height as u64) * 4);
        let bytes_per_row = (dst.width * 4) as NSUInteger;
        let src_size = metal::MTLSize {
            width: dst.width as NSUInteger,
            height: dst.height as NSUInteger,
            depth: 1,
        };
        let origin = metal::MTLOrigin { x: 0, y: 0, z: 0 };
        encoder.copy_from_buffer_to_texture(
            &src.buffer,
            0,
            bytes_per_row,
            bytes_per_row * dst.height as NSUInteger,
            src_size,
            &dst.texture,
            0,
            0,
            origin,
            metal::MTLBlitOption::empty(),
        );
        encoder.end_encoding();
    }

    unsafe fn blit_image(&self, src: &Image, dst: &Image) {
        let encoder = self.cmd_buf.new_blit_command_encoder();
        let src_size = metal::MTLSize {
            width: src.width.min(dst.width) as NSUInteger,
            height: src.width.min(dst.height) as NSUInteger,
            depth: 1,
        };
        let origin = metal::MTLOrigin { x: 0, y: 0, z: 0 };
        encoder.copy_from_texture(
            &src.texture,
            0,
            0,
            origin,
            src_size,
            &dst.texture,
            0,
            0,
            origin,
        );
        encoder.end_encoding();
    }

    unsafe fn reset_query_pool(&mut self, pool: &QueryPool) {}

    unsafe fn write_timestamp(&mut self, pool: &QueryPool, query: u32) {
        // TODO
        // This really a PITA because it's pretty different than Vulkan.
        // See https://developer.apple.com/documentation/metal/counter_sampling
    }
}

impl crate::backend::PipelineBuilder<MtlDevice> for PipelineBuilder {
    fn add_buffers(&mut self, _n_buffers: u32) {
        // My understanding is that Metal infers the pipeline layout from
        // the source.
    }

    fn add_images(&mut self, _n_images: u32) {}

    fn add_textures(&mut self, _max_textures: u32) {}

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
        let pipeline = device
            .device
            .new_compute_pipeline_state_with_function(&function)?;
        Ok(Pipeline(pipeline))
    }
}

impl crate::backend::DescriptorSetBuilder<MtlDevice> for DescriptorSetBuilder {
    fn add_buffers(&mut self, buffers: &[&Buffer]) {
        self.0.buffers.extend(buffers.iter().copied().cloned());
    }

    fn add_images(&mut self, images: &[&Image]) {
        self.0.images.extend(images.iter().copied().cloned());
    }

    fn add_textures(&mut self, images: &[&Image]) {
        self.add_images(images);
    }

    unsafe fn build(
        self,
        _device: &MtlDevice,
        _pipeline: &Pipeline,
    ) -> Result<DescriptorSet, Error> {
        Ok(self.0)
    }
}

impl MtlSwapchain {
    pub unsafe fn next(&mut self) -> Result<(usize, Semaphore), Error> {
        let drawable_ix = self.drawable_ix;
        self.drawable_ix = (drawable_ix + 1) % self.n_drawables;
        Ok((drawable_ix, Semaphore))
    }

    pub unsafe fn image(&self, _idx: usize) -> Image {
        let (drawable, texture) = autoreleasepool(|| {
            let drawable = self.layer.next_drawable().unwrap();
            (drawable.to_owned(), drawable.texture().to_owned())
        });
        *self.drawable.lock().unwrap() = Some(drawable);
        let size = self.layer.drawable_size();
        Image {
            texture,
            width: size.width.round() as u32,
            height: size.height.round() as u32,
        }
    }

    pub unsafe fn present(
        &self,
        _image_idx: usize,
        _semaphores: &[&Semaphore],
    ) -> Result<bool, Error> {
        let drawable = self.drawable.lock().unwrap().take();
        if let Some(drawable) = drawable {
            autoreleasepool(|| {
                let cmd_queue = self.cmd_queue.lock().unwrap();
                let cmd_buf = cmd_queue.new_command_buffer();
                cmd_buf.present_drawable(&drawable);
                cmd_buf.commit();
            });
        } else {
            println!("no drawable; present called without acquiring image?");
        }
        Ok(false)
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
