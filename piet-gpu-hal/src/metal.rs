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

mod clear;
mod timer;
mod util;

use std::mem;
use std::sync::{Arc, Mutex};

use block::Block;
use cocoa_foundation::base::id;
use cocoa_foundation::foundation::{NSInteger, NSUInteger};
use foreign_types::ForeignType;
use objc::rc::autoreleasepool;
use objc::runtime::{Object, BOOL, YES};
use objc::{class, msg_send, sel, sel_impl};

use metal::{CGFloat, CommandBufferRef, MTLFeatureSet};

use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};

use crate::{
    BufferUsage, ComputePassDescriptor, Error, GpuInfo, ImageFormat, MapMode, WorkgroupLimits,
};

use util::*;

use self::timer::{CounterSampleBuffer, CounterSet, TimeCalibration};

pub struct MtlInstance;

pub struct MtlDevice {
    device: metal::Device,
    cmd_queue: Arc<Mutex<metal::CommandQueue>>,
    gpu_info: GpuInfo,
    helpers: Arc<Helpers>,
    timer_set: Option<CounterSet>,
    counter_style: CounterStyle,
}

/// Type of counter sampling.
///
/// See https://developer.apple.com/documentation/metal/counter_sampling/sampling_gpu_data_into_counter_sample_buffers
#[derive(Clone, Copy, PartialEq, Eq)]
enum CounterStyle {
    None,
    Stage,
    Command,
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
    helpers: Arc<Helpers>,
    cur_encoder: Encoder,
    time_calibration: Arc<Mutex<TimeCalibration>>,
    counter_style: CounterStyle,
}

enum Encoder {
    None,
    Compute(metal::ComputeCommandEncoder),
    Blit(metal::BlitCommandEncoder),
}

#[derive(Default)]
pub struct QueryPool {
    counter_sample_buf: Option<CounterSampleBuffer>,
    calibration: Arc<Mutex<Option<Arc<Mutex<TimeCalibration>>>>>,
}

pub struct Pipeline(metal::ComputePipelineState);

#[derive(Default)]
pub struct DescriptorSetBuilder(DescriptorSet);

#[derive(Default)]
pub struct DescriptorSet {
    buffers: Vec<Buffer>,
    images: Vec<Image>,
}

struct Helpers {
    clear_pipeline: metal::ComputePipelineState,
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
            Ok(MtlDevice::new_from_raw_mtl(device, cmd_queue))
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

impl MtlDevice {
    pub fn new_from_raw_mtl(device: metal::Device, cmd_queue: metal::CommandQueue) -> MtlDevice {
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
        let helpers = Arc::new(Helpers {
            clear_pipeline: clear::make_clear_pipeline(&device),
        });
        // Timer stuff
        let timer_set = CounterSet::get_timer_counter_set(&device);
        let counter_style = if timer_set.is_some() {
            // TODO: M1 is stage style, but should do proper runtime detection.
            CounterStyle::Stage
        } else {
            CounterStyle::None
        };

        MtlDevice {
            device,
            cmd_queue: Arc::new(Mutex::new(cmd_queue)),
            gpu_info,
            helpers,
            timer_set,
            counter_style,
        }
    }

    pub fn cmd_buf_from_raw_mtl(&self, raw_cmd_buf: metal::CommandBuffer) -> CmdBuf {
        let cmd_buf = raw_cmd_buf;
        let helpers = self.helpers.clone();
        let cur_encoder = Encoder::None;
        let time_calibration = Default::default();
        CmdBuf {
            cmd_buf,
            helpers,
            cur_encoder,
            time_calibration,
            counter_style: self.counter_style,
        }
    }

    pub fn image_from_raw_mtl(&self, texture: metal::Texture, width: u32, height: u32) -> Image {
        Image {
            texture,
            width,
            height,
        }
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

    unsafe fn create_image2d(
        &self,
        width: u32,
        height: u32,
        format: ImageFormat,
    ) -> Result<Self::Image, Error> {
        let desc = metal::TextureDescriptor::new();
        desc.set_width(width as u64);
        desc.set_height(height as u64);
        // These are defaults so don't need to be explicitly set.
        //desc.set_depth(1);
        //desc.set_mipmap_level_count(1);
        let mtl_format = match format {
            ImageFormat::A8 => metal::MTLPixelFormat::R8Unorm,
            ImageFormat::Rgba8 => metal::MTLPixelFormat::RGBA8Unorm,
        };
        desc.set_pixel_format(mtl_format);
        desc.set_usage(metal::MTLTextureUsage::ShaderRead | metal::MTLTextureUsage::ShaderWrite);
        let texture = self.device.new_texture(&desc);
        Ok(Image {
            texture,
            width,
            height,
        })
    }

    unsafe fn destroy_image(&self, _image: &Self::Image) -> Result<(), Error> {
        // TODO figure out what we want to do here
        Ok(())
    }

    unsafe fn create_compute_pipeline(
        &self,
        code: &Self::ShaderSource,
        _bind_types: &[crate::BindType],
    ) -> Result<Self::Pipeline, Error> {
        let options = metal::CompileOptions::new();
        let library = self.device.new_library_with_source(code, &options)?;
        let function = library.get_function("main0", None)?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)?;
        Ok(Pipeline(pipeline))
    }

    unsafe fn descriptor_set_builder(&self) -> Self::DescriptorSetBuilder {
        DescriptorSetBuilder::default()
    }

    fn create_cmd_buf(&self) -> Result<Self::CmdBuf, Error> {
        let cmd_queue = self.cmd_queue.lock().unwrap();
        // A discussion about autorelease pools.
        //
        // Autorelease pools are a sore point in Rust/Objective-C interop. Basically,
        // you can have any two of correctness, ergonomics, and performance. Here we've
        // chosen the first two, using the pattern of a fine grained autorelease pool
        // to give the Obj-C object Rust-like lifetime semantics whenever objects are
        // created as autorelease (by convention, this is any object creation with an
        // Obj-C method name that doesn't begin with "new" or "alloc").
        //
        // To gain back some of the performance, we'd need a way to wrap an autorelease
        // pool over a chunk of work - that could be one frame of rendering, but for
        // tests that iterate a number of command buffer submissions, it would need to
        // be around that. On non-mac platforms, it would be a no-op.
        //
        // In any case, this way, the caller doesn't need to worry, and the performance
        // hit might not be so bad (perhaps we should measure).

        // consider new_command_buffer_with_unretained_references for performance
        let cmd_buf = autoreleasepool(|| cmd_queue.new_command_buffer().to_owned());
        let helpers = self.helpers.clone();
        let cur_encoder = Encoder::None;
        let time_calibration = Default::default();
        Ok(CmdBuf {
            cmd_buf,
            helpers,
            cur_encoder,
            time_calibration,
            counter_style: self.counter_style,
        })
    }

    unsafe fn destroy_cmd_buf(&self, _cmd_buf: Self::CmdBuf) -> Result<(), Error> {
        Ok(())
    }

    fn create_query_pool(&self, n_queries: u32) -> Result<Self::QueryPool, Error> {
        if let Some(timer_set) = &self.timer_set {
            let pool = CounterSampleBuffer::new(&self.device, n_queries as u64, timer_set)
                .ok_or("error creating timer query pool")?;
            return Ok(QueryPool {
                counter_sample_buf: Some(pool),
                calibration: Default::default(),
            });
        }
        Ok(QueryPool::default())
    }

    unsafe fn fetch_query_pool(&self, pool: &Self::QueryPool) -> Result<Vec<f64>, Error> {
        if let Some(raw) = &pool.counter_sample_buf {
            let resolved = raw.resolve();
            let calibration = pool.calibration.lock().unwrap();
            if let Some(calibration) = &*calibration {
                let calibration = calibration.lock().unwrap();
                let result = resolved
                    .iter()
                    .map(|time_ns| calibration.correlate(*time_ns))
                    .collect();
                return Ok(result);
            }
        }
        // Maybe should return None indicating it wasn't successful? But that might break.
        Ok(Vec::new())
    }

    unsafe fn run_cmd_bufs(
        &self,
        cmd_bufs: &[&Self::CmdBuf],
        _wait_semaphores: &[&Self::Semaphore],
        _signal_semaphores: &[&Self::Semaphore],
        fence: Option<&mut Self::Fence>,
    ) -> Result<(), Error> {
        unsafe fn add_scheduled_handler(
            cmd_buf: &metal::CommandBufferRef,
            block: &Block<(&CommandBufferRef,), ()>,
        ) {
            msg_send![cmd_buf, addScheduledHandler: block]
        }
        for cmd_buf in cmd_bufs {
            let time_calibration = cmd_buf.time_calibration.clone();
            let start_block = block::ConcreteBlock::new(move |buffer: &metal::CommandBufferRef| {
                let device: id = msg_send![buffer, device];
                let mut time_calibration = time_calibration.lock().unwrap();
                let cpu_ts_ptr = &mut time_calibration.cpu_start_ts as *mut _;
                let gpu_ts_ptr = &mut time_calibration.gpu_start_ts as *mut _;
                // TODO: only do this if supported.
                let () = msg_send![device, sampleTimestamps: cpu_ts_ptr gpuTimestamp: gpu_ts_ptr];
            })
            .copy();
            add_scheduled_handler(&cmd_buf.cmd_buf, &start_block);
            let time_calibration = cmd_buf.time_calibration.clone();
            let completed_block =
                block::ConcreteBlock::new(move |buffer: &metal::CommandBufferRef| {
                    let device: id = msg_send![buffer, device];
                    let mut time_calibration = time_calibration.lock().unwrap();
                    let cpu_ts_ptr = &mut time_calibration.cpu_end_ts as *mut _;
                    let gpu_ts_ptr = &mut time_calibration.gpu_end_ts as *mut _;
                    // TODO: only do this if supported.
                    let () =
                        msg_send![device, sampleTimestamps: cpu_ts_ptr gpuTimestamp: gpu_ts_ptr];
                })
                .copy();
            cmd_buf.cmd_buf.add_completed_handler(&completed_block);
            cmd_buf.cmd_buf.commit();
        }
        if let Some(last_cmd_buf) = cmd_bufs.last() {
            if let Some(fence) = fence {
                *fence = Fence::CmdBufPending(last_cmd_buf.cmd_buf.to_owned());
            }
        }
        Ok(())
    }

    unsafe fn map_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        _size: u64,
        _mode: MapMode,
    ) -> Result<*mut u8, Error> {
        let contents_ptr = buffer.buffer.contents();
        if contents_ptr.is_null() {
            return Err("probably trying to map private buffer".into());
        }
        Ok((contents_ptr as *mut u8).add(offset as usize))
    }

    unsafe fn unmap_buffer(
        &self,
        _buffer: &Self::Buffer,
        _offset: u64,
        _size: u64,
        _mode: MapMode,
    ) -> Result<(), Error> {
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

    unsafe fn finish(&mut self) {
        self.flush_encoder();
    }

    unsafe fn reset(&mut self) -> bool {
        false
    }

    unsafe fn begin_compute_pass(&mut self, desc: &ComputePassDescriptor) {
        // TODO: we might want to get better about validation but the following
        // assert is likely to trigger, and also a case can be made that
        // validation should be done at the hub level, for consistency.
        //debug_assert!(matches!(self.cur_encoder, Encoder::None));
        self.flush_encoder();
        autoreleasepool(|| {
            let encoder = if let Some(queries) = &desc.timer_queries {
                let descriptor: id =
                    msg_send![class!(MTLComputePassDescriptor), computePassDescriptor];
                let attachments: id = msg_send![descriptor, sampleBufferAttachments];
                let index: NSUInteger = 0;
                let attachment: id = msg_send![attachments, objectAtIndexedSubscript: index];
                // Here we break the hub/mux separation a bit, for expedience
                #[allow(irrefutable_let_patterns)]
                if let crate::hub::QueryPool::Mtl(query_pool) = queries.0 {
                    if let Some(sample_buf) = &query_pool.counter_sample_buf {
                        let () = msg_send![attachment, setSampleBuffer: sample_buf.id()];
                    }
                }
                let start_index = queries.1 as NSUInteger;
                let end_index = queries.2 as NSInteger;
                let () = msg_send![attachment, setStartOfEncoderSampleIndex: start_index];
                let () = msg_send![attachment, setEndOfEncoderSampleIndex: end_index];
                msg_send![
                    self.cmd_buf,
                    computeCommandEncoderWithDescriptor: descriptor
                ]
            } else {
                self.cmd_buf.new_compute_command_encoder()
            };
            self.cur_encoder = Encoder::Compute(encoder.to_owned());
        });
    }

    unsafe fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        descriptor_set: &DescriptorSet,
        workgroup_count: (u32, u32, u32),
        workgroup_size: (u32, u32, u32),
    ) {
        let encoder = self.compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline.0);
        let mut buf_ix = 0;
        for buffer in &descriptor_set.buffers {
            encoder.set_buffer(buf_ix, Some(&buffer.buffer), 0);
            buf_ix += 1;
        }
        let mut img_ix = buf_ix;
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
    }

    unsafe fn end_compute_pass(&mut self) {
        // TODO: might validate that we are in a compute encoder state
        self.flush_encoder();
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

    unsafe fn clear_buffer(&mut self, buffer: &Buffer, size: Option<u64>) {
        let size = size.unwrap_or(buffer.size);
        let _ = self.compute_command_encoder();
        // Getting this directly is a workaround for a borrow checker issue.
        if let Encoder::Compute(e) = &self.cur_encoder {
            clear::encode_clear(e, &self.helpers.clear_pipeline, &buffer.buffer, size);
        }
    }

    unsafe fn copy_buffer(&mut self, src: &Buffer, dst: &Buffer) {
        let encoder = self.blit_command_encoder();
        let size = src.size.min(dst.size);
        encoder.copy_from_buffer(&src.buffer, 0, &dst.buffer, 0, size);
    }

    unsafe fn copy_image_to_buffer(&mut self, src: &Image, dst: &Buffer) {
        let encoder = self.blit_command_encoder();
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
    }

    unsafe fn copy_buffer_to_image(&mut self, src: &Buffer, dst: &Image) {
        let encoder = self.blit_command_encoder();
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
    }

    unsafe fn blit_image(&mut self, src: &Image, dst: &Image) {
        let encoder = self.blit_command_encoder();
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
    }

    unsafe fn reset_query_pool(&mut self, pool: &QueryPool) {
        let mut calibration = pool.calibration.lock().unwrap();
        *calibration = Some(self.time_calibration.clone());
    }

    unsafe fn write_timestamp(&mut self, pool: &QueryPool, query: u32) {
        if let Some(buf) = &pool.counter_sample_buf {
            if matches!(self.cur_encoder, Encoder::None) {
                self.cur_encoder =
                    Encoder::Compute(self.cmd_buf.new_compute_command_encoder().to_owned());
            }
            let sample_index = query as NSUInteger;
            if self.counter_style == CounterStyle::Command {
                match &self.cur_encoder {
                    Encoder::Compute(e) => {
                        let () = msg_send![e.as_ptr(), sampleCountersInBuffer: buf.id() atSampleIndex: sample_index withBarrier: true];
                    }
                    Encoder::None => unreachable!(),
                    _ => todo!(),
                }
            } else if self.counter_style == CounterStyle::Stage {
                match &self.cur_encoder {
                    Encoder::Compute(_e) => {
                        println!("write_timestamp is not supported for stage-style encoders");
                    }
                    _ => (),
                }
            }
        }
    }
}

impl CmdBuf {
    fn compute_command_encoder(&mut self) -> &metal::ComputeCommandEncoder {
        if !matches!(self.cur_encoder, Encoder::Compute(_)) {
            self.flush_encoder();
            self.cur_encoder =
                Encoder::Compute(self.cmd_buf.new_compute_command_encoder().to_owned());
        }
        if let Encoder::Compute(e) = &self.cur_encoder {
            e
        } else {
            unreachable!()
        }
    }

    fn blit_command_encoder(&mut self) -> &metal::BlitCommandEncoder {
        if !matches!(self.cur_encoder, Encoder::Blit(_)) {
            self.flush_encoder();
            self.cur_encoder = Encoder::Blit(self.cmd_buf.new_blit_command_encoder().to_owned());
        }
        if let Encoder::Blit(e) = &self.cur_encoder {
            e
        } else {
            unreachable!()
        }
    }

    fn flush_encoder(&mut self) {
        match std::mem::replace(&mut self.cur_encoder, Encoder::None) {
            Encoder::Compute(e) => e.end_encoding(),
            Encoder::Blit(e) => e.end_encoding(),
            Encoder::None => (),
        }
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
