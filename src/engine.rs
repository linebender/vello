// Copyright 2022 Google LLC
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

use std::{
    borrow::Cow,
    cell::RefCell,
    collections::{hash_map::Entry, HashMap, HashSet},
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferUsages, CommandEncoder, ComputePipeline, Device,
    Queue, Texture, TextureAspect, TextureUsages, TextureView, TextureViewDimension,
};

use crate::cpu_dispatch::CpuBinding;

pub type Error = Box<dyn std::error::Error>;

#[derive(Clone, Copy)]
pub struct ShaderId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(NonZeroU64);

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct Engine {
    shaders: Vec<Shader>,
    pool: ResourcePool,
    bind_map: BindMap,
    downloads: HashMap<Id, Buffer>,
}

struct Shader {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    cpu_shader: Option<fn(u32, &[CpuBinding])>,
}

#[derive(Default)]
pub struct Recording {
    commands: Vec<Command>,
}

#[derive(Clone, Copy)]
pub struct BufProxy {
    size: u64,
    id: Id,
    name: &'static str,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ImageFormat {
    Rgba8,
    Bgra8,
}

#[derive(Clone, Copy)]
pub struct ImageProxy {
    width: u32,
    height: u32,
    format: ImageFormat,
    id: Id,
}

#[derive(Clone, Copy)]
pub enum ResourceProxy {
    Buf(BufProxy),
    Image(ImageProxy),
}

pub enum ExternalResource<'a> {
    Buf(BufProxy, &'a Buffer),
    Image(ImageProxy, &'a TextureView),
}

pub enum Command {
    Upload(BufProxy, Vec<u8>),
    UploadUniform(BufProxy, Vec<u8>),
    UploadImage(ImageProxy, Vec<u8>),
    WriteImage(ImageProxy, [u32; 4], Vec<u8>),
    // Discussion question: third argument is vec of resources?
    // Maybe use tricks to make more ergonomic?
    // Alternative: provide bufs & images as separate sequences
    Dispatch(ShaderId, (u32, u32, u32), Vec<ResourceProxy>),
    DispatchIndirect(ShaderId, BufProxy, u64, Vec<ResourceProxy>),
    Download(BufProxy),
    Clear(BufProxy, u64, Option<NonZeroU64>),
    FreeBuf(BufProxy),
    FreeImage(ImageProxy),
}

/// The type of resource that will be bound to a slot in a shader.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BindType {
    /// A storage buffer with read/write access.
    Buffer,
    /// A storage buffer with read only access.
    BufReadOnly,
    /// A small storage buffer to be used as uniforms.
    Uniform,
    /// A storage image.
    Image(ImageFormat),
    /// A storage image with read only access.
    ImageRead(ImageFormat),
    // TODO: Uniform, Sampler, maybe others
}

/// A buffer can exist either on the GPU or on CPU.
enum MaterializedBuffer {
    Gpu(Buffer),
    Cpu(RefCell<Vec<u8>>),
}

struct BindMapBuffer {
    buffer: MaterializedBuffer,
    #[cfg_attr(not(feature = "buffer_labels"), allow(unused))]
    label: &'static str,
}

#[derive(Default)]
struct BindMap {
    buf_map: HashMap<Id, BindMapBuffer>,
    image_map: HashMap<Id, (Texture, TextureView)>,
    pending_clears: HashSet<Id>,
}

#[derive(Hash, PartialEq, Eq)]
struct BufferProperties {
    size: u64,
    usages: BufferUsages,
    #[cfg(feature = "buffer_labels")]
    name: &'static str,
}

#[derive(Default)]
struct ResourcePool {
    bufs: HashMap<BufferProperties, Vec<Buffer>>,
}

/// The transient bind map contains short-lifetime resources.
///
/// In particular, it has resources scoped to a single call of
/// `run_recording()`, including external resources and also buffer
/// uploads.
#[derive(Default)]
struct TransientBindMap<'a> {
    bufs: HashMap<Id, TransientBuf<'a>>,
    // TODO: create transient image type
    images: HashMap<Id, &'a TextureView>,
}

enum TransientBuf<'a> {
    Cpu(&'a [u8]),
    Gpu(&'a Buffer),
}

impl Engine {
    pub fn new() -> Engine {
        Engine {
            shaders: vec![],
            pool: Default::default(),
            bind_map: Default::default(),
            downloads: Default::default(),
        }
    }

    /// Add a shader.
    ///
    /// This function is somewhat limited, it doesn't apply a label, only allows one bind group,
    /// doesn't support push constants, and entry point is hardcoded as "main".
    ///
    /// Maybe should do template instantiation here? But shader compilation pipeline feels maybe
    /// a bit separate.
    pub fn add_shader(
        &mut self,
        device: &Device,
        label: &'static str,
        wgsl: Cow<'static, str>,
        layout: &[BindType],
    ) -> Result<ShaderId, Error> {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl),
        });
        let entries = layout
            .iter()
            .enumerate()
            .map(|(i, bind_type)| match bind_type {
                BindType::Buffer | BindType::BufReadOnly => wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: *bind_type == BindType::BufReadOnly,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindType::Uniform => wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindType::Image(format) | BindType::ImageRead(format) => {
                    wgpu::BindGroupLayoutEntry {
                        binding: i as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: if *bind_type == BindType::ImageRead(*format) {
                            wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            }
                        } else {
                            wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: format.to_wgpu(),
                                view_dimension: wgpu::TextureViewDimension::D2,
                            }
                        },
                        count: None,
                    }
                }
            })
            .collect::<Vec<_>>();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });
        let cpu_shader = None;
        let shader = Shader {
            pipeline,
            bind_group_layout,
            cpu_shader,
        };
        let id = self.shaders.len();
        self.shaders.push(shader);
        Ok(ShaderId(id))
    }

    pub fn set_cpu_shader(&mut self, id: ShaderId, f: fn(u32, &[CpuBinding])) {
        self.shaders[id.0].cpu_shader = Some(f);
    }

    pub fn run_recording(
        &mut self,
        device: &Device,
        queue: &Queue,
        recording: &Recording,
        external_resources: &[ExternalResource],
    ) -> Result<(), Error> {
        let mut free_bufs: HashSet<Id> = Default::default();
        let mut free_images: HashSet<Id> = Default::default();
        let mut transient_map = TransientBindMap::new(external_resources);

        let mut encoder = device.create_command_encoder(&Default::default());
        for command in &recording.commands {
            match command {
                Command::Upload(buf_proxy, bytes) => {
                    transient_map
                        .bufs
                        .insert(buf_proxy.id, TransientBuf::Cpu(bytes));
                    let usage =
                        BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE;
                    let buf = self
                        .pool
                        .get_buf(buf_proxy.size, buf_proxy.name, usage, device);
                    // TODO: if buffer is newly created, might be better to make it mapped at creation
                    // and copy. However, we expect reuse will be most common.
                    queue.write_buffer(&buf, 0, bytes);
                    self.bind_map.insert_buf(buf_proxy, buf);
                }
                Command::UploadUniform(buf_proxy, bytes) => {
                    transient_map
                        .bufs
                        .insert(buf_proxy.id, TransientBuf::Cpu(bytes));
                    let usage = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
                    // Same consideration as above
                    let buf = self
                        .pool
                        .get_buf(buf_proxy.size, buf_proxy.name, usage, device);
                    queue.write_buffer(&buf, 0, bytes);
                    self.bind_map.insert_buf(buf_proxy, buf);
                }
                Command::UploadImage(image_proxy, bytes) => {
                    let format = image_proxy.format.to_wgpu();
                    let block_size = format
                        .block_size(None)
                        .expect("ImageFormat must have a valid block size");
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: None,
                        size: wgpu::Extent3d {
                            width: image_proxy.width,
                            height: image_proxy.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                        format,
                        view_formats: &[],
                    });
                    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
                        label: None,
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        mip_level_count: None,
                        base_mip_level: 0,
                        base_array_layer: 0,
                        array_layer_count: None,
                        format: Some(format),
                    });
                    queue.write_texture(
                        wgpu::ImageCopyTexture {
                            texture: &texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                            aspect: TextureAspect::All,
                        },
                        bytes,
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(image_proxy.width * block_size),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width: image_proxy.width,
                            height: image_proxy.height,
                            depth_or_array_layers: 1,
                        },
                    );
                    self.bind_map
                        .insert_image(image_proxy.id, texture, texture_view)
                }
                Command::WriteImage(proxy, [x, y, width, height], data) => {
                    if let Ok((texture, _)) = self.bind_map.get_or_create_image(*proxy, device) {
                        let format = proxy.format.to_wgpu();
                        let block_size = format
                            .block_size(None)
                            .expect("ImageFormat must have a valid block size");
                        queue.write_texture(
                            wgpu::ImageCopyTexture {
                                texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d { x: *x, y: *y, z: 0 },
                                aspect: TextureAspect::All,
                            },
                            &data[..],
                            wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(*width * block_size),
                                rows_per_image: None,
                            },
                            wgpu::Extent3d {
                                width: *width,
                                height: *height,
                                depth_or_array_layers: 1,
                            },
                        );
                    }
                }
                Command::Dispatch(shader_id, wg_size, bindings) => {
                    // println!("dispatching {:?} with {} bindings", wg_size, bindings.len());
                    let shader = &self.shaders[shader_id.0];
                    if let Some(cpu_shader) = shader.cpu_shader {
                        let resources =
                            transient_map.create_cpu_resources(&mut self.bind_map, bindings);
                        cpu_shader(wg_size.0, &resources);
                    } else {
                        let bind_group = transient_map.create_bind_group(
                            &mut self.bind_map,
                            &mut self.pool,
                            device,
                            queue,
                            &mut encoder,
                            &shader.bind_group_layout,
                            bindings,
                        )?;
                        let mut cpass = encoder.begin_compute_pass(&Default::default());
                        cpass.set_pipeline(&shader.pipeline);
                        cpass.set_bind_group(0, &bind_group, &[]);
                        cpass.dispatch_workgroups(wg_size.0, wg_size.1, wg_size.2);
                    }
                }
                Command::DispatchIndirect(shader_id, proxy, offset, bindings) => {
                    let shader = &self.shaders[shader_id.0];
                    if let Some(cpu_shader) = shader.cpu_shader {
                        let n_wg;
                        if let CpuBinding::BufferRW(b) = self.bind_map.get_cpu_buf(proxy.id) {
                            let slice = b.borrow();
                            let indirect: &[u32] = bytemuck::cast_slice(&slice);
                            n_wg = indirect[0];
                        } else {
                            panic!("indirect buffer missing from bind map");
                        }
                        let resources =
                            transient_map.create_cpu_resources(&mut self.bind_map, bindings);
                        cpu_shader(n_wg, &resources);
                    } else {
                        let bind_group = transient_map.create_bind_group(
                            &mut self.bind_map,
                            &mut self.pool,
                            device,
                            queue,
                            &mut encoder,
                            &shader.bind_group_layout,
                            bindings,
                        )?;
                        transient_map.materialize_gpu_buf_for_indirect(&mut self.bind_map, &mut self.pool, device, queue, proxy);
                        let mut cpass = encoder.begin_compute_pass(&Default::default());
                        cpass.set_pipeline(&shader.pipeline);
                        cpass.set_bind_group(0, &bind_group, &[]);
                        let buf = self
                            .bind_map
                            .get_gpu_buf(proxy.id)
                            .ok_or("buffer for indirect dispatch not in map")?;
                        cpass.dispatch_workgroups_indirect(&buf, *offset);
                    }
                }
                Command::Download(proxy) => {
                    let src_buf = self
                        .bind_map
                        .get_gpu_buf(proxy.id)
                        .ok_or("buffer not in map")?;
                    let usage = BufferUsages::MAP_READ | BufferUsages::COPY_DST;
                    let buf = self.pool.get_buf(proxy.size, "download", usage, device);
                    encoder.copy_buffer_to_buffer(&src_buf, 0, &buf, 0, proxy.size);
                    self.downloads.insert(proxy.id, buf);
                }
                Command::Clear(proxy, offset, size) => {
                    if let Some(buf) = self.bind_map.get_buf(*proxy) {
                        match &buf.buffer {
                            MaterializedBuffer::Gpu(b) => encoder.clear_buffer(&b, *offset, *size),
                            MaterializedBuffer::Cpu(b) => {
                                let mut slice = &mut b.borrow_mut()[*offset as usize..];
                                if let Some(size) = size {
                                    slice = &mut slice[..size.get() as usize];
                                }
                                for x in slice {
                                    *x = 0;
                                }
                            }
                        }
                    } else {
                        self.bind_map.pending_clears.insert(proxy.id);
                    }
                }
                Command::FreeBuf(proxy) => {
                    free_bufs.insert(proxy.id);
                }
                Command::FreeImage(proxy) => {
                    free_images.insert(proxy.id);
                }
            }
        }
        queue.submit(Some(encoder.finish()));
        for id in free_bufs {
            if let Some(buf) = self.bind_map.buf_map.remove(&id) {
                if let MaterializedBuffer::Gpu(gpu_buf) = buf.buffer {
                    let props = BufferProperties {
                        size: gpu_buf.size(),
                        usages: gpu_buf.usage(),
                        #[cfg(feature = "buffer_labels")]
                        name: buf.label,
                    };
                    self.pool.bufs.entry(props).or_default().push(gpu_buf);
                }
            }
        }
        for id in free_images {
            if let Some((texture, view)) = self.bind_map.image_map.remove(&id) {
                // TODO: have a pool to avoid needless re-allocation
                drop(texture);
                drop(view);
            }
        }
        Ok(())
    }

    pub fn get_download(&self, buf: BufProxy) -> Option<&Buffer> {
        self.downloads.get(&buf.id)
    }

    pub fn free_download(&mut self, buf: BufProxy) {
        self.downloads.remove(&buf.id);
    }
}

impl Recording {
    pub fn push(&mut self, cmd: Command) {
        self.commands.push(cmd);
    }

    pub fn upload(&mut self, name: &'static str, data: impl Into<Vec<u8>>) -> BufProxy {
        let data = data.into();
        let buf_proxy = BufProxy::new(data.len() as u64, name);
        self.push(Command::Upload(buf_proxy, data));
        buf_proxy
    }

    pub fn upload_uniform(&mut self, name: &'static str, data: impl Into<Vec<u8>>) -> BufProxy {
        let data = data.into();
        let buf_proxy = BufProxy::new(data.len() as u64, name);
        self.push(Command::UploadUniform(buf_proxy, data));
        buf_proxy
    }

    pub fn upload_image(
        &mut self,
        width: u32,
        height: u32,
        format: ImageFormat,
        data: impl Into<Vec<u8>>,
    ) -> ImageProxy {
        let data = data.into();
        let image_proxy = ImageProxy::new(width, height, format);
        self.push(Command::UploadImage(image_proxy, data));
        image_proxy
    }

    pub fn write_image(
        &mut self,
        image: ImageProxy,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        data: impl Into<Vec<u8>>,
    ) {
        let data = data.into();
        self.push(Command::WriteImage(image, [x, y, width, height], data));
    }

    pub fn dispatch<R>(&mut self, shader: ShaderId, wg_size: (u32, u32, u32), resources: R)
    where
        R: IntoIterator,
        R::Item: Into<ResourceProxy>,
    {
        let r = resources.into_iter().map(|r| r.into()).collect();
        self.push(Command::Dispatch(shader, wg_size, r));
    }

    pub fn dispatch_indirect<R>(
        &mut self,
        shader: ShaderId,
        buf: BufProxy,
        offset: u64,
        resources: R,
    ) where
        R: IntoIterator,
        R::Item: Into<ResourceProxy>,
    {
        let r = resources.into_iter().map(|r| r.into()).collect();
        self.push(Command::DispatchIndirect(shader, buf, offset, r));
    }

    /// Prepare a buffer for downloading.
    ///
    /// Currently this copies to a download buffer. The original buffer can be freed
    /// immediately after.
    pub fn download(&mut self, buf: BufProxy) {
        self.push(Command::Download(buf));
    }

    pub fn clear_all(&mut self, buf: BufProxy) {
        self.push(Command::Clear(buf, 0, None));
    }

    pub fn free_buf(&mut self, buf: BufProxy) {
        self.push(Command::FreeBuf(buf));
    }

    pub fn free_image(&mut self, image: ImageProxy) {
        self.push(Command::FreeImage(image));
    }

    pub fn free_resource(&mut self, resource: ResourceProxy) {
        match resource {
            ResourceProxy::Buf(buf) => self.free_buf(buf),
            ResourceProxy::Image(image) => self.free_image(image),
        }
    }
}

impl BufProxy {
    pub fn new(size: u64, name: &'static str) -> Self {
        let id = Id::next();
        BufProxy {
            id,
            size: size.max(16),
            name,
        }
    }
}

impl ImageFormat {
    pub fn to_wgpu(self) -> wgpu::TextureFormat {
        match self {
            Self::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
            Self::Bgra8 => wgpu::TextureFormat::Bgra8Unorm,
        }
    }
}

impl ImageProxy {
    pub fn new(width: u32, height: u32, format: ImageFormat) -> Self {
        let id = Id::next();
        ImageProxy {
            width,
            height,
            format,
            id,
        }
    }
}

impl ResourceProxy {
    pub fn new_buf(size: u64, name: &'static str) -> Self {
        Self::Buf(BufProxy::new(size, name))
    }

    pub fn new_image(width: u32, height: u32, format: ImageFormat) -> Self {
        Self::Image(ImageProxy::new(width, height, format))
    }

    pub fn as_buf(&self) -> Option<&BufProxy> {
        match self {
            Self::Buf(proxy) => Some(proxy),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<&ImageProxy> {
        match self {
            Self::Image(proxy) => Some(proxy),
            _ => None,
        }
    }
}

impl From<BufProxy> for ResourceProxy {
    fn from(value: BufProxy) -> Self {
        Self::Buf(value)
    }
}

impl From<ImageProxy> for ResourceProxy {
    fn from(value: ImageProxy) -> Self {
        Self::Image(value)
    }
}

impl Id {
    pub fn next() -> Id {
        let val = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        // could use new_unchecked
        Id(NonZeroU64::new(val + 1).unwrap())
    }
}

impl BindMap {
    fn insert_buf(&mut self, proxy: &BufProxy, buffer: Buffer) {
        self.buf_map.insert(
            proxy.id,
            BindMapBuffer {
                buffer: MaterializedBuffer::Gpu(buffer),
                label: proxy.name,
            },
        );
    }

    /// Get a buffer, only if it's on GPU.
    fn get_gpu_buf(&self, id: Id) -> Option<&Buffer> {
        self.buf_map.get(&id).and_then(|b| match &b.buffer {
            MaterializedBuffer::Gpu(b) => Some(b),
            _ => None,
        })
    }

    /// Get a CPU buffer.
    ///
    /// Panics if buffer is not present or is on GPU.
    fn get_cpu_buf(&self, id: Id) -> CpuBinding {
        match &self.buf_map[&id].buffer {
            MaterializedBuffer::Cpu(b) => CpuBinding::BufferRW(b),
            _ => panic!("getting cpu buffer, but it's on gpu"),
        }
    }

    fn materialize_cpu_buf(&mut self, buf: &BufProxy) {
        self.buf_map.entry(buf.id).or_insert_with(|| {
            let buffer = MaterializedBuffer::Cpu(RefCell::new(vec![0; buf.size as usize]));
            BindMapBuffer {
                buffer,
                // TODO: do we need to cfg this?
                label: buf.name,
            }
        });
    }

    fn insert_image(&mut self, id: Id, image: Texture, image_view: TextureView) {
        self.image_map.insert(id, (image, image_view));
    }

    /// Get a buffer (CPU or GPU), create on GPU if not present.
    fn get_or_create(
        &mut self,
        proxy: BufProxy,
        device: &Device,
        pool: &mut ResourcePool,
    ) -> Result<&MaterializedBuffer, Error> {
        match self.buf_map.entry(proxy.id) {
            Entry::Occupied(occupied) => Ok(&occupied.into_mut().buffer),
            Entry::Vacant(vacant) => {
                let usage = BufferUsages::COPY_SRC | BufferUsages::COPY_DST | BufferUsages::STORAGE;
                let buf = pool.get_buf(proxy.size, proxy.name, usage, device);
                Ok(&vacant
                    .insert(BindMapBuffer {
                        buffer: MaterializedBuffer::Gpu(buf),
                        label: proxy.name,
                    })
                    .buffer)
            }
        }
    }

    fn get_buf(&mut self, proxy: BufProxy) -> Option<&BindMapBuffer> {
        self.buf_map.get(&proxy.id)
    }

    fn get_or_create_image(
        &mut self,
        proxy: ImageProxy,
        device: &Device,
    ) -> Result<&(Texture, TextureView), Error> {
        match self.image_map.entry(proxy.id) {
            Entry::Occupied(occupied) => Ok(occupied.into_mut()),
            Entry::Vacant(vacant) => {
                let format = proxy.format.to_wgpu();
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: proxy.width,
                        height: proxy.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                    format,
                    view_formats: &[],
                });
                let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
                    label: None,
                    dimension: Some(TextureViewDimension::D2),
                    aspect: TextureAspect::All,
                    mip_level_count: None,
                    base_mip_level: 0,
                    base_array_layer: 0,
                    array_layer_count: None,
                    format: Some(proxy.format.to_wgpu()),
                });
                Ok(vacant.insert((texture, texture_view)))
            }
        }
    }
}

const SIZE_CLASS_BITS: u32 = 1;

impl ResourcePool {
    /// Get a buffer from the pool or create one.
    fn get_buf(
        &mut self,
        size: u64,
        name: &'static str,
        usage: BufferUsages,
        device: &Device,
    ) -> Buffer {
        let rounded_size = Self::size_class(size, SIZE_CLASS_BITS);
        let props = BufferProperties {
            size: rounded_size,
            usages: usage,
            #[cfg(feature = "buffer_labels")]
            name,
        };
        if let Some(buf_vec) = self.bufs.get_mut(&props) {
            if let Some(buf) = buf_vec.pop() {
                return buf;
            }
        }
        device.create_buffer(&wgpu::BufferDescriptor {
            #[cfg(feature = "buffer_labels")]
            label: Some(name),
            #[cfg(not(feature = "buffer_labels"))]
            label: None,
            size: rounded_size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Quantize a size up to the nearest size class.
    fn size_class(x: u64, bits: u32) -> u64 {
        if x > 1 << bits {
            let a = (x - 1).leading_zeros();
            let b = (x - 1) | (((u64::MAX / 2) >> bits) >> a);
            b + 1
        } else {
            1 << bits
        }
    }
}

impl BindMapBuffer {
    fn upload_if_needed(
        &mut self,
        proxy: &BufProxy,
        device: &Device,
        queue: &Queue,
        pool: &mut ResourcePool,
    ) {
        if let MaterializedBuffer::Cpu(cpu_buf) = &self.buffer {
            let usage = BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST
                | BufferUsages::STORAGE
                | BufferUsages::INDIRECT;
            let buf = pool.get_buf(proxy.size, proxy.name, usage, device);
            queue.write_buffer(&buf, 0, &cpu_buf.borrow());
            self.buffer = MaterializedBuffer::Gpu(buf);
        }
    }
}

impl<'a> TransientBindMap<'a> {
    /// Create new transient bind map, seeded from external resources
    fn new(external_resources: &'a [ExternalResource]) -> Self {
        let mut bufs = HashMap::default();
        let mut images = HashMap::default();
        for resource in external_resources {
            match resource {
                ExternalResource::Buf(proxy, gpu_buf) => {
                    bufs.insert(proxy.id, TransientBuf::Gpu(gpu_buf));
                }
                ExternalResource::Image(proxy, gpu_image) => {
                    images.insert(proxy.id, *gpu_image);
                }
            }
        }
        TransientBindMap { bufs, images }
    }

    fn materialize_gpu_buf_for_indirect(
        &mut self,
        bind_map: &mut BindMap,
        pool: &mut ResourcePool,
        device: &Device,
        queue: &Queue,
        buf: &BufProxy,
    ) {
        if !self.bufs.contains_key(&buf.id) {
            if let Some(b) = bind_map.buf_map.get_mut(&buf.id) {
                b.upload_if_needed(&buf, device, queue, pool);
            }
        }
    }

    fn create_bind_group(
        &mut self,
        bind_map: &mut BindMap,
        pool: &mut ResourcePool,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        layout: &BindGroupLayout,
        bindings: &[ResourceProxy],
    ) -> Result<BindGroup, Error> {
        for proxy in bindings {
            match proxy {
                ResourceProxy::Buf(proxy) => {
                    if self.bufs.contains_key(&proxy.id) {
                        continue;
                    }
                    match bind_map.buf_map.entry(proxy.id) {
                        Entry::Vacant(v) => {
                            // TODO: only some buffers will need indirect, but does it hurt?
                            let usage = BufferUsages::COPY_SRC
                                | BufferUsages::COPY_DST
                                | BufferUsages::STORAGE
                                | BufferUsages::INDIRECT;
                            let buf = pool.get_buf(proxy.size, proxy.name, usage, device);
                            if bind_map.pending_clears.remove(&proxy.id) {
                                encoder.clear_buffer(&buf, 0, None);
                            }
                            v.insert(BindMapBuffer {
                                buffer: MaterializedBuffer::Gpu(buf),
                                label: proxy.name,
                            });
                        }
                        Entry::Occupied(mut o) => {
                            o.get_mut().upload_if_needed(proxy, device, queue, pool)
                        }
                    }
                }
                ResourceProxy::Image(proxy) => {
                    if self.images.contains_key(&proxy.id) {
                        continue;
                    }
                    if let Entry::Vacant(v) = bind_map.image_map.entry(proxy.id) {
                        let format = proxy.format.to_wgpu();
                        let texture = device.create_texture(&wgpu::TextureDescriptor {
                            label: None,
                            size: wgpu::Extent3d {
                                width: proxy.width,
                                height: proxy.height,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                            format,
                            view_formats: &[],
                        });
                        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
                            label: None,
                            dimension: Some(TextureViewDimension::D2),
                            aspect: TextureAspect::All,
                            mip_level_count: None,
                            base_mip_level: 0,
                            base_array_layer: 0,
                            array_layer_count: None,
                            format: Some(proxy.format.to_wgpu()),
                        });
                        v.insert((texture, texture_view));
                    }
                }
            }
        }
        let entries = bindings
            .iter()
            .enumerate()
            .map(|(i, proxy)| match proxy {
                ResourceProxy::Buf(proxy) => {
                    let buf = match self.bufs.get(&proxy.id) {
                        Some(TransientBuf::Gpu(b)) => b,
                        _ => bind_map.get_gpu_buf(proxy.id).unwrap(),
                    };
                    Ok(wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: buf.as_entire_binding(),
                    })
                }
                ResourceProxy::Image(proxy) => {
                    let view = self
                        .images
                        .get(&proxy.id)
                        .copied()
                        .or_else(|| bind_map.image_map.get(&proxy.id).map(|v| &v.1))
                        .unwrap();
                    Ok(wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: wgpu::BindingResource::TextureView(view),
                    })
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        });
        Ok(bind_group)
    }

    fn create_cpu_resources(
        &self,
        bind_map: &'a mut BindMap,
        bindings: &[ResourceProxy],
    ) -> Vec<CpuBinding> {
        // First pass is mutable; create buffers as needed
        for resource in bindings {
            match resource {
                ResourceProxy::Buf(buf) => match self.bufs.get(&buf.id) {
                    Some(TransientBuf::Cpu(_)) => (),
                    _ => bind_map.materialize_cpu_buf(buf),
                },
                ResourceProxy::Image(_) => todo!(),
            };
        }
        // Second pass takes immutable references
        bindings
            .iter()
            .map(|resource| match resource {
                ResourceProxy::Buf(buf) => match self.bufs.get(&buf.id) {
                    Some(TransientBuf::Cpu(b)) => CpuBinding::Buffer(b),
                    _ => bind_map.get_cpu_buf(buf.id),
                },
                ResourceProxy::Image(_) => todo!(),
            })
            .collect()
    }
}
