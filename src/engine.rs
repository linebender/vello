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
    collections::{hash_map::Entry, HashMap},
    num::{NonZeroU32, NonZeroU64},
    sync::atomic::{AtomicU64, Ordering},
};

use futures_intrusive::channel::shared::GenericOneshotReceiver;
use parking_lot::RawMutex;
use wgpu::{
    util::DeviceExt, BindGroup, BindGroupLayout, Buffer, BufferAsyncError, BufferSlice, BufferView,
    ComputePipeline, Device, Queue, Texture, TextureAspect, TextureFormat, TextureUsages,
    TextureView, TextureViewDimension,
};

pub type Error = Box<dyn std::error::Error>;

#[derive(Clone, Copy)]
pub struct ShaderId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(NonZeroU64);

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct Engine {
    shaders: Vec<Shader>,
}

struct Shader {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

#[derive(Default)]
pub struct Recording {
    commands: Vec<Command>,
}

#[derive(Clone, Copy)]
pub struct BufProxy {
    size: u64,
    id: Id,
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
    // Discussion question: third argument is vec of resources?
    // Maybe use tricks to make more ergonomic?
    Dispatch(ShaderId, (u32, u32, u32), Vec<ResourceProxy>),
    Download(BufProxy),
    Clear(BufProxy, u64, Option<NonZeroU64>),
}

#[derive(Default)]
pub struct Downloads {
    buf_map: HashMap<Id, Buffer>,
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

#[derive(Default)]
struct BindMap {
    buf_map: HashMap<Id, Buffer>,
    image_map: HashMap<Id, (Texture, TextureView)>,
}

impl Engine {
    pub fn new() -> Engine {
        Engine { shaders: vec![] }
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
        wgsl: Cow<'static, str>,
        layout: &[BindType],
    ) -> Result<ShaderId, Error> {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
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
                _ => todo!(),
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
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });
        let shader = Shader {
            pipeline,
            bind_group_layout,
        };
        let id = self.shaders.len();
        self.shaders.push(shader);
        Ok(ShaderId(id))
    }

    pub fn run_recording(
        &mut self,
        device: &Device,
        queue: &Queue,
        recording: &Recording,
        external_resources: &[ExternalResource],
    ) -> Result<Downloads, Error> {
        let mut bind_map = BindMap::default();
        let mut downloads = Downloads::default();

        let mut encoder = device.create_command_encoder(&Default::default());
        for command in &recording.commands {
            match command {
                Command::Upload(buf_proxy, bytes) => {
                    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: &bytes,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    });
                    bind_map.insert_buf(buf_proxy.id, buf);
                }
                Command::UploadUniform(buf_proxy, bytes) => {
                    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: &bytes,
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    bind_map.insert_buf(buf_proxy.id, buf);
                }
                Command::UploadImage(image_proxy, bytes) => {
                    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: &bytes,
                        usage: wgpu::BufferUsages::COPY_SRC,
                    });
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
                        format: image_proxy.format.to_wgpu(),
                    });
                    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
                        label: None,
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        mip_level_count: None,
                        base_mip_level: 0,
                        base_array_layer: 0,
                        array_layer_count: None,
                        format: Some(TextureFormat::Rgba8Unorm),
                    });
                    encoder.copy_buffer_to_texture(
                        wgpu::ImageCopyBuffer {
                            buffer: &buf,
                            layout: wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: NonZeroU32::new(image_proxy.width * 4),
                                rows_per_image: None,
                            },
                        },
                        wgpu::ImageCopyTexture {
                            texture: &texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                            aspect: TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: image_proxy.width,
                            height: image_proxy.height,
                            depth_or_array_layers: 1,
                        },
                    );
                    bind_map.insert_image(image_proxy.id, texture, texture_view)
                }
                Command::Dispatch(shader_id, wg_size, bindings) => {
                    // println!("dispatching {:?} with {} bindings", wg_size, bindings.len());
                    let shader = &self.shaders[shader_id.0];
                    let bind_group = bind_map.create_bind_group(
                        device,
                        &shader.bind_group_layout,
                        bindings,
                        external_resources,
                    )?;
                    let mut cpass = encoder.begin_compute_pass(&Default::default());
                    cpass.set_pipeline(&shader.pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch_workgroups(wg_size.0, wg_size.1, wg_size.2);
                }
                Command::Download(proxy) => {
                    let src_buf = bind_map.buf_map.get(&proxy.id).ok_or("buffer not in map")?;
                    let buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: proxy.size,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    encoder.copy_buffer_to_buffer(src_buf, 0, &buf, 0, proxy.size);
                    downloads.buf_map.insert(proxy.id, buf);
                }
                Command::Clear(proxy, offset, size) => {
                    let buffer = bind_map.get_or_create(*proxy, device)?;
                    encoder.clear_buffer(buffer, *offset, *size);
                }
            }
        }
        queue.submit(Some(encoder.finish()));
        Ok(downloads)
    }
}

impl Recording {
    pub fn push(&mut self, cmd: Command) {
        self.commands.push(cmd);
    }

    pub fn upload(&mut self, data: impl Into<Vec<u8>>) -> BufProxy {
        let data = data.into();
        let buf_proxy = BufProxy::new(data.len() as u64);
        self.push(Command::Upload(buf_proxy, data));
        buf_proxy
    }

    pub fn upload_uniform(&mut self, data: impl Into<Vec<u8>>) -> BufProxy {
        let data = data.into();
        let buf_proxy = BufProxy::new(data.len() as u64);
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

    pub fn dispatch<R>(&mut self, shader: ShaderId, wg_size: (u32, u32, u32), resources: R)
    where
        R: IntoIterator,
        R::Item: Into<ResourceProxy>,
    {
        self.push(Command::Dispatch(
            shader,
            wg_size,
            resources.into_iter().map(|r| r.into()).collect(),
        ));
    }

    pub fn download(&mut self, buf: BufProxy) {
        self.push(Command::Download(buf));
    }

    pub fn clear_all(&mut self, buf: BufProxy) {
        self.push(Command::Clear(buf, 0, None));
    }
}

impl BufProxy {
    pub fn new(size: u64) -> Self {
        let id = Id::next();
        BufProxy {
            id,
            size: size.max(16),
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
    pub fn new_buf(size: u64) -> Self {
        Self::Buf(BufProxy::new(size))
    }

    pub fn new_image(width: u32, height: u32, format: ImageFormat) -> Self {
        Self::Image(ImageProxy::new(width, height, format))
    }

    pub fn as_buf(&self) -> Option<&BufProxy> {
        match self {
            Self::Buf(proxy) => Some(&proxy),
            _ => None,
        }
    }

    pub fn as_image(&self) -> Option<&ImageProxy> {
        match self {
            Self::Image(proxy) => Some(&proxy),
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
    fn insert_buf(&mut self, id: Id, buf: Buffer) {
        self.buf_map.insert(id, buf);
    }

    fn insert_image(&mut self, id: Id, image: Texture, image_view: TextureView) {
        self.image_map.insert(id, (image, image_view));
    }

    fn create_bind_group(
        &mut self,
        device: &Device,
        layout: &BindGroupLayout,
        bindings: &[ResourceProxy],
        external_resources: &[ExternalResource],
    ) -> Result<BindGroup, Error> {
        // These functions are ugly and linear, but the remap array should generally be
        // small. Should find a better solution for this.
        fn find_buf<'a>(
            resources: &[ExternalResource<'a>],
            proxy: &BufProxy,
        ) -> Option<&'a Buffer> {
            for resource in resources {
                match resource {
                    ExternalResource::Buf(p, buf) if p.id == proxy.id => {
                        return Some(buf);
                    }
                    _ => {}
                }
            }
            None
        }
        fn find_image<'a>(
            resources: &[ExternalResource<'a>],
            proxy: &ImageProxy,
        ) -> Option<&'a TextureView> {
            for resource in resources {
                match resource {
                    ExternalResource::Image(p, view) if p.id == proxy.id => {
                        return Some(view);
                    }
                    _ => {}
                }
            }
            None
        }
        for proxy in bindings {
            match proxy {
                ResourceProxy::Buf(proxy) => {
                    if find_buf(external_resources, proxy).is_some() {
                        continue;
                    }
                    if let Entry::Vacant(v) = self.buf_map.entry(proxy.id) {
                        let buf = device.create_buffer(&wgpu::BufferDescriptor {
                            label: None,
                            size: proxy.size,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_DST
                                | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        });
                        v.insert(buf);
                    }
                }
                ResourceProxy::Image(proxy) => {
                    if find_image(external_resources, proxy).is_some() {
                        continue;
                    }
                    if let Entry::Vacant(v) = self.image_map.entry(proxy.id) {
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
                            format: proxy.format.to_wgpu(),
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
                    let buf = find_buf(external_resources, proxy)
                        .or_else(|| self.buf_map.get(&proxy.id))
                        .unwrap();
                    Ok(wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: buf.as_entire_binding(),
                    })
                }
                ResourceProxy::Image(proxy) => {
                    let view = find_image(external_resources, proxy)
                        .or_else(|| self.image_map.get(&proxy.id).map(|v| &v.1))
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

    fn get_or_create(&mut self, proxy: BufProxy, device: &Device) -> Result<&Buffer, Error> {
        match self.buf_map.entry(proxy.id) {
            Entry::Occupied(occupied) => Ok(occupied.into_mut()),
            Entry::Vacant(vacant) => {
                let buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: proxy.size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                Ok(vacant.insert(buf))
            }
        }
    }
}

pub struct DownloadsMapped<'a>(
    HashMap<
        Id,
        (
            BufferSlice<'a>,
            GenericOneshotReceiver<RawMutex, Result<(), BufferAsyncError>>,
        ),
    >,
);

impl Downloads {
    // Discussion: should API change so we get one buffer, rather than mapping all?
    pub fn map(&self) -> DownloadsMapped {
        let mut map = HashMap::new();
        for (id, buf) in &self.buf_map {
            let buf_slice = buf.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            map.insert(*id, (buf_slice, receiver));
        }
        DownloadsMapped(map)
    }
}

impl<'a> DownloadsMapped<'a> {
    pub async fn get_mapped(&self, proxy: BufProxy) -> Result<BufferView, Error> {
        let (slice, recv) = self.0.get(&proxy.id).ok_or("buffer not in map")?;
        if let Some(recv_result) = recv.receive().await {
            recv_result?;
        } else {
            return Err("channel was closed".into());
        }
        Ok(slice.get_mapped_range())
    }
}
