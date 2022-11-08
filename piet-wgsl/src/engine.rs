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
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

use futures_intrusive::channel::shared::GenericOneshotReceiver;
use parking_lot::RawMutex;
use wgpu::{
    util::DeviceExt, BindGroup, BindGroupLayout, Buffer, BufferAsyncError, BufferSlice, BufferView,
    ComputePipeline, Device, Queue,
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

pub enum DownloadBufUsage {
    MapRead,
    BlitSrc,
}

pub enum Command {
    Upload(BufProxy, Vec<u8>),
    // Discussion question: third argument is vec of resources?
    // Maybe use tricks to make more ergonomic?
    // Alternative: provide bufs & images as separate sequences, like piet-gpu.
    Dispatch(ShaderId, (u32, u32, u32), Vec<BufProxy>),

    // TODO(armansito): The second field is currently a stop-gap to make the Buffer->PNG and
    // Buffer->Blit->Swapchain modes work for the WebGPU example. Instead having a version that
    // returns a texture is probably more future-proof.
    Download(BufProxy, DownloadBufUsage),
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
    /// A storage image.
    #[allow(unused)] // TODO
    Image,
    /// A storage image with read only access.
    #[allow(unused)] // TODO
    ImageRead,
    // TODO: Uniform, Sampler, maybe others
}

#[derive(Default)]
struct BindMap {
    buf_map: HashMap<Id, Buffer>,
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
                Command::Dispatch(shader_id, wg_size, bindings) => {
                    let shader = &self.shaders[shader_id.0];
                    let bind_group =
                        bind_map.create_bind_group(device, &shader.bind_group_layout, bindings)?;
                    let mut cpass = encoder.begin_compute_pass(&Default::default());
                    cpass.set_pipeline(&shader.pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    cpass.dispatch_workgroups(wg_size.0, wg_size.1, wg_size.2);
                }
                Command::Download(proxy, usage) => {
                    let src_buf = bind_map.buf_map.get(&proxy.id).ok_or("buffer not in map")?;
                    let buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: proxy.size,
                        usage: match usage {
                            DownloadBufUsage::MapRead => {
                                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
                            }
                            DownloadBufUsage::BlitSrc => {
                                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
                            }
                        },
                        mapped_at_creation: false,
                    });
                    encoder.copy_buffer_to_buffer(src_buf, 0, &buf, 0, proxy.size);
                    downloads.buf_map.insert(proxy.id, buf);
                }
                Command::Clear(proxy, offset, size) => {
                    let buffer = bind_map.get_or_create(*proxy, device)?;
                    encoder.clear_buffer(buffer, *offset, *size)
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

    pub fn dispatch(
        &mut self,
        shader: ShaderId,
        wg_size: (u32, u32, u32),
        resources: impl Into<Vec<BufProxy>>,
    ) {
        self.push(Command::Dispatch(shader, wg_size, resources.into()));
    }

    pub fn download(&mut self, buf: BufProxy, usage: DownloadBufUsage) {
        self.push(Command::Download(buf, usage));
    }

    pub fn clear_all(&mut self, buf: BufProxy) {
        self.push(Command::Clear(buf, 0, None));
    }
}

impl BufProxy {
    pub fn new(size: u64) -> Self {
        let id = Id::next();
        BufProxy { id, size }
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

    fn create_bind_group(
        &mut self,
        device: &Device,
        layout: &BindGroupLayout,
        bindings: &[BufProxy],
    ) -> Result<BindGroup, Error> {
        for proxy in bindings {
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
        let entries = bindings
            .iter()
            .enumerate()
            .map(|(i, proxy)| {
                let buf = self.buf_map.get(&proxy.id).unwrap();
                Ok(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buf.as_entire_binding(),
                })
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

    pub fn get_buffer(&self, proxy: &BufProxy) -> &Buffer {
        self.buf_map.get(&proxy.id).unwrap()
    }
}

impl<'a> DownloadsMapped<'a> {
    pub async fn get_mapped(&self, proxy: BufProxy) -> Result<BufferView<'a>, Error> {
        let (slice, recv) = self.0.get(&proxy.id).ok_or("buffer not in map")?;
        if let Some(recv_result) = recv.receive().await {
            recv_result?;
        } else {
            return Err("channel was closed".into());
        }
        Ok(slice.get_mapped_range())
    }
}
