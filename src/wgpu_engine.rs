// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::{
    borrow::Cow,
    cell::RefCell,
    collections::{hash_map::Entry, HashMap, HashSet},
};

use wgpu::{
    BindGroup, BindGroupLayout, Buffer, BufferUsages, CommandEncoder, CommandEncoderDescriptor,
    ComputePipeline, Device, Queue, Texture, TextureAspect, TextureUsages, TextureView,
    TextureViewDimension,
};

use crate::{
    cpu_dispatch::CpuBinding, workflow::BindType, BufferProxy, Command, Error, ImageProxy,
    ResourceId, ResourceProxy, ShaderId, Workflow,
};

#[cfg(not(target_arch = "wasm32"))]
struct UninitialisedShader {
    wgsl: Cow<'static, str>,
    label: &'static str,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
    shader_id: ShaderId,
}

#[derive(Default)]
pub struct WgpuEngine {
    shaders: Vec<Shader>,
    pool: ResourcePool,
    bind_map: BindMap,
    downloads: HashMap<ResourceId, Buffer>,
    #[cfg(not(target_arch = "wasm32"))]
    shaders_to_initialise: Option<Vec<UninitialisedShader>>,
    pub(crate) use_cpu: bool,
}

struct WgpuShader {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

pub enum CpuShaderType {
    Present(fn(u32, &[CpuBinding])),
    Missing,
    Skipped,
}

struct CpuShader {
    shader: fn(u32, &[CpuBinding]),
}

enum ShaderKind<'a> {
    Wgpu(&'a WgpuShader),
    Cpu(&'a CpuShader),
}

struct Shader {
    #[allow(dead_code)]
    label: &'static str,
    wgpu: Option<WgpuShader>,
    cpu: Option<CpuShader>,
}

impl Shader {
    fn select(&self) -> ShaderKind {
        if let Some(cpu) = self.cpu.as_ref() {
            ShaderKind::Cpu(cpu)
        } else if let Some(wgpu) = self.wgpu.as_ref() {
            ShaderKind::Wgpu(wgpu)
        } else {
            panic!("no available shader for {}", self.label)
        }
    }
}

pub enum ExternalResource<'a> {
    #[allow(unused)]
    Buffer(BufferProxy, &'a Buffer),
    Image(ImageProxy, &'a TextureView),
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
    buf_map: HashMap<ResourceId, BindMapBuffer>,
    image_map: HashMap<ResourceId, (Texture, TextureView)>,
    pending_clears: HashSet<ResourceId>,
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
    bufs: HashMap<ResourceId, TransientBuf<'a>>,
    // TODO: create transient image type
    images: HashMap<ResourceId, &'a TextureView>,
}

enum TransientBuf<'a> {
    Cpu(&'a [u8]),
    Gpu(&'a Buffer),
}

impl WgpuEngine {
    pub fn new(use_cpu: bool) -> WgpuEngine {
        Self {
            use_cpu,
            ..Default::default()
        }
    }

    /// Enable creating any remaining shaders in parallel
    #[cfg(not(target_arch = "wasm32"))]
    pub fn use_parallel_initialisation(&mut self) {
        if self.shaders_to_initialise.is_some() {
            return;
        }
        self.shaders_to_initialise = Some(Vec::new());
    }

    #[cfg(not(target_arch = "wasm32"))]
    /// Initialise (in parallel) any shaders which are yet to be created
    pub fn build_shaders_if_needed(
        &mut self,
        device: &Device,
        num_threads: Option<std::num::NonZeroUsize>,
    ) {
        use std::num::NonZeroUsize;

        if let Some(mut new_shaders) = self.shaders_to_initialise.take() {
            let num_threads = num_threads
                .map(NonZeroUsize::get)
                .unwrap_or_else(|| {
                    // Fallback onto a heuristic. This tries to not to use all threads.
                    // We keep the main thread blocked and not doing much whilst this is running,
                    // so we broadly leave two cores unused at the point of maximum parallelism
                    // (This choice is arbitrary, and could be tuned, although a 'proper' threadpool
                    // should probably be used instead)
                    std::thread::available_parallelism().map_or(2, |it| it.get().max(4) - 2)
                })
                .min(new_shaders.len());
            eprintln!("Initialising in parallel using {num_threads} threads");
            let remainder = new_shaders.split_off(num_threads);
            let (tx, rx) = std::sync::mpsc::channel::<(ShaderId, WgpuShader)>();

            // We expect each initialisation to take much longer than acquiring a lock, so we just use a mutex for our work queue
            let work_queue = std::sync::Mutex::new(remainder.into_iter());
            let work_queue = &work_queue;
            std::thread::scope(|scope| {
                let tx = tx;
                new_shaders
                    .into_iter()
                    .map(|it| {
                        let tx = tx.clone();
                        std::thread::Builder::new()
                            .name("Vello shader initialisation worker thread".into())
                            .spawn_scoped(scope, move || {
                                let shader = Self::create_compute_pipeline(
                                    device, it.label, it.wgsl, it.entries,
                                );
                                // We know the rx can only be closed if all the tx references are dropped
                                tx.send((it.shader_id, shader)).unwrap();
                                while let Ok(mut guard) = work_queue.lock() {
                                    if let Some(value) = guard.next() {
                                        drop(guard);
                                        let shader = Self::create_compute_pipeline(
                                            device,
                                            value.label,
                                            value.wgsl,
                                            value.entries,
                                        );
                                        tx.send((value.shader_id, shader)).unwrap();
                                    } else {
                                        break;
                                    }
                                }
                                // Another thread panicked or we finished.
                                // If another thread panicked, we ignore that here and finish our processing
                                drop(tx);
                            })
                            .expect("failed to spawn thread");
                    })
                    .for_each(drop);
                // Drop the initial sender, to mean that there will be no more senders if and only if all other threads have finished
                drop(tx);

                while let Ok((id, value)) = rx.recv() {
                    self.shaders[id.0].wgpu = Some(value);
                }
            });
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
        cpu_shader: CpuShaderType,
    ) -> Result<ShaderId, Error> {
        let mut add = |shader| {
            let id = self.shaders.len();
            self.shaders.push(shader);
            Ok(ShaderId(id))
        };

        if self.use_cpu {
            match cpu_shader {
                CpuShaderType::Present(shader) => {
                    return add(Shader {
                        wgpu: None,
                        cpu: Some(CpuShader { shader }),
                        label,
                    });
                }
                // This shader is unused in CPU mode, create a dummy shader
                CpuShaderType::Skipped => {
                    return add(Shader {
                        wgpu: None,
                        cpu: None,
                        label,
                    });
                }
                // Create a GPU shader as we don't have a CPU shader
                CpuShaderType::Missing => {}
            }
        }

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
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(uninit) = self.shaders_to_initialise.as_mut() {
            let id = add(Shader {
                label,
                wgpu: None,
                cpu: None,
            })?;
            uninit.push(UninitialisedShader {
                wgsl,
                label,
                entries,
                shader_id: id,
            });
            return Ok(id);
        }
        let wgpu = Self::create_compute_pipeline(device, label, wgsl, entries);
        add(Shader {
            wgpu: Some(wgpu),
            cpu: None,
            label,
        })
    }

    pub fn run_workflow(
        &mut self,
        device: &Device,
        queue: &Queue,
        recording: &Workflow,
        external_resources: &[ExternalResource],
        label: &'static str,
        #[cfg(feature = "wgpu-profiler")] profiler: &mut wgpu_profiler::GpuProfiler,
    ) -> Result<(), Error> {
        let mut free_bufs: HashSet<ResourceId> = Default::default();
        let mut free_images: HashSet<ResourceId> = Default::default();
        let mut transient_map = TransientBindMap::new(external_resources);

        let mut encoder =
            device.create_command_encoder(&CommandEncoderDescriptor { label: Some(label) });
        #[cfg(feature = "wgpu-profiler")]
        let query = profiler.begin_query(label, &mut encoder, device);
        for command in recording.commands() {
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
                        .block_copy_size(None)
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
                        .insert_image(image_proxy.id, texture, texture_view);
                }
                Command::WriteImage(proxy, [x, y, width, height], data) => {
                    if let Ok((texture, _)) = self.bind_map.get_or_create_image(*proxy, device) {
                        let format = proxy.format.to_wgpu();
                        let block_size = format
                            .block_copy_size(None)
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
                    match shader.select() {
                        ShaderKind::Cpu(cpu_shader) => {
                            // The current strategy is to run the CPU shader synchronously. This
                            // works because there is currently the added constraint that data
                            // can only flow from CPU to GPU, not the other way around. If and
                            // when we implement that, we will need to defer the execution. Of
                            // course, we will also need to wire up more async synchronization
                            // mechanisms, as the CPU dispatch can't run until the preceding
                            // command buffer submission completes (and, in WebGPU, the async
                            // mapping operations on the buffers completes).
                            let resources =
                                transient_map.create_cpu_resources(&mut self.bind_map, bindings);
                            (cpu_shader.shader)(wg_size.0, &resources);
                        }
                        ShaderKind::Wgpu(wgpu_shader) => {
                            let bind_group = transient_map.create_bind_group(
                                &mut self.bind_map,
                                &mut self.pool,
                                device,
                                queue,
                                &mut encoder,
                                &wgpu_shader.bind_group_layout,
                                bindings,
                            )?;
                            let mut cpass = encoder.begin_compute_pass(&Default::default());
                            #[cfg(feature = "wgpu-profiler")]
                            let query = profiler
                                .begin_query(shader.label, &mut cpass, device)
                                .with_parent(Some(&query));
                            cpass.set_pipeline(&wgpu_shader.pipeline);
                            cpass.set_bind_group(0, &bind_group, &[]);
                            cpass.dispatch_workgroups(wg_size.0, wg_size.1, wg_size.2);
                            #[cfg(feature = "wgpu-profiler")]
                            profiler.end_query(&mut cpass, query);
                        }
                    }
                }
                Command::DispatchIndirect(shader_id, proxy, offset, bindings) => {
                    let shader = &self.shaders[shader_id.0];
                    match shader.select() {
                        ShaderKind::Cpu(cpu_shader) => {
                            // Same consideration as above about running the CPU shader synchronously.
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
                            (cpu_shader.shader)(n_wg, &resources);
                        }
                        ShaderKind::Wgpu(wgpu_shader) => {
                            let bind_group = transient_map.create_bind_group(
                                &mut self.bind_map,
                                &mut self.pool,
                                device,
                                queue,
                                &mut encoder,
                                &wgpu_shader.bind_group_layout,
                                bindings,
                            )?;
                            transient_map.materialize_gpu_buf_for_indirect(
                                &mut self.bind_map,
                                &mut self.pool,
                                device,
                                queue,
                                proxy,
                            );
                            let mut cpass = encoder.begin_compute_pass(&Default::default());
                            #[cfg(feature = "wgpu-profiler")]
                            let query = profiler
                                .begin_query(shader.label, &mut cpass, device)
                                .with_parent(Some(&query));
                            cpass.set_pipeline(&wgpu_shader.pipeline);
                            cpass.set_bind_group(0, &bind_group, &[]);
                            let buf = self
                                .bind_map
                                .get_gpu_buf(proxy.id)
                                .ok_or("buffer for indirect dispatch not in map")?;
                            cpass.dispatch_workgroups_indirect(buf, *offset);
                            #[cfg(feature = "wgpu-profiler")]
                            profiler.end_query(&mut cpass, query);
                        }
                    }
                }
                Command::Download(proxy) => {
                    let src_buf = self
                        .bind_map
                        .get_gpu_buf(proxy.id)
                        .ok_or("buffer not in map")?;
                    let usage = BufferUsages::MAP_READ | BufferUsages::COPY_DST;
                    let buf = self.pool.get_buf(proxy.size, "download", usage, device);
                    encoder.copy_buffer_to_buffer(src_buf, 0, &buf, 0, proxy.size);
                    self.downloads.insert(proxy.id, buf);
                }
                Command::Clear(proxy, offset, size) => {
                    if let Some(buf) = self.bind_map.get_buf(*proxy) {
                        match &buf.buffer {
                            MaterializedBuffer::Gpu(b) => encoder.clear_buffer(b, *offset, *size),
                            MaterializedBuffer::Cpu(b) => {
                                let mut slice = &mut b.borrow_mut()[*offset as usize..];
                                if let Some(size) = size {
                                    slice = &mut slice[..*size as usize];
                                }
                                slice.fill(0);
                            }
                        }
                    } else {
                        self.bind_map.pending_clears.insert(proxy.id);
                    }
                }
                Command::FreeBuffer(proxy) => {
                    free_bufs.insert(proxy.id);
                }
                Command::FreeImage(proxy) => {
                    free_images.insert(proxy.id);
                }
            }
        }
        #[cfg(feature = "wgpu-profiler")]
        profiler.end_query(&mut encoder, query);
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

    pub fn get_download(&self, buf: BufferProxy) -> Option<&Buffer> {
        self.downloads.get(&buf.id)
    }

    pub fn free_download(&mut self, buf: BufferProxy) {
        self.downloads.remove(&buf.id);
    }

    fn create_compute_pipeline(
        device: &Device,
        label: &str,
        wgsl: Cow<'_, str>,
        entries: Vec<wgpu::BindGroupLayoutEntry>,
    ) -> WgpuShader {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl),
        });
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
        WgpuShader {
            pipeline,
            bind_group_layout,
        }
    }
}

impl BindMap {
    fn insert_buf(&mut self, proxy: &BufferProxy, buffer: Buffer) {
        self.buf_map.insert(
            proxy.id,
            BindMapBuffer {
                buffer: MaterializedBuffer::Gpu(buffer),
                label: proxy.name,
            },
        );
    }

    /// Get a buffer, only if it's on GPU.
    fn get_gpu_buf(&self, id: ResourceId) -> Option<&Buffer> {
        self.buf_map.get(&id).and_then(|b| match &b.buffer {
            MaterializedBuffer::Gpu(b) => Some(b),
            _ => None,
        })
    }

    /// Get a CPU buffer.
    ///
    /// Panics if buffer is not present or is on GPU.
    fn get_cpu_buf(&self, id: ResourceId) -> CpuBinding {
        match &self.buf_map[&id].buffer {
            MaterializedBuffer::Cpu(b) => CpuBinding::BufferRW(b),
            _ => panic!("getting cpu buffer, but it's on gpu"),
        }
    }

    fn materialize_cpu_buf(&mut self, buf: &BufferProxy) {
        self.buf_map.entry(buf.id).or_insert_with(|| {
            let buffer = MaterializedBuffer::Cpu(RefCell::new(vec![0; buf.size as usize]));
            BindMapBuffer {
                buffer,
                // TODO: do we need to cfg this?
                label: buf.name,
            }
        });
    }

    fn insert_image(&mut self, id: ResourceId, image: Texture, image_view: TextureView) {
        self.image_map.insert(id, (image, image_view));
    }

    fn get_buf(&mut self, proxy: BufferProxy) -> Option<&BindMapBuffer> {
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
        #[allow(unused)] name: &'static str,
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
    // Upload a buffer from CPU to GPU if needed.
    //
    // Note data flow is one way only, from CPU to GPU. Once this method is
    // called, the buffer is no longer materialized on CPU, and cannot be
    // accessed from a CPU shader.
    fn upload_if_needed(
        &mut self,
        proxy: &BufferProxy,
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
                ExternalResource::Buffer(proxy, gpu_buf) => {
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
        buf: &BufferProxy,
    ) {
        if !self.bufs.contains_key(&buf.id) {
            if let Some(b) = bind_map.buf_map.get_mut(&buf.id) {
                b.upload_if_needed(buf, device, queue, pool);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
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
                ResourceProxy::Buffer(proxy) => {
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
                            o.get_mut().upload_if_needed(proxy, device, queue, pool);
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
                ResourceProxy::Buffer(proxy) => {
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
                ResourceProxy::Buffer(buf) => match self.bufs.get(&buf.id) {
                    Some(TransientBuf::Cpu(_)) => (),
                    Some(TransientBuf::Gpu(_)) => panic!("buffer was already materialized on GPU"),
                    _ => bind_map.materialize_cpu_buf(buf),
                },
                ResourceProxy::Image(_) => todo!(),
            };
        }
        // Second pass takes immutable references
        bindings
            .iter()
            .map(|resource| match resource {
                ResourceProxy::Buffer(buf) => match self.bufs.get(&buf.id) {
                    Some(TransientBuf::Cpu(b)) => CpuBinding::Buffer(b),
                    _ => bind_map.get_cpu_buf(buf.id),
                },
                ResourceProxy::Image(_) => todo!(),
            })
            .collect()
    }
}
