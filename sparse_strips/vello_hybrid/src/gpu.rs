// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The GPU parts of a hybrid CPU/GPU rendering engine.

#[cfg(feature = "perf_measurement")]
use crate::perf_measurement::PerfMeasurement;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupLayout, BlendState, Buffer, ColorTargetState, ColorWrites, Device,
    PipelineCompilationOptions, Queue, RenderPipeline, Surface, SurfaceConfiguration, Texture,
    TextureView, util::DeviceExt,
};
use winit::window::Window;

/// Represents the target for rendering - either a window or specific dimensions
pub enum RenderTarget {
    /// Render to a window
    Window(Arc<Window>),
    /// Render to a texture with specific dimensions
    Headless { width: u32, height: u32 },
}

impl RenderTarget {
    /// Get the dimensions of the render target
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            RenderTarget::Window(window) => {
                let size = window.inner_size();
                (size.width, size.height)
            }
            RenderTarget::Headless { width, height } => (*width, *height),
        }
    }
}

pub struct Renderer {
    pub device: Device,
    pub queue: Queue,
    #[allow(dead_code)]
    pub surface: Option<Surface<'static>>,
    pub surface_config: SurfaceConfiguration,
    pub render_bind_group_layout: BindGroupLayout,
    pub render_pipeline: RenderPipeline,
    pub render_bind_group: BindGroup,
    pub config_buf: Buffer,
    pub strips_buffer: Buffer,
    pub alphas_texture: Texture,
    pub alphas_texture_view: TextureView,

    // Performance measurement with timestamp queries
    #[cfg(feature = "perf_measurement")]
    pub perf_measurement: PerfMeasurement,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub strip_height: u32,
    pub alpha_texture_width: u32,
}

pub struct RenderData {
    pub strips: Vec<GpuStrip>,
    pub alphas: Vec<u32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct GpuStrip {
    pub x: u16,
    pub y: u16,
    pub width: u16,
    pub dense_width: u16,
    pub col: u32,
    pub rgba: u32,
}

impl GpuStrip {
    pub fn vertex_attributes() -> [wgpu::VertexAttribute; 4] {
        wgpu::vertex_attr_array![
            0 => Uint32,
            1 => Uint32,
            2 => Uint32,
            3 => Uint32,
        ]
    }
}

impl Renderer {
    /// Creates a new renderer
    ///
    /// The target parameter determines if we render to a window or headless
    pub async fn new(target: RenderTarget, render_data: &RenderData) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            backend_options: wgpu::BackendOptions::default(),
        });

        // Get dimensions and possibly create a surface
        let (dimensions, surface) = match &target {
            RenderTarget::Window(window) => {
                let surface = instance.create_surface(window.clone()).unwrap();
                let dimensions = target.dimensions();
                (dimensions, Some(surface))
            }
            RenderTarget::Headless { .. } => {
                let dimensions = target.dimensions();
                (dimensions, None)
            }
        };

        // Get adapter with or without surface
        let adapter = if let Some(surface) = &surface {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: Default::default(),
                    force_fallback_adapter: false,
                    compatible_surface: Some(surface),
                })
                .await
                .expect("Failed to find an appropriate adapter")
        } else {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: Default::default(),
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .expect("Failed to find an appropriate adapter")
        };

        #[cfg(feature = "perf_measurement")]
        let required_features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        #[cfg(not(feature = "perf_measurement"))]
        let required_features = wgpu::Features::empty();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Vello Hybrid"),
                    required_features,
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let (width, height) = dimensions;
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // Configure surface if it exists
        if let Some(surface) = &surface {
            surface.configure(&device, &surface_config);
        }

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render.wgsl").into()),
        });
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuStrip>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &GpuStrip::vertex_attributes(),
                }],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        let config = Config {
            width,
            height,
            strip_height: 4,
            alpha_texture_width: max_texture_dimension_2d,
        };
        let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create a buffer for the strip instances
        let strips_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Strips Buffer"),
            contents: bytemuck::cast_slice(&render_data.strips),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // Create texture for alpha values
        // We pack the alpha values into a 2D texture with max width imension
        let alpha_len = render_data.alphas.len();
        let alpha_texture_width = config.alpha_texture_width;
        let alpha_texture_height =
            ((alpha_len as u32) + alpha_texture_width - 1) / alpha_texture_width;

        // Ensure dimensions don't exceed WebGL2 limits
        assert!(
            alpha_texture_height <= max_texture_dimension_2d,
            "Alpha texture height exceeds WebGL2 limit"
        );

        let alphas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Alpha Texture"),
            size: wgpu::Extent3d {
                width: alpha_texture_width,
                height: alpha_texture_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let alphas_texture_view =
            alphas_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Prepare alpha data for the texture
        let mut alpha_data = vec![0u32; (alpha_texture_width * alpha_texture_height) as usize];

        // Fill the buffer with alpha data
        for (idx, alpha) in render_data.alphas.iter().enumerate() {
            if idx < alpha_data.len() {
                alpha_data[idx] = *alpha;
            }
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &alphas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&alpha_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 4 bytes per u32
                bytes_per_row: Some(alpha_texture_width * 4),
                rows_per_image: Some(alpha_texture_height),
            },
            wgpu::Extent3d {
                width: alpha_texture_width,
                height: alpha_texture_height,
                depth_or_array_layers: 1,
            },
        );

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&alphas_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: config_buf.as_entire_binding(),
                },
            ],
        });

        #[cfg(feature = "perf_measurement")]
        let perf_measurement = PerfMeasurement::new(&device);

        Self {
            device,
            queue,
            surface,
            surface_config,
            render_bind_group_layout,
            render_pipeline,
            render_bind_group,
            config_buf,
            strips_buffer,
            alphas_texture,
            alphas_texture_view,
            #[cfg(feature = "perf_measurement")]
            perf_measurement,
        }
    }

    pub fn prepare(&self, render_data: &RenderData) {
        // TODO: update buffers
    }

    pub fn render_to_surface(&self, render_data: &RenderData) {
        let Some(surface) = &self.surface else {
            // Cannot render to surface in headless mode
            return;
        };

        let frame = surface
            .get_current_texture()
            .expect("Failed to get current texture");

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Record start timestamp if feature is enabled
        #[cfg(feature = "perf_measurement")]
        self.perf_measurement.write_timestamp(&mut encoder, 0);

        {
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.strips_buffer.slice(..));
            let strips_to_draw = render_data.strips.len();
            render_pass.draw(0..4, 0..strips_to_draw as u32);
        }

        #[cfg(feature = "perf_measurement")]
        self.perf_measurement.write_timestamp(&mut encoder, 1);

        #[cfg(feature = "perf_measurement")]
        self.perf_measurement
            .resolve_timestamp_queries(&mut encoder, &self.device);
        self.queue.submit(std::iter::once(encoder.finish()));

        #[cfg(feature = "perf_measurement")]
        self.perf_measurement
            .map_and_read_timestamp_buffer(&self.device, &self.queue);

        frame.present();
    }

    pub fn render_to_texture(&self, render_data: &RenderData, width: u32, height: u32) -> Vec<u8> {
        // Create a texture to render to
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render to Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create a buffer to copy the texture data to
        let bytes_per_row = (width * 4).next_multiple_of(256);
        let texture_copy_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Copy to Buffer"),
            size: bytes_per_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Record start timestamp if feature is enabled
        #[cfg(feature = "perf_measurement")]
        self.perf_measurement.write_timestamp(&mut encoder, 0);

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Redner to Texture Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.render_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.strips_buffer.slice(..));
            let strips_to_draw = render_data.strips.len();
            render_pass.draw(0..4, 0..strips_to_draw as u32);
        }

        #[cfg(feature = "perf_measurement")]
        self.perf_measurement.write_timestamp(&mut encoder, 1);

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &texture_copy_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit([encoder.finish()]);

        // Map the buffer to read the data
        texture_copy_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_err() {
                    panic!("Failed to map texture for reading");
                }
            });
        self.device.poll(wgpu::Maintain::Wait);

        // Copy the data into a Vec<u8>
        let mut img_data = Vec::with_capacity((width * height * 4) as usize);
        for row in texture_copy_buffer
            .slice(..)
            .get_mapped_range()
            .chunks_exact(bytes_per_row as usize)
        {
            img_data.extend_from_slice(&row[0..width as usize * 4]);
        }

        texture_copy_buffer.unmap();

        img_data
    }
}
