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

pub struct Renderer {
    pub device: Device,
    pub queue: Queue,
    pub surface: Surface<'static>,
    pub surface_config: SurfaceConfiguration,
    pub render_bind_group_layout: BindGroupLayout,
    pub render_pipeline: RenderPipeline,
    pub render_bind_group: BindGroup,
    pub config_buf: Buffer,
    pub strips_texture: Texture,
    pub strips_texture_view: TextureView,
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
    // Add parameters for strip texture layout
    pub strips_per_row: u32,
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

impl Renderer {
    pub async fn new(window: Arc<Window>, render_data: &RenderData) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            backend_options: wgpu::BackendOptions::default(),
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: Default::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

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
        let size = window.inner_size();
        let format = wgpu::TextureFormat::Bgra8Unorm;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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
                buffers: &[],
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
        let strips_per_row = max_texture_dimension_2d / 4;
        let config = Config {
            width: size.width,
            height: size.height,
            strip_height: 4,
            strips_per_row,
            alpha_texture_width: max_texture_dimension_2d,
        };
        let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create textures for strips data with 2D layout
        // Compute how many strips we can fit in each row
        let strips_len = render_data.strips.len();
        // 4 values per strip
        let strips_texture_width = 4 * strips_per_row;
        let strips_texture_height = (strips_len as u32 + strips_per_row - 1) / strips_per_row;

        assert!(
            strips_texture_width <= max_texture_dimension_2d,
            "Strips texture width exceeds WebGL2 limit"
        );
        assert!(
            strips_texture_height <= max_texture_dimension_2d,
            "Strips texture height exceeds WebGL2 limit"
        );

        let strips_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Strips Texture"),
            size: wgpu::Extent3d {
                width: strips_texture_width,
                height: strips_texture_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let strips_texture_view =
            strips_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create texture for alpha values
        // We pack the alpha values into a 2D texture with a reasonable width
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

        // Create a buffer to hold all strip data in the correct layout
        let mut strips_data = vec![0u32; (strips_texture_width * strips_texture_height) as usize];

        // Fill the buffer with strip data
        for (i, strip) in render_data.strips.iter().enumerate() {
            let i = i as u32;
            let row = i / strips_per_row;
            let col = i % strips_per_row;
            let base_idx = (row * strips_texture_width + col * 4) as usize;

            // Avoid out-of-bounds
            if base_idx + 3 < strips_data.len() {
                let xy = ((strip.y as u32) << 16) | (strip.x as u32);
                let widths = ((strip.dense_width as u32) << 16) | (strip.width as u32);

                strips_data[base_idx] = xy;
                strips_data[base_idx + 1] = widths;
                strips_data[base_idx + 2] = strip.col;
                strips_data[base_idx + 3] = strip.rgba;
            }
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &strips_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&strips_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(strips_texture_width * 4), // 4 bytes per u32
                rows_per_image: Some(strips_texture_height),
            },
            wgpu::Extent3d {
                width: strips_texture_width,
                height: strips_texture_height,
                depth_or_array_layers: 1,
            },
        );

        // Create a buffer for alpha data in the correct layout
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&strips_texture_view),
                },
            ],
        });

        // Performance measurement initialization
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
            strips_texture,
            strips_texture_view,
            alphas_texture,
            alphas_texture_view,
            #[cfg(feature = "perf_measurement")]
            perf_measurement,
        }
    }

    pub fn prepare(&self, render_data: &RenderData) {
        // TODO: update buffers
    }

    pub fn render(&self, render_data: &RenderData) {
        let frame = self
            .surface
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
                label: None,
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

            // Determine how many strips we can draw
            let max_strips_in_texture =
                (self.strips_texture.size().width / 4) * self.strips_texture.size().height;
            let strips_len = render_data.strips.len();
            let strips_to_draw = strips_len.min(max_strips_in_texture as usize);

            assert!(
                strips_len <= max_strips_in_texture as usize,
                "Available strips to draw exceeds max strips in texture"
            );

            // The same quad is rendered for each instance, and the shader handles positioning and sizing
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
}
