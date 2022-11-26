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

mod engine;
mod ramp;
mod render;
mod shaders;

pub mod util;

use engine::{Engine, ExternalResource};
use shaders::FullShaders;

use piet_scene::Scene;
use wgpu::{Device, Queue, SurfaceTexture, Texture, TextureFormat, TextureView};

/// Catch-all error type.
pub type Error = Box<dyn std::error::Error>;

/// Specialization of `Result` for our catch-all error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Renders a scene into a texture or surface.
pub struct Renderer {
    engine: Engine,
    shaders: FullShaders,
    blit: BlitPipeline,
    target: Option<TargetTexture>,
}

impl Renderer {
    /// Creates a new renderer for the specified device.
    pub fn new(device: &Device) -> Result<Self> {
        let mut engine = Engine::new();
        let shaders = shaders::full_shaders(device, &mut engine)?;
        let blit = BlitPipeline::new(device, TextureFormat::Bgra8Unorm);
        Ok(Self {
            engine,
            shaders,
            blit,
            target: None,
        })
    }

    /// Renders a scene to the target texture.
    ///
    /// The texture is assumed to be of the specified dimensions and have been created with
    /// the [wgpu::TextureFormat::Rgba8Unorm] format and the [wgpu::TextureUsages::STORAGE_BINDING]
    /// flag set.
    pub fn render_to_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        texture: &TextureView,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let (recording, target) = render::render_full(&scene, &self.shaders, width, height);
        let external_resources = [ExternalResource::Image(
            *target.as_image().unwrap(),
            texture,
        )];
        let _ = self
            .engine
            .run_recording(device, queue, &recording, &external_resources)?;
        Ok(())
    }

    /// Renders a scene to the target surface.
    ///
    /// This renders to an intermediate texture and then runs a render pass to blit to the
    /// specified surface texture.
    ///
    /// The surface is assumed to be of the specified dimensions and have been created with the
    /// [wgpu::TextureFormat::Bgra8Unorm] format.
    pub fn render_to_surface(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        surface: &SurfaceTexture,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let mut target = self
            .target
            .take()
            .unwrap_or_else(|| TargetTexture::new(device, width, height));
        // TODO: implement clever resizing semantics here to avoid thrashing the memory allocator
        // during resize, specifically on metal.
        if target.width != width || target.height != height {
            target = TargetTexture::new(device, width, height);
        }
        self.render_to_texture(device, queue, scene, &target.view, width, height)?;
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let surface_view = surface
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.blit.bind_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target.view),
                }],
            });
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::default()),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.blit.pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        self.target = Some(target);
        Ok(())
    }
}

struct TargetTexture {
    texture: Texture,
    view: TextureView,
    width: u32,
    height: u32,
}

impl TargetTexture {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            format: wgpu::TextureFormat::Rgba8Unorm,
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            width,
            height,
        }
    }
}

struct BlitPipeline {
    bind_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl BlitPipeline {
    fn new(device: &Device, format: TextureFormat) -> Self {
        const SHADERS: &str = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) ix: u32) -> @builtin(position) vec4<f32> {
                // Generate a full screen quad in NDCs
                var vertex = vec2<f32>(-1.0, 1.0);
                switch ix {
                    case 1u: {
                        vertex = vec2<f32>(-1.0, -1.0);
                    }
                    case 2u, 4u: {
                        vertex = vec2<f32>(1.0, -1.0);
                    }
                    case 5u: {
                        vertex = vec2<f32>(1.0, 1.0);
                    }
                    default: {}
                }
                return vec4<f32>(vertex, 0.0, 1.0);
            }
            
            @group(0) @binding(0)
            var fine_output: texture_2d<f32>;
            
            @fragment
            fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
                return textureLoad(fine_output, vec2<i32>(pos.xy), 0);
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit shaders"),
            source: wgpu::ShaderSource::Wgsl(SHADERS.into()),
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                visibility: wgpu::ShaderStages::FRAGMENT,
                binding: 0,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });
        Self {
            bind_layout,
            pipeline,
        }
    }
}
