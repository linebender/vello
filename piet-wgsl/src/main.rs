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

//! A simple application to run a compute shader.

use engine::{Engine, Error, ExternalResource};

use piet_scene::{Scene, SceneBuilder};
use wgpu::{Device, Instance, Limits, Queue, Surface, SurfaceConfiguration};
use winit::window::Window;

mod debug;
mod engine;
mod pico_svg;
mod ramp;
mod render;
mod shaders;
mod simple_text;
mod test_scene;

use pico_svg::PicoSvg;
use simple_text::SimpleText;

pub struct Dimensions {
    width: u32,
    height: u32,
}

pub struct WgpuState {
    pub instance: Instance,
    pub device: Device,
    pub queue: Queue,
    pub surface: Option<Surface>,
    pub surface_config: SurfaceConfiguration,
}

impl WgpuState {
    pub async fn new(window: Option<&Window>) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = Instance::new(wgpu::Backends::PRIMARY);
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let features = adapter.features();
        let mut limits = Limits::default();
        limits.max_storage_buffers_per_shader_stage = 16;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: features
                        & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::CLEAR_TEXTURE),
                    limits,
                },
                None,
            )
            .await?;
        let (surface, surface_config) = if let Some(window) = window {
            let surface = unsafe { instance.create_surface(window) };
            let size = window.inner_size();
            // let format = surface.get_supported_formats(&adapter)[0];
            let format = wgpu::TextureFormat::Bgra8Unorm;
            println!("surface: {:?} {:?}", size, format);
            let surface_config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
            };
            surface.configure(&device, &surface_config);
            (Some(surface), surface_config)
        } else {
            let surface_config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::empty(),
                format: wgpu::TextureFormat::Bgra8Unorm,
                width: 0,
                height: 0,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
            };
            (None, surface_config)
        };
        Ok(Self {
            instance,
            device,
            queue,
            surface,
            surface_config,
        })
    }

    pub fn create_target_texture(&self) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            format: wgpu::TextureFormat::Rgba8Unorm,
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}

async fn run_interactive() -> Result<(), Error> {
    use winit::{
        dpi::LogicalSize,
        event::*,
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
    };
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(LogicalSize::new(1044, 800))
        .with_resizable(true)
        .build(&event_loop)
        .unwrap();
    let mut state = WgpuState::new(Some(&window)).await?;
    let mut engine = Engine::new();
    let full_shaders = shaders::full_shaders(&state.device, &mut engine)?;
    let (blit_layout, blit_pipeline) = create_blit_pipeline(&state);
    let mut simple_text = SimpleText::new();
    let mut current_frame = 0usize;
    let mut scene_ix = 0usize;
    let (mut _target_texture, mut target_view) = state.create_target_texture();
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::KeyboardInput { input, .. } => {
                if input.state == ElementState::Pressed {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::Left) => scene_ix = scene_ix.saturating_sub(1),
                        Some(VirtualKeyCode::Right) => scene_ix = scene_ix.saturating_add(1),
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                state.surface_config.width = size.width;
                state.surface_config.height = size.height;
                state
                    .surface
                    .as_ref()
                    .unwrap()
                    .configure(&state.device, &state.surface_config);
                let (t, v) = state.create_target_texture();
                _target_texture = t;
                target_view = v;
                window.request_redraw();
            }
            _ => {}
        },
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            current_frame += 1;
            let surface_texture = state
                .surface
                .as_ref()
                .unwrap()
                .get_current_texture()
                .unwrap();
            let dimensions = Dimensions {
                width: state.surface_config.width,
                height: state.surface_config.height,
            };
            let mut scene = Scene::default();
            let mut builder = SceneBuilder::for_scene(&mut scene);
            const N_SCENES: usize = 6;
            match scene_ix % N_SCENES {
                0 => test_scene::render_anim_frame(&mut builder, &mut simple_text, current_frame),
                1 => test_scene::render_blend_grid(&mut builder),
                2 => test_scene::render_tiger(&mut builder, false),
                3 => test_scene::render_brush_transform(&mut builder, current_frame),
                4 => test_scene::render_funky_paths(&mut builder),
                _ => test_scene::render_scene(&mut builder),
            }
            builder.finish();
            let (recording, target) = render::render_full(&scene, &full_shaders, &dimensions);
            let external_resources = [ExternalResource::Image(
                *target.as_image().unwrap(),
                &target_view,
            )];
            let _ = engine
                .run_recording(&state.device, &state.queue, &recording, &external_resources)
                .unwrap();
            let mut encoder = state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let surface_view = surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &blit_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&target_view),
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
                render_pass.set_pipeline(&blit_pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.draw(0..6, 0..1);
            }
            state.queue.submit(Some(encoder.finish()));
            surface_texture.present();
        }
        _ => {}
    });
}

fn main() {
    pollster::block_on(run_interactive()).unwrap();
}

// Fit this into the recording code somehow?
fn create_blit_pipeline(state: &WgpuState) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
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

    let shader = state
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit shaders"),
            source: wgpu::ShaderSource::Wgsl(SHADERS.into()),
        });
    let bind_group_layout =
        state
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    let pipeline_layout = state
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    let pipeline = state
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    format: state.surface_config.format,
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
    (bind_group_layout, pipeline)
}
