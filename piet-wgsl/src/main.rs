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

use std::{fs::File, io::BufWriter};
use engine::{BufProxy, DownloadBufUsage, Downloads, Engine};

use render::render;
use test_scene::dump_scene_info;
use wgpu::{Device, Queue};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

mod engine;
mod render;
mod shaders;
mod test_scene;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 1024;

async fn render_demo_scene(
    device: &Device,
    queue: &Queue,
    engine: &mut Engine,
    usage: DownloadBufUsage,
) -> Result<(Downloads, BufProxy), Box<dyn std::error::Error>> {
    let shaders = shaders::init_shaders(device, engine)?;
    let scene = test_scene::gen_test_scene();
    dump_scene_info(&scene);
    let (recording, buf) = render(&scene, &shaders, usage);
    let downloads = engine.run_recording(&device, &queue, &recording)?;
    Ok((downloads, buf))
}

// ------ Native ------
async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let features = adapter.features();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: features & wgpu::Features::TIMESTAMP_QUERY,
                limits: Default::default(),
            },
            None,
        )
        .await?;
    let mut engine = Engine::new();
    do_render(&device, &queue, &mut engine).await?;

    Ok(())
}

async fn do_render(
    device: &Device,
    queue: &Queue,
    engine: &mut Engine,
) -> Result<(), Box<dyn std::error::Error>> {
    let (downloads, buf) = render_demo_scene(device, queue, engine, DownloadBufUsage::MapRead).await?;
    let mapped = downloads.map();
    device.poll(wgpu::Maintain::Wait);
    let buf = mapped.get_mapped(buf).await?;

    let file = File::create("image.png")?;
    let w = BufWriter::new(file);
    let encoder = png::Encoder::new(w, WIDTH, HEIGHT);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&buf)?;
    Ok(())
}

// ------ WASM ------
struct BufferDimensions {
    width: u32,
    height: u32,
    unpadded_bytes_per_row: u32,
    padded_bytes_per_row: u32,
}

impl BufferDimensions {
    fn new(width: u32, height: u32) -> Self {
        let bytes_per_pixel = std::mem::size_of::<u32>() as u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let alignment_mask = align - 1;
        let padded_bytes_per_row = (unpadded_bytes_per_row + alignment_mask) & !alignment_mask;
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }

    fn get_extent(&self) -> wgpu::Extent3d {
        let bytes_per_pixel = std::mem::size_of::<u32>() as u32;
        wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        }
    }
}

async fn run_wasm(event_loop: EventLoop<()>, window: Window) {
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let mut engine = Engine::new();

    let (downloads, buf) = render_demo_scene(&device, &queue, &mut engine, DownloadBufUsage::BlitSrc).await.unwrap();
    {
        device.poll(wgpu::Maintain::Wait);
    }

    // TODO: the output buffer dimensions assigned internally don't quite align to a canvas size of
    // 1024x1024 with pixel format bgra8unorm. We compensate here by shrinking the dimensions for
    // the swapchain texture. Fix this by conforming to given texture dimensions in the engine?
    let buffer_dimensions = BufferDimensions::new(WIDTH / 2, HEIGHT / 2);
    let swapchain_format = surface.get_supported_formats(&adapter)[0];
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::COPY_DST,
        format: swapchain_format,
        width: buffer_dimensions.width,
        height: buffer_dimensions.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface.get_supported_alpha_modes(&adapter)[0],
    };
    surface.configure(&device, &config);

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        let _ = (&instance, &adapter);
        let buffer = downloads.get_buffer(&buf);

        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Reconfigure the surface with the new size
                config.width = buffer_dimensions.width;
                config.height = buffer_dimensions.height;
                surface.configure(&device, &config);
                // On macos the window needs to be redrawn manually after resizing
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    encoder.copy_buffer_to_texture(
                        wgpu::ImageCopyBuffer {
                            buffer,
                            layout: wgpu::ImageDataLayout {
                                offset: 0,
                                bytes_per_row: Some(
                                    std::num::NonZeroU32::new(
                                        buffer_dimensions.padded_bytes_per_row,
                                    ).unwrap()
                                ),
                                rows_per_image: None,
                            }
                        },
                        frame.texture.as_image_copy(),
                        buffer_dimensions.get_extent(),
                    );
                }

                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        pollster::block_on(run()).unwrap();
    }
    #[cfg(target_arch = "wasm32")]
    {
        let event_loop = EventLoop::new();
        let window = winit::window::Window::new(&event_loop).unwrap();

        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;

        // On wasm, append the canvas to the document body
        let canvas = window.canvas();
        canvas.set_width(WIDTH);
        canvas.set_height(HEIGHT);
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(canvas))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run_wasm(event_loop, window));
    }
}
