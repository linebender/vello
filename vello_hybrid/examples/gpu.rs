// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Example program for CPU/GPU hybrid rendering.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vello_api::peniko::{
    color::palette,
    kurbo::{BezPath, Stroke},
};
use vello_hybrid::{GpuRenderCtx, GpuSession};
use wgpu::util::DeviceExt;

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Config {
    width: u32,
    height: u32,
    strip_height: u32,
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let window = Arc::new(window);
    let window_clone = window.clone();
    let instance = wgpu::Instance::new(&Default::default());
    let surface = instance.create_surface(&window).unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: Default::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("error finding adapter");

    let (device, queue) = adapter
        .request_device(&Default::default(), None)
        .await
        .expect("error creating device");
    let size = window.inner_size();
    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let format = swapchain_capabilities.formats[0];
    let sc = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: swapchain_capabilities.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &sc);

    let session = GpuSession::new(&device, format);

    let mut render_ctx = GpuRenderCtx::new(size.width as usize, size.height as usize);
    draw_simple_scene(&mut render_ctx);
    let bufs = render_ctx.harvest();

    let config = Config {
        width: size.width,
        height: size.height,
        strip_height: 4,
    };

    let config_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("config"),
        contents: bytemuck::bytes_of(&config),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let strip_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("strip"),
        contents: bytemuck::cast_slice(&bufs.strips),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let alpha_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("alpha"),
        contents: bytemuck::cast_slice(&bufs.alphas),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &session.render_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: alpha_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: config_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: strip_buf.as_entire_binding(),
            },
        ],
    });

    event_loop
        .run(move |event, target| {
            if let Event::WindowEvent {
                window_id: _,
                event: window_event,
            } = event
            {
                match window_event {
                    WindowEvent::RedrawRequested => {
                        let frame = surface
                            .get_current_texture()
                            .expect("error getting texture from swap chain");

                        let mut encoder = device.create_command_encoder(&Default::default());
                        {
                            let view = frame
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            let mut rpass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            rpass.set_pipeline(&session.render_pipeline);
                            rpass.set_bind_group(0, &render_bind_group, &[]);
                            let n_strips = bufs.strips.len().try_into().expect("too many strips");
                            rpass.draw(0..4, 0..n_strips);
                        }
                        queue.submit(Some(encoder.finish()));
                        frame.present();
                        window_clone.request_redraw();
                    }
                    WindowEvent::CloseRequested => {
                        target.exit();
                    }
                    _ => (),
                }
            }
        })
        .unwrap();
}

fn draw_simple_scene(ctx: &mut GpuRenderCtx) {
    let mut path = BezPath::new();
    path.move_to((10.0, 10.0));
    path.line_to((180.0, 20.0));
    path.line_to((30.0, 40.0));
    path.close_path();
    let piet_path = path.into();
    ctx.fill(&piet_path, palette::css::REBECCA_PURPLE.into());
    let stroke = Stroke::new(5.0);
    ctx.stroke(&piet_path, &stroke, palette::css::DARK_BLUE.into());
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Window::new(&event_loop).unwrap();
    window.set_resizable(false);
    pollster::block_on(run(event_loop, window));
}
