// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Example program for CPU/GPU hybrid rendering.

use std::sync::Arc;

use kurbo::Affine;
use peniko::{
    color::palette,
    kurbo::{BezPath, Stroke},
};
use vello_hybrid::{Config, RenderContext, Renderer};

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Window::new(&event_loop).unwrap();
    window.set_resizable(false);
    pollster::block_on(run(event_loop, window));
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

    let mut render_ctx = RenderContext::new(size.width as u16, size.height as u16);
    draw_simple_scene(&mut render_ctx);
    let bufs = render_ctx.prepare_gpu_buffers();

    let config = Config {
        width: size.width,
        height: size.height,
        strip_height: 4,
    };
    let renderer = Renderer::new(&device, format, &config, &bufs);
    renderer.prepare(&device, &config, &bufs);

    event_loop
        .run(move |event, target| {
            if let Event::WindowEvent {
                window_id: _,
                event: window_event,
            } = event
            {
                match window_event {
                    WindowEvent::RedrawRequested => {
                        renderer.render(&device, &surface, &queue, &bufs);
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

fn draw_simple_scene(ctx: &mut RenderContext) {
    let mut path = BezPath::new();
    path.move_to((10.0, 10.0));
    path.line_to((180.0, 20.0));
    path.line_to((30.0, 40.0));
    path.close_path();
    let piet_path = path.into();
    ctx.set_transform(Affine::scale(5.0));
    ctx.set_paint(palette::css::REBECCA_PURPLE.into());
    ctx.fill_path(&piet_path);
    let stroke = Stroke::new(1.0);
    ctx.set_paint(palette::css::DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(&piet_path);
}
