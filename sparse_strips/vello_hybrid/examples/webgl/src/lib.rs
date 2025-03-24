// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Demonstrates using Vello Hybrid using a WebGL2 backend in the browser.

use vello_common::peniko::{color::palette, kurbo::BezPath};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
struct RendererWrapper {
    renderer: vello_hybrid::Renderer,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
}

#[cfg(target_arch = "wasm32")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl RendererWrapper {
    #[cfg(target_arch = "wasm32")]
    async fn new(canvas: web_sys::HtmlCanvasElement) -> Self {
        let width = canvas.width();
        let height = canvas.height();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .expect("Canvas surface to be valid");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("Adapter to be valid");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        // WGPU's downlevel defaults use a generous number of color attachments
                        // (8). Some devices (including CI) support only up to 4.
                        max_color_attachments: 4,
                        ..wgpu::Limits::downlevel_webgl2_defaults()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("Device to be valid");

        // Configure the surface
        let surface_format = wgpu::TextureFormat::Rgba8Unorm;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let renderer = vello_hybrid::Renderer::new(
            &device,
            &vello_hybrid::RendererOptions {
                format: surface_format,
            },
        );

        Self {
            renderer,
            device,
            queue,
            surface,
        }
    }
}

/// Creates a `HTMLCanvasElement` of the given dimensions and renders the given `Scene` into it.
#[cfg(target_arch = "wasm32")]
pub async fn render_scene(scene: vello_hybrid::Scene, width: u16, height: u16) {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).unwrap();
    let canvas = web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(width as u32);
    canvas.set_height(height as u32);
    // Add canvas to body
    web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .body()
        .unwrap()
        .append_child(&canvas)
        .unwrap();

    let RendererWrapper {
        mut renderer,
        device,
        queue,
        surface,
    } = RendererWrapper::new(canvas).await;

    let params = vello_hybrid::RenderParams {
        width: width as u32,
        height: height as u32,
    };
    renderer.prepare(&device, &queue, &scene, &params);

    let surface_texture = surface.get_current_texture().unwrap();
    let surface_texture_view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &surface_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        renderer.render(&scene, &mut pass, &params);
    }

    queue.submit([encoder.finish()]);
    surface_texture.present();
}

/// Draws a blue triangle into the given `Scene`.
pub fn draw_triangle(ctx: &mut vello_hybrid::Scene) {
    let mut path = BezPath::new();
    path.move_to((30.0, 40.0));
    path.line_to((50.0, 20.0));
    path.line_to((70.0, 40.0));
    path.close_path();
    ctx.set_paint(palette::css::BLUE.into());
    ctx.fill_path(&path);
}
