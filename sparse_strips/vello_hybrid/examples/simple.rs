// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple rendering example for the sparse strips hybrid CPU/GPU renderer
//!
//! This example demonstrates drawing basic shapes using the hybrid renderer.
//! It creates a window and continuously renders a simple scene with various shapes.
//! The rendering pipeline uses CPU-side path processing and GPU-accelerated compositing.

mod common;

use common::{RenderContext, RenderSurface, create_vello_renderer, create_winit_window};
use std::sync::Arc;
use vello_common::kurbo::Affine;
use vello_common::peniko::{
    color::palette,
    kurbo::{BezPath, Stroke},
};
use vello_hybrid::{RenderParams, Renderer, Scene};
use wgpu::RenderPassDescriptor;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

/// Main entry point for the simple GPU renderer example.
/// Creates a window and continuously renders a simple scene using the hybrid CPU/GPU renderer.
fn main() {
    let mut app = SimpleVelloApp {
        context: RenderContext::new(),
        renderers: vec![],
        state: RenderState::Suspended(None),
        scene: Scene::new(900, 600),
    };

    let event_loop = EventLoop::new().unwrap();
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
}

#[derive(Debug)]
enum RenderState<'s> {
    Active {
        surface: Box<RenderSurface<'s>>,
        window: Arc<Window>,
    },
    Suspended(Option<Arc<Window>>),
}

struct SimpleVelloApp<'s> {
    context: RenderContext,
    renderers: Vec<Option<Renderer>>,
    state: RenderState<'s>,
    scene: Scene,
}

impl ApplicationHandler for SimpleVelloApp<'_> {
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let RenderState::Active { window, .. } = &self.state {
            self.state = RenderState::Suspended(Some(window.clone()));
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let RenderState::Suspended(cached_window) = &mut self.state else {
            return;
        };

        let window = cached_window.take().unwrap_or_else(|| {
            create_winit_window(
                event_loop,
                self.scene.width().into(),
                self.scene.height().into(),
                true,
            )
        });

        let size = window.inner_size();
        let surface = pollster::block_on(self.context.create_surface(
            window.clone(),
            size.width,
            size.height,
            wgpu::PresentMode::AutoVsync,
            wgpu::TextureFormat::Bgra8Unorm,
        ));

        self.renderers
            .resize_with(self.context.devices.len(), || None);
        self.renderers[surface.dev_id]
            .get_or_insert_with(|| create_vello_renderer(&self.context, &surface));

        self.state = RenderState::Active {
            surface: Box::new(surface),
            window,
        };
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let surface = match &mut self.state {
            RenderState::Active { surface, window } if window.id() == window_id => surface,
            _ => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.context
                    .resize_surface(surface, size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                self.scene.reset();

                draw_simple_scene(&mut self.scene);
                let device_handle = &self.context.devices[surface.dev_id];
                let render_params = RenderParams {
                    width: surface.config.width,
                    height: surface.config.height,
                };
                self.renderers[surface.dev_id].as_mut().unwrap().prepare(
                    &device_handle.device,
                    &device_handle.queue,
                    &self.scene,
                    &render_params,
                );

                let surface_texture = surface
                    .surface
                    .get_current_texture()
                    .expect("failed to get surface texture");

                let texture_view = surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder =
                    device_handle
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Vello Render to Surface pass"),
                        });
                {
                    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                        label: Some("Render to Texture Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &texture_view,
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
                    self.renderers[surface.dev_id].as_mut().unwrap().render(
                        &self.scene,
                        &mut pass,
                        &render_params,
                    );
                }

                device_handle.queue.submit([encoder.finish()]);
                surface_texture.present();

                device_handle.device.poll(wgpu::Maintain::Poll);
            }
            _ => {}
        }
    }
}

/// Draws a simple scene with shapes
fn draw_simple_scene(ctx: &mut Scene) {
    let mut path = BezPath::new();
    path.move_to((10.0, 10.0));
    path.line_to((180.0, 20.0));
    path.line_to((30.0, 40.0));
    path.close_path();
    ctx.set_transform(Affine::scale(5.0));
    ctx.set_paint(palette::css::REBECCA_PURPLE.into());
    ctx.fill_path(&path);
    let stroke = Stroke::new(1.0);
    ctx.set_paint(palette::css::DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(&path);
}
