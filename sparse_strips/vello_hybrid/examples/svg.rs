// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG example for sparse strips hybrid CPU/GPU renderer
//!
//! This example demonstrates loading and rendering an SVG file using the hybrid renderer.
//! It creates a window and renders the SVG using CPU-side path processing
//! and GPU-accelerated compositing.

mod common;

use std::sync::Arc;

use common::{
    RenderContext, RenderSurface, create_vello_renderer, create_winit_window, render_svg,
};
use vello_common::pico_svg::PicoSvg;
use vello_hybrid::{RenderParams, Renderer, Scene};
use wgpu::RenderPassDescriptor;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

/// Main entry point for the SVG GPU renderer example.
/// Creates a window and renders the SVG using the hybrid CPU/GPU renderer.
fn main() {
    let mut app = SvgVelloApp {
        context: RenderContext::new(),
        renderers: vec![],
        state: RenderState::Suspended(None),
        scene: Scene::new(1600, 1200),
        render_scale: 5.0,
        parsed_svg: None,
    };

    let event_loop = EventLoop::new().expect("Couldn't create event loop");
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
}

// Constants for zoom behavior
const MIN_SCALE: f64 = 0.1;
const MAX_SCALE: f64 = 20.0;
const ZOOM_STEP: f64 = 0.5;

#[derive(Debug)]
enum RenderState<'s> {
    /// `RenderSurface` and `Window` for active rendering.
    Active {
        // The `RenderSurface` and the `Window` must be in this order, so that the surface is dropped first.
        surface: Box<RenderSurface<'s>>,
        window: Arc<Window>,
    },
    /// Cache a window so that it can be reused when the app is resumed after being suspended.
    Suspended(Option<Arc<Window>>),
}

struct SvgVelloApp<'s> {
    // The vello RenderContext which is a global context that lasts for the
    // lifetime of the application
    context: RenderContext,

    // An array of renderers, one per wgpu device
    renderers: Vec<Option<Renderer>>,

    // State for our example where we store the winit Window and the wgpu Surface
    state: RenderState<'s>,

    // A vello Scene which is a data structure which allows one to build up a
    // description a scene to be drawn (with paths, fills, images, text, etc)
    // which is then passed to a renderer for rendering
    scene: Scene,

    // The scale factor for the rendered SVG
    render_scale: f64,

    // The parsed SVG
    parsed_svg: Option<PicoSvg>,
}

impl SvgVelloApp<'_> {
    /// Adjust the render scale by the given delta, clamping to min/max values
    fn adjust_scale(&mut self, delta: f64) {
        self.render_scale = (self.render_scale + delta).clamp(MIN_SCALE, MAX_SCALE);
    }
}

impl ApplicationHandler for SvgVelloApp<'_> {
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

        // Create a vello Renderer for the surface (using its device id)
        self.renderers
            .resize_with(self.context.devices.len(), || None);
        self.renderers[surface.dev_id]
            .get_or_insert_with(|| create_vello_renderer(&self.context, &surface));

        let svg_filename = std::env::args().nth(1).expect("svg filename is first arg");
        let svg: String = std::fs::read_to_string(svg_filename).expect("error reading file");
        self.parsed_svg = Some(PicoSvg::load(&svg, 1.0).expect("error parsing SVG"));

        self.state = RenderState::Active {
            surface: Box::new(surface),
            window,
        };
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let RenderState::Active { window, .. } = &self.state {
            self.state = RenderState::Suspended(Some(window.clone()));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        // Only process events for our window, and only when we have a surface.
        let (surface, window) = match &mut self.state {
            RenderState::Active { surface, window } if window.id() == window_id => {
                (surface, window)
            }
            _ => return,
        };

        match event {
            // Exit the event loop when a close is requested (e.g. window's close button is pressed)
            WindowEvent::CloseRequested => event_loop.exit(),

            // Resize the surface when the window is resized
            WindowEvent::Resized(size) => {
                self.context
                    .resize_surface(surface, size.width, size.height);
            }

            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::PixelDelta(pos),
                ..
            } => {
                // Convert pixel delta to a scale adjustment
                // Divide by a factor to make the zoom less sensitive
                self.adjust_scale(pos.y * ZOOM_STEP / 50.0);
            }

            WindowEvent::PinchGesture { delta, .. } => {
                self.adjust_scale(delta * ZOOM_STEP);
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                match logical_key {
                    Key::Character(c) => match c.as_str() {
                        "+" | "=" => self.adjust_scale(ZOOM_STEP),
                        "-" | "_" => self.adjust_scale(-ZOOM_STEP),
                        // Reset to original scale
                        "0" => {
                            self.render_scale = 5.0;
                        }
                        _ => {}
                    },
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    _ => {}
                }
            }

            WindowEvent::RedrawRequested => {
                self.scene.reset();

                if let Some(parsed_svg) = &self.parsed_svg {
                    render_svg(&mut self.scene, self.render_scale, &parsed_svg.items);
                }

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
                let view = surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                // Copy texture to buffer
                let mut encoder =
                    device_handle
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Vello Render To Buffer"),
                        });
                {
                    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
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
                    self.renderers[surface.dev_id].as_mut().unwrap().render(
                        &self.scene,
                        &mut pass,
                        &render_params,
                    );
                }
                device_handle.queue.submit([encoder.finish()]);
                surface_texture.present();
                window.request_redraw();
            }
            _ => {}
        }
    }
}
