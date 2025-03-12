// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG example for sparse strips hybrid CPU/GPU renderer
//!
//! This example demonstrates loading and rendering an SVG file using the hybrid renderer.
//! It creates a window and renders the SVG using CPU-side path processing
//! and GPU-accelerated compositing.

mod common;

use std::sync::Arc;

use common::{create_vello_renderer, create_winit_window, render_svg};
use peniko::color::palette;
use vello_common::pico_svg::PicoSvg;
use vello_hybrid::{
    RenderParams, Renderer, Scene,
    util::{RenderContext, RenderSurface},
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
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
    };

    let event_loop = EventLoop::new().expect("Couldn't create event loop");
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
}

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
}

impl ApplicationHandler for SvgVelloApp<'_> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Width and height are expected to fit within u16 range"
    )]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let RenderState::Suspended(cached_window) = &mut self.state else {
            return;
        };

        let window = cached_window.take().unwrap_or_else(|| {
            create_winit_window(
                event_loop,
                self.scene.width() as u32,
                self.scene.height() as u32,
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

        self.scene.reset();

        let render_scale = 5.0;
        let svg_filename = std::env::args().nth(1).expect("svg filename is first arg");
        let svg: String = std::fs::read_to_string(svg_filename).expect("error reading file");
        let parsed_svg = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");
        render_svg(&mut self.scene, render_scale, &parsed_svg.items);
        let device_handle = &self.context.devices[surface.dev_id];
        self.renderers[surface.dev_id].as_mut().unwrap().prepare(
            &device_handle.device,
            &device_handle.queue,
            &self.scene,
            &RenderParams {
                base_color: Some(palette::css::BLACK),
                width: surface.config.width,
                height: surface.config.height,
                strip_height: 4,
            },
        );

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

            WindowEvent::RedrawRequested => {
                let width = surface.config.width;
                let height = surface.config.height;
                let device_handle = &self.context.devices[surface.dev_id];
                let surface_texture = surface
                    .surface
                    .get_current_texture()
                    .expect("failed to get surface texture");
                let view = surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                self.renderers[surface.dev_id]
                    .as_mut()
                    .unwrap()
                    .render_to_texture(
                        &device_handle.device,
                        &device_handle.queue,
                        &self.scene,
                        &view,
                        &RenderParams {
                            base_color: Some(palette::css::BLACK), // Background color
                            width,
                            height,
                            strip_height: 4,
                        },
                    );

                surface_texture.present();
                window.request_redraw();
            }
            _ => {}
        }
    }
}
