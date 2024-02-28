// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use anyhow::Result;
use std::num::NonZeroUsize;
use std::sync::Arc;
use vello::kurbo::{Circle, Ellipse, Line, RoundedRect, Stroke};
use vello::peniko::Color;
use vello::util::RenderSurface;
use vello::RendererOptions;
use vello::{kurbo::Affine, util::RenderContext, AaConfig, Renderer, Scene};
use winit::{dpi::LogicalSize, window::WindowBuilder};
use winit::{event::*, event_loop::ControlFlow};
use winit::{event_loop::EventLoop, window::Window};

// Simple struct to hold the state of the renderer
pub struct RenderState<'s> {
    // The fields MUST be in this order, so that the surface is dropped before the window
    surface: RenderSurface<'s>,
    window: Arc<Window>,
}

fn main() -> Result<()> {
    // Setup a bunch of application state
    let mut render_cx = RenderContext::new().unwrap();
    let mut renderers: Vec<Option<Renderer>> = vec![];
    let mut render_state: Option<RenderState> = None;
    // Cache a window so that it can be reused when the app is resumed after being suspended
    let mut cached_window = None;
    let mut scene = Scene::new();

    // Create and run a winit event loop
    let event_loop = EventLoop::new()?;
    event_loop
        .run(move |event, event_loop| match event {
            // Setup renderer. In winit apps it is recommended to do setup in Event::Resumed
            // for best cross-platform compatibility
            Event::Resumed => {
                let Option::None = render_state else { return };

                // Get the winit window cached in a previous Suspended event or else create a new window
                let window = cached_window
                    .take()
                    .unwrap_or_else(|| create_winit_window(event_loop));

                // Create a vello Surface
                let size = window.inner_size();
                let surface_future =
                    render_cx.create_surface(window.clone(), size.width, size.height);
                let surface = pollster::block_on(surface_future).expect("Error creating surface");

                // Create a vello Renderer for the surface (using its device id)
                renderers.resize_with(render_cx.devices.len(), || None);
                renderers[surface.dev_id]
                    .get_or_insert_with(|| create_vello_renderer(&render_cx, &surface));

                // Save the Window and Surface to a state variable
                render_state = Some(RenderState { window, surface });

                event_loop.set_control_flow(ControlFlow::Poll);
            }

            // Save window state on suspend
            Event::Suspended => {
                if let Some(render_state) = render_state.take() {
                    cached_window = Some(render_state.window);
                }
                event_loop.set_control_flow(ControlFlow::Wait);
            }

            Event::WindowEvent {
                ref event,
                window_id,
            } => {
                let Some(render_state) = &mut render_state else {
                    return;
                };
                if render_state.window.id() != window_id {
                    return;
                }
                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::Resized(size) => {
                        // Resize the surface when the window is resized
                        render_cx.resize_surface(
                            &mut render_state.surface,
                            size.width,
                            size.height,
                        );
                        render_state.window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        // This is where all the rendering happens

                        // Get the window size
                        let width = render_state.surface.config.width;
                        let height = render_state.surface.config.height;
                        // Get a handle to the device
                        let device_handle = &render_cx.devices[render_state.surface.dev_id];
                        // Use the render_state to retrieve the surface to draw to
                        let surface_texture = render_state
                            .surface
                            .surface
                            .get_current_texture()
                            .expect("failed to get surface texture");

                        // Define the render parameters.
                        let render_params = vello::RenderParams {
                            // Background color
                            base_color: Color::BLACK,
                            // Width
                            width,
                            // Height
                            height,
                            // Antialiasing method to use. Other methods: AaConfig::Area, AaConfig::Msaa8
                            antialiasing_method: AaConfig::Msaa16,
                        };

                        // Create some shapes!
                        let stroke = Stroke::new(6.0);

                        let rect = RoundedRect::new(10.0, 10.0, 240.0, 240.0, 20.0);
                        let rect_stroke_color = Color::rgb(0.9804, 0.702, 0.5294);

                        let circle = Circle::new((420.0, 200.0), 120.0);
                        let circle_fill_color = Color::rgb(0.9529, 0.5451, 0.6588);

                        let ellipse = Ellipse::new((250.0, 420.0), (100.0, 160.0), -90.0);
                        let ellipse_fill_color = Color::rgb(0.7961, 0.651, 0.9686);

                        let line = Line::new((260.0, 20.0), (620.0, 100.0));
                        let line_stroke_color = Color::rgb(0.5373, 0.7059, 0.9804);

                        scene.reset();

                        // Draw the shapes!
                        scene.stroke(&stroke, Affine::IDENTITY, rect_stroke_color, None, &rect);
                        scene.fill(
                            vello::peniko::Fill::NonZero,
                            Affine::IDENTITY,
                            circle_fill_color,
                            None,
                            &circle,
                        );
                        scene.fill(
                            vello::peniko::Fill::NonZero,
                            Affine::IDENTITY,
                            ellipse_fill_color,
                            None,
                            &ellipse,
                        );
                        scene.stroke(&stroke, Affine::IDENTITY, line_stroke_color, None, &line);

                        // Render to the surface
                        renderers[render_state.surface.dev_id]
                            .as_mut()
                            .unwrap()
                            .render_to_surface(
                                &device_handle.device,
                                &device_handle.queue,
                                &scene,
                                &surface_texture,
                                &render_params,
                            )
                            .expect("failed to render to surface");
                        surface_texture.present();
                        device_handle.device.poll(wgpu::Maintain::Poll);
                    }
                    _ => {}
                }
            }
            _ => {}
        })
        .expect("Couldn't run event loop");
    Ok(())
}

/// Helper function that creates a Winit window and returns it (wrapped in an Arc for thread safety)
fn create_winit_window(event_loop: &winit::event_loop::EventLoopWindowTarget<()>) -> Arc<Window> {
    Arc::new(
        WindowBuilder::new()
            .with_inner_size(LogicalSize::new(1044, 800))
            .with_resizable(true)
            .with_title("Vello Shapes")
            .build(event_loop)
            .unwrap(),
    )
}

/// Helper function that creates a vello Renderer for a given RenderContext and Surface
fn create_vello_renderer(render_cx: &RenderContext, surface: &RenderSurface) -> Renderer {
    Renderer::new(
        &render_cx.devices[surface.dev_id].device,
        RendererOptions {
            surface_format: Some(surface.format),
            use_cpu: false,
            antialiasing_support: vello::AaSupport::all(),
            num_init_threads: NonZeroUsize::new(1),
        },
    )
    .expect("Could create renderer")
}
