use anyhow::Result;
use std::num::NonZeroUsize;
use std::sync::Arc;
use vello::kurbo::{Circle, RoundedRect, Stroke};
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

// Helper function that creates a Winit window and returns it (wrapped in an Arc for thread safety)
fn create_window(event_loop: &winit::event_loop::EventLoopWindowTarget<()>) -> Arc<Window> {
    Arc::new(
        WindowBuilder::new()
            .with_inner_size(LogicalSize::new(1044, 800))
            .with_resizable(true)
            .with_title("Vello Shapes")
            .build(event_loop)
            .unwrap(),
    )
}

// Runs the Winit event loop and renders the shapes
fn run(event_loop: EventLoop<()>, mut render_cx: RenderContext) {
    // Create an empty vec of `Renderer`s.
    let mut renderers: Vec<Option<Renderer>> = vec![];
    // Create a RenderState wrapped in an Option.
    let mut render_state = None::<RenderState>;
    // Cache a window so that it can be reused when the app is resumed after being suspended
    let mut cached_window = None;
    // Create a new scene
    let mut scene = Scene::new();

    // Run the event loop
    event_loop
        .run(move |event, event_loop| match event {
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
                        render_cx.resize_surface(&mut render_state.surface, size.width, size.height)
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
                        let rect = RoundedRect::new(10.0, 10.0, 240.0, 240.0, 20.0);
                        let rect_stroke = Stroke::new(6.0);
                        let rect_stroke_color = Color::rgb(0.9804, 0.702, 0.5294);

                        let circle = Circle::new((420.0, 200.0), 120.0);
                        let circle_fill_color = Color::rgb(0.9529, 0.5451, 0.6588);

                        // Draw the shapes!
                        scene.stroke(
                            &rect_stroke,
                            Affine::IDENTITY,
                            rect_stroke_color,
                            None,
                            &rect,
                        );
                        scene.fill(
                            vello::peniko::Fill::NonZero,
                            Affine::IDENTITY,
                            circle_fill_color,
                            None,
                            &circle,
                        );

                        // Render to the surface
                        vello::block_on_wgpu(
                            &device_handle.device,
                            renderers[render_state.surface.dev_id]
                                .as_mut()
                                .unwrap()
                                .render_to_surface_async(
                                    &device_handle.device,
                                    &device_handle.queue,
                                    &scene,
                                    &surface_texture,
                                    &render_params,
                                ),
                        )
                        .expect("failed to render to surface");
                        surface_texture.present();
                        device_handle.device.poll(wgpu::Maintain::Poll);
                    }
                    _ => {}
                }
            }
            Event::Suspended => {
                if let Some(render_state) = render_state.take() {
                    cached_window = Some(render_state.window);
                }
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            Event::Resumed => {
                let Option::None = render_state else { return };
                // Get the window
                let window = cached_window
                    .take()
                    .unwrap_or_else(|| create_window(event_loop));
                let size = window.inner_size();
                let surface_future =
                    render_cx.create_surface(window.clone(), size.width, size.height);
                // Create a surface
                let surface = pollster::block_on(surface_future).expect("Error creating surface");
                render_state = {
                    let render_state = RenderState { window, surface };
                    renderers.resize_with(render_cx.devices.len(), || None);
                    let id = render_state.surface.dev_id;
                    renderers[id].get_or_insert_with(|| {
                        Renderer::new(
                            &render_cx.devices[id].device,
                            RendererOptions {
                                surface_format: Some(render_state.surface.format),
                                use_cpu: false,
                                antialiasing_support: vello::AaSupport::all(),
                                num_init_threads: NonZeroUsize::new(1),
                            },
                        )
                        .expect("Could create renderer")
                    });
                    Some(render_state)
                };
                event_loop.set_control_flow(ControlFlow::Poll);
            }
            _ => {}
        })
        .expect("Couldnt run event loop");
}

fn main() -> Result<()> {
    // Create the event loop and the rendex context
    let event_loop = EventLoop::new()?;
    let render_cx = RenderContext::new().unwrap();

    // Run the event loop
    run(event_loop, render_cx);

    Ok(())
}
