// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple rendering example for the sparse strips hybrid CPU/GPU renderer
//!
//! This example demonstrates drawing basic shapes using the hybrid renderer.
//! It creates a window and continuously renders a simple scene with various shapes.
//! The rendering pipeline uses CPU-side path processing and GPU-accelerated compositing.

use std::sync::Arc;

use kurbo::Affine;
use peniko::{
    color::palette,
    kurbo::{BezPath, Stroke},
};
use vello_hybrid::{RenderContext, RenderData, RenderTarget, Renderer};
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
        context: None,
        state: None,
    };

    let event_loop = EventLoop::new().unwrap();
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
}

/// State for active rendering
struct SimpleRenderState {
    // The window for presenting the rendered content
    window: Arc<Window>,
    // The renderer for handling GPU operations
    renderer: Renderer,
    // The processed data ready to be sent to the GPU
    render_data: RenderData,
}

/// Main application state
struct SimpleVelloApp {
    // The vello RenderContext which contains rendering state
    context: Option<RenderContext>,
    // Active render state (either active or none)
    state: Option<SimpleRenderState>,
}

impl ApplicationHandler for SimpleVelloApp {
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // Release resources when app is suspended
        self.state = None;
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "Width and height are expected to fit within u16 range"
    )]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window = create_winit_window(event_loop, 900, 600, false);

            let mut context = RenderContext::new(
                window.inner_size().width as u16,
                window.inner_size().height as u16,
            );
            draw_simple_scene(&mut context);
            let render_data = context.prepare_render_data();
            self.context = Some(context);

            let mut renderer = pollster::block_on(async {
                Renderer::new(RenderTarget::Window(Arc::clone(&window))).await
            });

            renderer.prepare(&render_data);

            window.set_visible(true);
            window.focus_window();
            window.request_redraw();

            self.state = Some(SimpleRenderState {
                window,
                renderer,
                render_data,
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = &self.state {
            match event {
                WindowEvent::RedrawRequested => {
                    state.renderer.render_to_surface(&state.render_data);
                    state.window.request_redraw();
                }
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                _ => {}
            }
        }
    }
}

/// Helper function that creates a Winit window and returns it (wrapped in an Arc for sharing)
fn create_winit_window(
    event_loop: &ActiveEventLoop,
    width: u32,
    height: u32,
    initially_visible: bool, // Whether the window should be visible initially or made visible later
) -> Arc<Window> {
    let attr = Window::default_attributes()
        .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
        .with_resizable(false)
        .with_title("Vello Simple Renderer")
        .with_visible(initially_visible)
        .with_active(true);
    Arc::new(event_loop.create_window(attr).unwrap())
}

/// Draws a simple scene with shapes
fn draw_simple_scene(ctx: &mut RenderContext) {
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
