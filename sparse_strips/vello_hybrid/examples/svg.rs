// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG example for sparse strips hybrid CPU/GPU renderer
//!
//! This example demonstrates loading and rendering an SVG file using the hybrid renderer.
//! It creates a window and continuously renders the SVG using CPU-side path processing
//! and GPU-accelerated compositing.

mod common;

use std::sync::Arc;

use common::render_svg;
use vello_common::pico_svg::PicoSvg;
use vello_hybrid::{
    DimensionConstraints, RenderContext, RenderData, RenderTarget, Renderer, SurfaceTarget,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

/// Main entry point for the SVG GPU renderer example.
/// Creates a window and continuously renders the SVG using the hybrid CPU/GPU renderer.
fn main() {
    let mut app = SvgVelloApp {
        context: None,
        parsed_svg: None,
        render_scale: 5.0,
        state: None,
    };

    let event_loop = EventLoop::new().unwrap();
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
}

/// State for active rendering
struct SvgRenderState {
    // The window for presenting the rendered content
    window: Arc<Window>,
    // The renderer for handling GPU operations
    renderer: Renderer,
    // The processed data ready to be sent to the GPU
    render_data: RenderData,
}

/// Main application state
struct SvgVelloApp {
    // The vello RenderContext which contains rendering state
    context: Option<RenderContext>,
    // The SVG that we'll be rendering
    parsed_svg: Option<PicoSvg>,
    // Rendering scale factor
    render_scale: f64,
    // Active render state (either active or none)
    state: Option<SvgRenderState>,
}

impl ApplicationHandler for SvgVelloApp {
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.state = None;
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "Width and height are expected to fit within u16 range"
    )]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let svg_filename = std::env::args().nth(1).expect("svg filename is first arg");
            let svg = std::fs::read_to_string(svg_filename).expect("error reading file");
            let parsed_svg = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");
            self.parsed_svg = Some(parsed_svg);

            let parsed = self.parsed_svg.as_ref().unwrap();
            let constraints = DimensionConstraints::default();
            let svg_width = parsed.size.width * self.render_scale;
            let svg_height = parsed.size.height * self.render_scale;
            let (width, height) = constraints.calculate_dimensions(svg_width, svg_height);

            let window = create_winit_window(event_loop, width as u32, height as u32, false);

            let mut context = RenderContext::new(width as u16, height as u16);
            render_svg(&mut context, self.render_scale, &parsed.items);
            let render_data = context.prepare_render_data();
            self.context = Some(context);

            let mut renderer = pollster::block_on(async {
                Renderer::new(RenderTarget::Surface {
                    target: Arc::clone(&window) as Arc<dyn SurfaceTarget>,
                    width: width as u32,
                    height: height as u32,
                })
                .await
            });

            renderer.prepare(&render_data);

            window.set_visible(true);
            window.focus_window();
            window.request_redraw();

            self.state = Some(SvgRenderState {
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
    initially_visible: bool,
) -> Arc<Window> {
    let attr = Window::default_attributes()
        .with_inner_size(winit::dpi::PhysicalSize::new(width, height))
        .with_resizable(false)
        .with_title("Vello SVG Renderer")
        .with_visible(initially_visible)
        .with_active(true);
    Arc::new(event_loop.create_window(attr).unwrap())
}
