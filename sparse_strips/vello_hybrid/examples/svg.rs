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
use vello_hybrid::{DimensionConstraints, RenderContext, RenderTarget, Renderer};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

/// Main entry point for the SVG GPU renderer example.
/// Creates a window and continuously renders the SVG using the hybrid CPU/GPU renderer.
fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Window::new(&event_loop).unwrap();
    window.set_resizable(false);
    pollster::block_on(run(event_loop, window));
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "Width and height are expected to fit within u16 range"
)]
async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut args = std::env::args().skip(1);
    let svg_filename: String = args.next().expect("svg filename is first arg");
    let svg = std::fs::read_to_string(svg_filename).expect("error reading file");
    let render_scale = 5.0;
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");

    let constraints = DimensionConstraints::default();
    let svg_width = parsed.size.width * render_scale;
    let svg_height = parsed.size.height * render_scale;
    let (width, height) = constraints.calculate_dimensions(svg_width, svg_height);

    let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(width, height));

    let window = Arc::new(window);
    let mut render_ctx = RenderContext::new(width as u16, height as u16);
    render_svg(&mut render_ctx, render_scale, &parsed.items);

    let render_data = render_ctx.prepare_render_data();
    let mut renderer = Renderer::new(RenderTarget::Window(window.clone())).await;
    renderer.prepare(&render_data);

    event_loop
        .run(move |event, target| {
            if let Event::WindowEvent {
                window_id: _,
                event: window_event,
            } = event
            {
                match window_event {
                    WindowEvent::RedrawRequested => {
                        renderer.render_to_surface(&render_data);
                        window.request_redraw();
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
