// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG example for hybrid renderer

mod common;

use std::sync::Arc;

use common::pico_svg::PicoSvg;
use common::render_svg;
use vello_hybrid::{DimensionConstraints, RenderContext, RenderTarget, Renderer};
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
    let mut args = std::env::args().skip(1);
    let svg_filename: String = args.next().expect("svg filename is first arg");
    let svg = std::fs::read_to_string(svg_filename).expect("error reading file");
    let render_scale = 5.0;
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");

    let constraints = DimensionConstraints::default();
    let svg_width = (parsed.size.width * render_scale) as u32;
    let svg_height = (parsed.size.height * render_scale) as u32;
    let (width, height) = constraints.calculate_dimensions(svg_width, svg_height);

    let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(width, height));

    let window = Arc::new(window);
    let width = window.inner_size().width;
    let height = window.inner_size().height;
    let mut render_ctx = RenderContext::new(width as u16, height as u16);
    render_svg(&mut render_ctx, render_scale, &parsed.items);

    let render_data = render_ctx.prepare_render_data();
    let renderer = Renderer::new(RenderTarget::Window(window.clone()), &render_data).await;
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
