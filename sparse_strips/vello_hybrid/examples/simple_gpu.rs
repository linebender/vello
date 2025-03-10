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
use vello_hybrid::{RenderContext, RenderTarget, Renderer};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

/// Main entry point for the simple GPU renderer example.
/// Creates a window and continuously renders a simple scene using the hybrid CPU/GPU renderer.
fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Window::new(&event_loop).unwrap();
    window.set_resizable(false);
    pollster::block_on(run(event_loop, window));
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let window = Arc::new(window);
    let window_clone = window.clone();
    let mut render_ctx = RenderContext::new(
        window.inner_size().width as u16,
        window.inner_size().height as u16,
    );
    draw_simple_scene(&mut render_ctx);
    let render_data = render_ctx.prepare_render_data();
    let mut renderer = Renderer::new(RenderTarget::Window(window)).await;
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
                        window_clone.request_redraw();
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

fn draw_simple_scene(ctx: &mut RenderContext) {
    let mut path = BezPath::new();
    path.move_to((10.0, 10.0));
    path.line_to((180.0, 20.0));
    path.line_to((30.0, 40.0));
    path.close_path();
    let piet_path = path.into();
    ctx.set_transform(Affine::scale(5.0));
    ctx.set_paint(palette::css::REBECCA_PURPLE.into());
    ctx.fill_path(&piet_path);
    let stroke = Stroke::new(1.0);
    ctx.set_paint(palette::css::DARK_BLUE.into());
    ctx.set_stroke(stroke);
    ctx.stroke_path(&piet_path);
}
