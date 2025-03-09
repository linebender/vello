// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(missing_docs, reason = "will add them later")]
#![allow(missing_debug_implementations, reason = "prototyping")]
#![allow(clippy::cast_possible_truncation, reason = "we're doing it on purpose")]

//! SVG example for hybrid renderer

mod pico_svg;

use std::sync::Arc;

use kurbo::{Affine, Stroke};
use pico_svg::{Item, PicoSvg};
use vello_hybrid::{RenderContext, Renderer};
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
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");

    let window = Arc::new(window);
    let mut render_ctx = RenderContext::new(
        window.inner_size().width as u16,
        window.inner_size().height as u16,
    );
    render_svg(&mut render_ctx, &parsed.items);

    let bufs = render_ctx.prepare_render_data();
    let renderer = Renderer::new(window.clone(), &bufs).await;
    renderer.prepare(&bufs);

    event_loop
        .run(move |event, target| {
            if let Event::WindowEvent {
                window_id: _,
                event: window_event,
            } = event
            {
                match window_event {
                    WindowEvent::RedrawRequested => {
                        renderer.render(&bufs);
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

fn render_svg(ctx: &mut RenderContext, items: &[Item]) {
    ctx.set_transform(Affine::scale(5.0));
    for item in items {
        match item {
            Item::Fill(fill_item) => {
                ctx.set_paint(fill_item.color.into());
                ctx.fill_path(&fill_item.path.path);
            }
            Item::Stroke(stroke_item) => {
                let style = Stroke::new(stroke_item.width);
                ctx.set_stroke(style);
                ctx.set_paint(stroke_item.color.into());
                ctx.stroke_path(&stroke_item.path.path);
            }
            Item::Group(group_item) => {
                // TODO: apply transform from group
                render_svg(ctx, &group_item.children);
            }
        }
    }
}
