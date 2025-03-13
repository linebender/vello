// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::sync::Arc;

use kurbo::Affine;
use vello_common::pico_svg::Item;
use vello_hybrid::{
    Renderer, RendererOptions, Scene,
    util::{RenderContext, RenderSurface},
};
use winit::{event_loop::ActiveEventLoop, window::Window};

/// Define a render function that works with our `pico_svg::Item` type
#[allow(dead_code, reason = "This is a helper function for the examples")]
pub(crate) fn render_svg(ctx: &mut Scene, scale: f64, items: &[Item]) {
    ctx.set_transform(Affine::scale(scale));
    for item in items {
        match item {
            Item::Fill(fill_item) => {
                ctx.set_paint(fill_item.color.into());
                ctx.fill_path(&fill_item.path);
            }
            Item::Stroke(stroke_item) => {
                let style = kurbo::Stroke::new(stroke_item.width);
                ctx.set_stroke(style);
                ctx.set_paint(stroke_item.color.into());
                ctx.stroke_path(&stroke_item.path);
            }
            Item::Group(group_item) => {
                // TODO: apply transform from group
                render_svg(ctx, scale, &group_item.children);
            }
        }
    }
}

/// Helper function that creates a Winit window and returns it (wrapped in an Arc for sharing)
#[allow(dead_code, reason = "This is a helper function for the examples")]
pub(crate) fn create_winit_window(
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

/// Helper function that creates a Vello Hybrid renderer
#[allow(dead_code, reason = "This is a helper function for the examples")]
pub(crate) fn create_vello_renderer(
    render_cx: &RenderContext,
    surface: &RenderSurface<'_>,
) -> Renderer {
    Renderer::new(
        &render_cx.devices[surface.dev_id].device,
        &RendererOptions {},
    )
}
