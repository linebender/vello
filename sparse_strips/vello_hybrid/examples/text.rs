// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Draws some text on screen.

mod common;

use common::{RenderContext, RenderSurface, create_vello_renderer, create_winit_window};
use parley::FontFamily;
use parley::{
    Alignment, AlignmentOptions, FontContext, GlyphRun, Layout, LayoutContext,
    PositionedLayoutItem, StyleProperty,
};
use std::sync::Arc;
use vello_common::color::palette::css::WHITE;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::glyph::Glyph;
use vello_hybrid::{RenderParams, Renderer, Scene};
use wgpu::RenderPassDescriptor;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const ROBOTO_FONT: &[u8] = include_bytes!("../../../examples/assets/roboto/Roboto-Regular.ttf");

#[derive(Clone, Copy, Debug, PartialEq)]
struct ColorBrush {
    color: AlphaColor<Srgb>,
}

impl Default for ColorBrush {
    fn default() -> Self {
        Self { color: WHITE }
    }
}

fn main() {
    let mut app = App {
        context: RenderContext::new(),
        font_cx: FontContext::new(),
        layout_cx: LayoutContext::new(),
        renderers: vec![],
        state: RenderState::Suspended(None),
        scene: Scene::new(900, 600),
    };

    // Note: If you set `default-features = true` in the `parley` dependency, you automatically
    // get access to system fonts. Since we want to ensure this example can be compiled to Wasm,
    // we are passing the font data directly to the font context.
    app.font_cx.collection.register_fonts(ROBOTO_FONT.to_vec());

    let event_loop = EventLoop::new().unwrap();
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
}

#[derive(Debug)]
enum RenderState<'s> {
    Active {
        surface: Box<RenderSurface<'s>>,
        window: Arc<Window>,
    },
    Suspended(Option<Arc<Window>>),
}

struct App<'s> {
    context: RenderContext,
    font_cx: FontContext,
    layout_cx: LayoutContext<ColorBrush>,
    renderers: Vec<Option<Renderer>>,
    state: RenderState<'s>,
    scene: Scene,
}

impl ApplicationHandler for App<'_> {
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let RenderState::Active { window, .. } = &self.state {
            self.state = RenderState::Suspended(Some(window.clone()));
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let RenderState::Suspended(cached_window) = &mut self.state else {
            return;
        };

        let window = cached_window.take().unwrap_or_else(|| {
            create_winit_window(
                event_loop,
                self.scene.width().into(),
                self.scene.height().into(),
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

        self.renderers
            .resize_with(self.context.devices.len(), || None);
        self.renderers[surface.dev_id]
            .get_or_insert_with(|| create_vello_renderer(&self.context, &surface));

        self.state = RenderState::Active {
            surface: Box::new(surface),
            window,
        };
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let surface = match &mut self.state {
            RenderState::Active { surface, window } if window.id() == window_id => surface,
            _ => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.context
                    .resize_surface(surface, size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                self.scene.reset();

                draw_text(
                    &mut self.scene,
                    "Hello from Vello Hybrid and Parley!",
                    &mut self.font_cx,
                    &mut self.layout_cx,
                );

                let device_handle = &self.context.devices[surface.dev_id];
                let render_params = RenderParams {
                    width: surface.config.width,
                    height: surface.config.height,
                };
                self.renderers[surface.dev_id].as_mut().unwrap().prepare(
                    &device_handle.device,
                    &device_handle.queue,
                    &self.scene,
                    &render_params,
                );

                let surface_texture = surface
                    .surface
                    .get_current_texture()
                    .expect("failed to get surface texture");

                let texture_view = surface_texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder =
                    device_handle
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Vello Render to Surface pass"),
                        });
                {
                    let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                        label: Some("Render to Texture Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &texture_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    });
                    self.renderers[surface.dev_id].as_mut().unwrap().render(
                        &self.scene,
                        &mut pass,
                        &render_params,
                    );
                }

                device_handle.queue.submit([encoder.finish()]);
                surface_texture.present();

                device_handle.device.poll(wgpu::Maintain::Poll);
            }
            _ => {}
        }
    }
}

fn draw_text(
    ctx: &mut Scene,
    text: &str,
    font_cx: &mut FontContext,
    layout_cx: &mut LayoutContext<ColorBrush>,
) {
    let mut builder = layout_cx.ranged_builder(font_cx, text, 1.0);
    builder.push_default(FontFamily::parse("Roboto").unwrap());
    builder.push_default(StyleProperty::LineHeight(1.3));
    builder.push_default(StyleProperty::FontSize(32.0));

    let mut layout: Layout<ColorBrush> = builder.build(text);
    let max_advance = Some(400.0);
    layout.break_all_lines(max_advance);
    layout.align(max_advance, Alignment::Middle, AlignmentOptions::default());

    for line in layout.lines() {
        for item in line.items() {
            if let PositionedLayoutItem::GlyphRun(glyph_run) = item {
                render_glyph_run(ctx, &glyph_run, 30);
            }
        }
    }
}

fn render_glyph_run(ctx: &mut Scene, glyph_run: &GlyphRun<'_, ColorBrush>, padding: u32) {
    let mut run_x = glyph_run.offset();
    let run_y = glyph_run.baseline();
    let glyphs = glyph_run.glyphs().map(|glyph| {
        let glyph_x = run_x + glyph.x + padding as f32;
        let glyph_y = run_y - glyph.y + padding as f32;
        run_x += glyph.advance;

        Glyph {
            id: glyph.id as u32,
            x: glyph_x,
            y: glyph_y,
        }
    });

    let run = glyph_run.run();
    let font = run.font();
    let font_size = run.font_size();
    let normalized_coords = bytemuck::cast_slice(run.normalized_coords());

    let style = glyph_run.style();
    ctx.set_paint(style.brush.color.into());
    ctx.glyph_run(font)
        .font_size(font_size)
        .normalized_coords(normalized_coords)
        .hint(true)
        .fill_glyphs(glyphs);
}
