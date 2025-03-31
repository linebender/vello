// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Draws some text on screen.

mod common;

use common::{RenderContext, RenderSurface, create_vello_renderer, create_winit_window};
use skrifa::MetadataProvider;
use skrifa::raw::FileRef;
use std::sync::Arc;
use vello_common::glyph::Glyph;
use vello_common::kurbo::Affine;
use vello_common::peniko::color::palette;
use vello_common::peniko::{Blob, Font};
use vello_hybrid::{RenderParams, Renderer, Scene};
use wgpu::RenderPassDescriptor;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const ROBOTO_FONT: &[u8] = include_bytes!("../../../examples/assets/roboto/Roboto-Regular.ttf");

fn main() {
    let mut app = App {
        context: RenderContext::new(),
        renderers: vec![],
        state: RenderState::Suspended(None),
        scene: Scene::new(900, 600),
    };

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

                draw_text(&mut self.scene);
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

fn draw_text(ctx: &mut Scene) {
    let font = Font::new(Blob::new(Arc::new(ROBOTO_FONT)), 0);
    let font_ref = {
        let file_ref = FileRef::new(font.data.as_ref()).unwrap();
        match file_ref {
            FileRef::Font(f) => f,
            FileRef::Collection(collection) => collection.get(font.index).unwrap(),
        }
    };
    let axes = font_ref.axes();
    let size = 52_f32;
    let font_size = skrifa::instance::Size::new(size);
    let variations: Vec<(&str, f32)> = vec![];
    let var_loc = axes.location(variations.as_slice());
    let charmap = font_ref.charmap();
    let metrics = font_ref.metrics(font_size, &var_loc);
    let line_height = metrics.ascent - metrics.descent + metrics.leading;
    let glyph_metrics = font_ref.glyph_metrics(font_size, &var_loc);

    let mut pen_x = 0_f32;
    let mut pen_y = 0_f32;

    let text = "Hello, world!";

    let glyphs = text
        .chars()
        .filter_map(|ch| {
            if ch == '\n' {
                pen_y += line_height;
                pen_x = 0.0;
                return None;
            }
            let gid = charmap.map(ch).unwrap_or_default();
            let advance = glyph_metrics.advance_width(gid).unwrap_or_default();
            let x = pen_x;
            pen_x += advance;
            Some(Glyph {
                id: gid.to_u32(),
                x,
                y: pen_y,
            })
        })
        .collect::<Vec<_>>();

    ctx.set_paint(palette::css::WHITE.into());
    let transform = Affine::scale(2.0).then_translate((0., f64::from(size) * 2.0).into());
    ctx.set_transform(transform);

    // Fill the text
    ctx.glyph_run(&font)
        .normalized_coords(vec![])
        .font_size(size)
        .hint(true)
        .fill_glyphs(glyphs.iter());

    ctx.set_transform(transform.then_translate((0., f64::from(size) * 2.0).into()));

    // Stroke the text
    ctx.glyph_run(&font)
        .font_size(size)
        .hint(true)
        .stroke_glyphs(glyphs.iter());

    ctx.set_transform(transform.then_translate((0., f64::from(size) * 4.0).into()));

    // Skew the text to the right
    ctx.glyph_run(&font)
        .font_size(size)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.0))
        .hint(true)
        .stroke_glyphs(glyphs.iter());
}
