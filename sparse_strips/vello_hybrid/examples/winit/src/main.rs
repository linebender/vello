// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renders our example scenes with Vello Hybrid.

mod render_context;
use render_context::{RenderContext, RenderSurface, create_vello_renderer, create_winit_window};
#[cfg(not(target_arch = "wasm32"))]
use std::env;
use std::sync::Arc;
use vello_common::color::palette::css::WHITE;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Affine, Vec2};
use vello_hybrid::{ImageCache, RenderSize, Renderer, Scene};
use vello_hybrid_scenes::{AnyScene, get_example_scenes};
use wgpu::RenderPassDescriptor;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const ZOOM_STEP: f64 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq)]
struct ColorBrush {
    color: AlphaColor<Srgb>,
}

impl Default for ColorBrush {
    fn default() -> Self {
        Self { color: WHITE }
    }
}

struct App<'s> {
    context: RenderContext,
    scenes: Box<[AnyScene]>,
    current_scene: usize,
    renderers: Vec<Option<Renderer>>,
    render_state: RenderState<'s>,
    scene: Scene,
    transform: Affine,
    mouse_down: bool,
    last_cursor_position: Option<Vec2>,
    image_cache: ImageCache,
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    let (scenes, start_scene_index) = {
        let mut start_scene_index = 4;
        let args: Vec<String> = env::args().collect();
        let mut svg_paths: Vec<&str> = Vec::new();

        if args.len() > 1 {
            // Check if the first argument is a number (scene index)
            if let Ok(index) = args[1].parse::<usize>() {
                start_scene_index = index;
            } else {
                // Collect all arguments as SVG paths
                for arg in args.iter().skip(1) {
                    svg_paths.push(arg);
                }
            }
        }
        let scenes = if svg_paths.is_empty() {
            get_example_scenes(None)
        } else {
            get_example_scenes(Some(svg_paths))
        };

        start_scene_index = start_scene_index.min(scenes.len() - 1);
        (scenes, start_scene_index)
    };
    #[cfg(target_arch = "wasm32")]
    let (scenes, start_scene_index) = (get_example_scenes(), 0);

    let mut app = App {
        context: RenderContext::new(),
        renderers: vec![],
        scenes,
        current_scene: start_scene_index,
        render_state: RenderState::Suspended(None),
        scene: Scene::new(1800, 1200),
        transform: Affine::IDENTITY,
        mouse_down: false,
        last_cursor_position: None,
        image_cache: ImageCache::new(),
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

impl ApplicationHandler for App<'_> {
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let RenderState::Active { window, .. } = &self.render_state {
            self.render_state = RenderState::Suspended(Some(window.clone()));
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let RenderState::Suspended(cached_window) = &mut self.render_state else {
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

        let device_handle = &self.context.devices[surface.dev_id];
        self.image_cache.create_bind_group(&device_handle.device);

        self.render_state = RenderState::Active {
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
        let RenderState::Active { surface, window } = &mut self.render_state else {
            return;
        };

        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.context
                    .resize_surface(surface, size.width, size.height);
                self.scene = Scene::new(
                    u16::try_from(size.width).unwrap(),
                    u16::try_from(size.height).unwrap(),
                );
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match logical_key {
                Key::Named(NamedKey::ArrowRight) => {
                    self.current_scene = (self.current_scene + 1) % self.scenes.len();
                    self.transform = Affine::IDENTITY;
                    window.request_redraw();
                }
                Key::Named(NamedKey::ArrowLeft) => {
                    self.current_scene = if self.current_scene == 0 {
                        self.scenes.len() - 1
                    } else {
                        self.current_scene - 1
                    };
                    self.transform = Affine::IDENTITY;
                    window.request_redraw();
                }
                Key::Named(NamedKey::Space) => {
                    // Reset transform on spacebar
                    self.transform = Affine::IDENTITY;
                    window.request_redraw();
                }
                Key::Named(NamedKey::Escape) => {
                    event_loop.exit();
                }
                _ => {}
            },
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_down = state == ElementState::Pressed;
                    if !self.mouse_down {
                        // Mouse button released
                        self.last_cursor_position = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current_pos = Vec2::new(position.x, position.y);

                if self.mouse_down {
                    // Pan the scene if mouse is down
                    if let Some(last_pos) = self.last_cursor_position {
                        let delta = current_pos - last_pos;
                        self.transform = Affine::translate(delta) * self.transform;
                        window.request_redraw();
                    }
                }

                self.last_cursor_position = Some(current_pos);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Handle zoom with mouse wheel
                let delta_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(pos) => pos.y / 100.0,
                };

                if let Some(cursor_pos) = self.last_cursor_position {
                    let zoom_factor = (1.0 + delta_y * ZOOM_STEP).max(0.1);

                    // Zoom centered at cursor position
                    self.transform = Affine::translate(cursor_pos)
                        * Affine::scale(zoom_factor)
                        * Affine::translate(-cursor_pos)
                        * self.transform;

                    window.request_redraw();
                }
            }
            WindowEvent::PinchGesture { delta, .. } => {
                // Handle pinch-to-zoom on touchpad
                let center = Vec2::new(
                    f64::from(surface.config.width) / 2.0,
                    f64::from(surface.config.height) / 2.0,
                );

                let zoom_factor = 1.0 + delta * ZOOM_STEP * 5.0;

                self.transform = Affine::translate(center)
                    * Affine::scale(zoom_factor)
                    * Affine::translate(-center)
                    * self.transform;

                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let device_handle = &self.context.devices[surface.dev_id];
                self.scene.reset();

                self.scene.set_transform(self.transform);
                self.scenes[self.current_scene].render(&mut self.scene, self.transform);

                let render_size = RenderSize {
                    width: surface.config.width,
                    height: surface.config.height,
                };
                self.renderers[surface.dev_id].as_mut().unwrap().prepare(
                    &device_handle.device,
                    &device_handle.queue,
                    &self.scene,
                    &render_size,
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
                        &device_handle.device,
                        &device_handle.queue,
                        &mut self.scene,
                        &mut pass,
                        &mut self.image_cache,
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
