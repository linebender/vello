// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renders our example scenes with Vello Hybrid.

mod render_context;
use render_context::{RenderContext, RenderSurface, create_vello_renderer, create_winit_window};
#[cfg(not(target_arch = "wasm32"))]
use std::env;
use std::sync::Arc;
use std::time::Instant;
use vello_common::kurbo::{Affine, Point};
use vello_common::paint::ImageId;
use vello_common::paint::ImageSource;
use vello_example_scenes::image::ImageScene;
use vello_example_scenes::{AnyScene, get_example_scenes};
use vello_hybrid::{Pixmap, RenderSize, Renderer, Scene};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const ZOOM_STEP: f64 = 0.1;

struct App<'s> {
    context: RenderContext,
    scenes: Box<[AnyScene<Scene>]>,
    current_scene: usize,
    renderers: Vec<Option<Renderer>>,
    render_state: RenderState<'s>,
    scene: Scene,
    transform: Affine,
    mouse_down: bool,
    last_cursor_position: Option<Point>,
    last_frame_time: Option<Instant>,
    frame_count: u32,
    fps_update_time: Instant,
    accumulated_frame_time: f64,
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    let (scenes, start_scene_index) = {
        let mut start_scene_index = 0;
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
        let img_sources = vec![
            ImageSource::OpaqueId(ImageId::new(0)),
            ImageSource::OpaqueId(ImageId::new(1)),
        ];
        let scenes = if svg_paths.is_empty() {
            get_example_scenes(None, img_sources)
        } else {
            get_example_scenes(Some(svg_paths), img_sources)
        };

        start_scene_index = start_scene_index.min(scenes.len() - 1);
        (scenes, start_scene_index)
    };
    #[cfg(target_arch = "wasm32")]
    let (scenes, start_scene_index) = (
        get_example_scenes(vec![
            ImageSource::OpaqueId(ImageId::new(0)),
            ImageSource::OpaqueId(ImageId::new(1)),
        ]),
        0,
    );

    let now = Instant::now();
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
        last_frame_time: None,
        frame_count: 0,
        fps_update_time: now,
        accumulated_frame_time: 0.0,
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
            wgpu::PresentMode::Immediate, // Unlimited FPS mode
            wgpu::TextureFormat::Bgra8Unorm,
        ));

        self.renderers
            .resize_with(self.context.devices.len(), || None);
        self.renderers[surface.dev_id]
            .get_or_insert_with(|| create_vello_renderer(&self.context, &surface));

        self.upload_images_to_atlas(surface.dev_id);

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
                Key::Character(ch) => {
                    if let Some(scene) = self.scenes.get_mut(self.current_scene)
                        && scene.handle_key(ch.as_str())
                    {
                        window.request_redraw();
                    }
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
                let current_pos = Point {
                    x: position.x,
                    y: position.y,
                };

                if self.mouse_down {
                    // Pan the scene if mouse is down
                    if let Some(last_pos) = self.last_cursor_position {
                        self.transform = self.transform.then_translate(current_pos - last_pos);
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
                    self.transform = self.transform.then_scale_about(zoom_factor, cursor_pos);

                    window.request_redraw();
                }
            }
            WindowEvent::PinchGesture { delta, .. } => {
                // Handle pinch-to-zoom on touchpad.
                let zoom_factor = 1.0 + delta * ZOOM_STEP * 5.0;

                // Zoom centered at cursor position, or the center if no position is set.
                self.transform = self.transform.then_scale_about(
                    zoom_factor,
                    self.last_cursor_position.unwrap_or(Point {
                        x: 0.5 * surface.config.width as f64,
                        y: 0.5 * surface.config.height as f64,
                    }),
                );

                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                // Measure frame time
                let now = Instant::now();
                if let Some(last_time) = self.last_frame_time {
                    let frame_time = now.duration_since(last_time).as_secs_f64() * 1000.0; // Convert to milliseconds
                    self.accumulated_frame_time += frame_time;
                    self.frame_count += 1;

                    // Update window title every second with average FPS
                    if now.duration_since(self.fps_update_time).as_secs_f64() >= 1.0 {
                        let avg_frame_time = self.accumulated_frame_time / self.frame_count as f64;
                        let avg_fps = 1000.0 / avg_frame_time;
                        println!("Average FPS: {avg_fps:.1}");
                        window.set_title(&format!(
                            "Vello Hybrid - Scene {} - {:.1} FPS ({:.2}ms avg)",
                            self.current_scene, avg_fps, avg_frame_time
                        ));

                        // Reset counters
                        self.frame_count = 0;
                        self.accumulated_frame_time = 0.0;
                        self.fps_update_time = now;
                    }
                }
                self.last_frame_time = Some(now);

                self.scene.reset();

                self.scene.set_transform(self.transform);
                self.scenes[self.current_scene].render(&mut self.scene, self.transform);

                let device_handle = &self.context.devices[surface.dev_id];
                let render_size = RenderSize {
                    width: surface.config.width,
                    height: surface.config.height,
                };

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
                self.renderers[surface.dev_id]
                    .as_mut()
                    .unwrap()
                    .render(
                        &self.scene,
                        &device_handle.device,
                        &device_handle.queue,
                        &mut encoder,
                        &render_size,
                        &texture_view,
                    )
                    .unwrap();

                device_handle.queue.submit([encoder.finish()]);
                surface_texture.present();

                device_handle.device.poll(wgpu::PollType::Poll).unwrap();

                // Request continuous redraw for FPS measurement
                window.request_redraw();
            }
            _ => {}
        }
    }
}

impl App<'_> {
    fn upload_images_to_atlas(&mut self, device_id: usize) {
        let device_handle = &self.context.devices[device_id];
        let mut encoder =
            device_handle
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Upload Image pass"),
                });

        // 1st example — uploading pixmap directly
        let pixmap1 = ImageScene::read_flower_image();
        self.renderers[device_id].as_mut().unwrap().upload_image(
            &device_handle.device,
            &device_handle.queue,
            &mut encoder,
            &pixmap1,
        );

        // 2nd example — uploading from a texture (for cases where you already have a texture)
        let pixmap2 = ImageScene::read_cowboy_image();
        let texture2 =
            self.upload_image_to_texture(&device_handle.device, &device_handle.queue, &pixmap2);
        self.renderers[device_id].as_mut().unwrap().upload_image(
            &device_handle.device,
            &device_handle.queue,
            &mut encoder,
            &texture2,
        );

        device_handle.queue.submit([encoder.finish()]);
    }

    fn upload_image_to_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &Pixmap,
    ) -> wgpu::Texture {
        let image_width = image.width() as u32;
        let image_height = image.height() as u32;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Uploaded Image Texture"),
            size: wgpu::Extent3d {
                width: image_width,
                height: image_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            image.data_as_u8_slice(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 4 bytes per RGBA pixel
                bytes_per_row: Some(4 * image_width),
                rows_per_image: Some(image_height),
            },
            wgpu::Extent3d {
                width: image_width,
                height: image_height,
                depth_or_array_layers: 1,
            },
        );

        texture
    }
}
