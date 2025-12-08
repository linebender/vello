// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renders our example scenes with Vello CPU.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this demo"
)]

#[cfg(not(target_arch = "wasm32"))]
use std::env;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;
use vello_common::kurbo::{Affine, Point};
use vello_common::paint::ImageSource;
use vello_common::pixmap::Pixmap;
use vello_cpu::{RenderContext, RenderSettings};
use vello_example_scenes::image::ImageScene;
use vello_example_scenes::{AnyScene, get_example_scenes};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, Modifiers, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const ZOOM_STEP: f64 = 0.1;
const ROTATION_STEP: f64 = 0.1;
const SHEAR_STEP: f64 = 0.05;

struct App {
    scenes: Box<[AnyScene<RenderContext>]>,
    current_scene: usize,
    render_state: RenderState,
    renderer: RenderContext,
    pixmap: Pixmap,
    transform: Affine,
    mouse_down: bool,
    last_cursor_position: Option<Point>,
    last_frame_time: Option<Instant>,
    frame_count: u32,
    fps_update_time: Instant,
    accumulated_frame_time: f64,
    rotating: bool,
    rotation_speed: f64,
    shearing: bool,
    shear_speed: f64,
    shear_amplitude: f64,
    current_shear: f64,
    shear_direction: f64,
    modifiers: Modifiers,
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

        let pixmap1 = ImageScene::read_flower_image();
        let pixmap2 = ImageScene::read_cowboy_image();
        let img_sources = vec![
            ImageSource::Pixmap(std::sync::Arc::new(pixmap1)),
            ImageSource::Pixmap(std::sync::Arc::new(pixmap2)),
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
    let (scenes, start_scene_index) = {
        let pixmap1 = ImageScene::read_flower_image();
        let pixmap2 = ImageScene::read_cowboy_image();
        (
            get_example_scenes(vec![
                ImageSource::Pixmap(std::sync::Arc::new(pixmap1)),
                ImageSource::Pixmap(std::sync::Arc::new(pixmap2)),
            ]),
            0,
        )
    };

    let width = 1800;
    let height = 1200;
    let now = Instant::now();
    let mut app = App {
        scenes,
        current_scene: start_scene_index,
        render_state: RenderState::Suspended,
        renderer: RenderContext::new_with(
            width,
            height,
            RenderSettings {
                num_threads: 0, // 0 means use default (number of CPU cores)
                ..Default::default()
            },
        ),
        pixmap: Pixmap::new(width, height),
        transform: Affine::IDENTITY,
        mouse_down: false,
        last_cursor_position: None,
        last_frame_time: None,
        frame_count: 0,
        fps_update_time: now,
        accumulated_frame_time: 0.0,
        rotating: false,
        rotation_speed: 1.0,
        shearing: false,
        // shear rate (units of tan(angle)) per second
        shear_speed: 0.8,
        // maximum |shear| to oscillate between
        shear_amplitude: 0.35,
        current_shear: 0.0,
        // 1 for increasing toward +amplitude, -1 toward -amplitude
        shear_direction: 1.0,
        modifiers: Modifiers::default(),
    };

    let event_loop = EventLoop::new().unwrap();
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
}

enum RenderState {
    Active {
        window: Rc<Window>,
        surface: softbuffer::Surface<Rc<Window>, Rc<Window>>,
    },
    Suspended,
}

impl ApplicationHandler for App {
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.render_state = RenderState::Suspended;
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if matches!(self.render_state, RenderState::Active { .. }) {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.pixmap.width() as u32,
                self.pixmap.height() as u32,
            ))
            .with_resizable(true)
            .with_title("Vello CPU - Scene 0")
            .with_visible(true)
            .with_active(true);

        let window = Rc::new(event_loop.create_window(window_attrs).unwrap());

        let context = softbuffer::Context::new(window.clone()).unwrap();
        let surface = softbuffer::Surface::new(&context, window.clone()).unwrap();

        self.render_state = RenderState::Active { window, surface };
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let RenderState::Active { window, surface } = &mut self.render_state else {
            return;
        };

        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                let width = size.width.max(1);
                let height = size.height.max(1);

                surface
                    .resize(
                        NonZeroU32::new(width).unwrap(),
                        NonZeroU32::new(height).unwrap(),
                    )
                    .unwrap();

                self.pixmap.resize(width as u16, height as u16);
                self.renderer = RenderContext::new_with(
                    width as u16,
                    height as u16,
                    RenderSettings {
                        num_threads: 0,
                        ..Default::default()
                    },
                );

                window.request_redraw();
            }
            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers;
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                let is_cmd = self.modifiers.state().super_key();
                match logical_key {
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
                        if ch.as_str() == "r" {
                            if is_cmd {
                                // Cmd+r: Toggle continuous rotation around the window center
                                self.rotating = !self.rotating;
                                window.request_redraw();
                            } else {
                                // r: Single-step rotation around the window center
                                let center = Point {
                                    x: 0.5 * self.pixmap.width() as f64,
                                    y: 0.5 * self.pixmap.height() as f64,
                                };
                                self.transform =
                                    self.transform.then_rotate_about(ROTATION_STEP, center);
                                window.request_redraw();
                            }
                        } else if ch.as_str() == "R" {
                            // R: Counter-rotation step (opposite direction of r)
                            let center = Point {
                                x: 0.5 * self.pixmap.width() as f64,
                                y: 0.5 * self.pixmap.height() as f64,
                            };
                            self.transform =
                                self.transform.then_rotate_about(-ROTATION_STEP, center);
                            window.request_redraw();
                        } else if ch.as_str() == "s" {
                            if is_cmd {
                                // Cmd+s: Toggle shear oscillation around the window center
                                self.shearing = !self.shearing;
                                window.request_redraw();
                            } else {
                                // s: Single-step shear about the window center in X
                                let center = Point {
                                    x: 0.5 * self.pixmap.width() as f64,
                                    y: 0.5 * self.pixmap.height() as f64,
                                };
                                let about_center = Affine::translate((-center.x, -center.y))
                                    * Affine::skew(SHEAR_STEP, 0.0)
                                    * Affine::translate((center.x, center.y));
                                self.transform *= about_center;
                                window.request_redraw();
                            }
                        } else if ch.as_str() == "S" {
                            // S: Counter-shear step (opposite direction of s)
                            let center = Point {
                                x: 0.5 * self.pixmap.width() as f64,
                                y: 0.5 * self.pixmap.height() as f64,
                            };
                            let about_center = Affine::translate((-center.x, -center.y))
                                * Affine::skew(-SHEAR_STEP, 0.0)
                                * Affine::translate((center.x, center.y));
                            self.transform *= about_center;
                            window.request_redraw();
                        } else if let Some(scene) = self.scenes.get_mut(self.current_scene)
                            && scene.handle_key(ch.as_str())
                        {
                            window.request_redraw();
                        }
                    }
                    _ => {}
                }
            }
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
                        x: 0.5 * self.pixmap.width() as f64,
                        y: 0.5 * self.pixmap.height() as f64,
                    }),
                );

                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                // Measure frame time
                let now = Instant::now();
                let delta_s = self
                    .last_frame_time
                    .map(|t| now.duration_since(t).as_secs_f64())
                    .unwrap_or(0.0);
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
                            "Vello CPU - Scene {} - {:.1} FPS ({:.2}ms avg)",
                            self.current_scene, avg_fps, avg_frame_time
                        ));

                        // Reset counters
                        self.frame_count = 0;
                        self.accumulated_frame_time = 0.0;
                        self.fps_update_time = now;
                    }
                }
                self.last_frame_time = Some(now);

                // Apply rotation animation if enabled
                if self.rotating && delta_s > 0.0 {
                    let center = Point {
                        x: 0.5 * self.pixmap.width() as f64,
                        y: 0.5 * self.pixmap.height() as f64,
                    };
                    let angle = self.rotation_speed * delta_s;
                    self.transform = self.transform.then_rotate_about(angle, center);
                }

                // Apply shear oscillation if enabled (bounded back-and-forth)
                if self.shearing && delta_s > 0.0 {
                    let old = self.current_shear;
                    let mut new = old + self.shear_speed * delta_s * self.shear_direction;

                    if self.shear_direction > 0.0 && new > self.shear_amplitude {
                        let overshoot = new - self.shear_amplitude;
                        new = self.shear_amplitude - overshoot;
                        self.shear_direction = -1.0;
                    } else if self.shear_direction < 0.0 && new < -self.shear_amplitude {
                        let overshoot = -self.shear_amplitude - new;
                        new = -self.shear_amplitude + overshoot;
                        self.shear_direction = 1.0;
                    }

                    let delta_shear = new - old;
                    if delta_shear.abs() > 0.0 {
                        let center = Point {
                            x: 0.5 * self.pixmap.width() as f64,
                            y: 0.5 * self.pixmap.height() as f64,
                        };
                        // Shear about window center in X; Y shear remains 0.0
                        let about_center = Affine::translate((-center.x, -center.y))
                            * Affine::skew(delta_shear, 0.0)
                            * Affine::translate((center.x, center.y));
                        self.transform *= about_center;
                        self.current_shear = new;
                    }
                }

                // Render the scene
                self.renderer.reset();

                self.scenes[self.current_scene].render(&mut self.renderer, self.transform);
                self.renderer.flush();
                self.renderer.render_to_pixmap(&mut self.pixmap);

                // Copy pixmap to window surface
                let mut buffer = surface.buffer_mut().unwrap();
                let pixmap_data = self.pixmap.data();

                // Convert RGBA to BGRA/XRGB format expected by softbuffer
                for (buffer_pixel, pixel) in buffer.iter_mut().zip(pixmap_data.iter()) {
                    // softbuffer expects 0RGB format (little-endian: B, G, R, 0)
                    // Our pixmap is premultiplied RGBA
                    *buffer_pixel = u32::from_le_bytes([pixel.b, pixel.g, pixel.r, 0]);
                }

                buffer.present().unwrap();

                // Request continuous redraw for FPS measurement
                if self.rotating || self.shearing {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}
