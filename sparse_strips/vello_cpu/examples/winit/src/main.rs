// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renders our example scenes with Vello CPU.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this demo"
)]

use parley_draw::renderers::vello_renderer::replay_atlas_commands;
use parley_draw::{
    AtlasConfig, CpuGlyphCaches, GlyphCache, GlyphCacheConfig, ImageCache, PendingClearRect,
};
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
use vello_example_scenes::{AnyScene, TextConfig, get_example_scenes};
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
    glyph_renderer: RenderContext,
    glyph_caches: CpuGlyphCaches,
    image_cache: ImageCache,
    text_config: TextConfig,
    pixmap: Pixmap,
    transform: Affine,
    mouse_down: bool,
    last_cursor_position: Option<Point>,
    last_frame_time: Option<Instant>,
    frame_count: u32,
    fps_update_time: Instant,
    accumulated_frame_time: f64,
    accumulated_render_time: f64,
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
        let mut start_scene_index = 1;
        let args: Vec<String> = env::args().collect();
        let mut svg_paths: Vec<&str> = Vec::new();

        if args.len() > 1 {
            if let Ok(index) = args[1].parse::<usize>() {
                start_scene_index = index;
            } else {
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
                num_threads: 0,
                ..Default::default()
            },
        ),
        glyph_renderer: RenderContext::new_with(
            512,
            512,
            RenderSettings {
                num_threads: 0,
                ..Default::default()
            },
        ),
        glyph_caches: CpuGlyphCaches::with_config(
            512,
            512,
            GlyphCacheConfig {
                max_entry_age: u32::MAX,
                eviction_frequency: u32::MAX,
            },
        ),
        image_cache: ImageCache::new_with_config(AtlasConfig {
            atlas_size: (512, 512),
            ..AtlasConfig::default()
        }),
        text_config: TextConfig::default(),
        pixmap: Pixmap::new(width, height),
        transform: Affine::IDENTITY,
        mouse_down: false,
        last_cursor_position: None,
        last_frame_time: None,
        frame_count: 0,
        fps_update_time: now,
        accumulated_frame_time: 0.0,
        accumulated_render_time: 0.0,
        rotating: false,
        rotation_speed: 1.0,
        shearing: false,
        shear_speed: 0.8,
        shear_amplitude: 0.35,
        current_shear: 0.0,
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
                        self.transform = Affine::IDENTITY;
                        window.request_redraw();
                    }
                    Key::Named(NamedKey::Escape) => {
                        event_loop.exit();
                    }
                    Key::Character(ch) => match ch.as_str() {
                        "a" | "A" => {
                            self.text_config.use_atlas_cache = !self.text_config.use_atlas_cache;
                            println!(
                                "Atlas cache: {}",
                                if self.text_config.use_atlas_cache {
                                    "ON"
                                } else {
                                    "OFF"
                                }
                            );
                            window.request_redraw();
                        }
                        "r" if is_cmd => {
                            self.rotating = !self.rotating;
                            window.request_redraw();
                        }
                        "r" => {
                            let center = Point {
                                x: 0.5 * self.pixmap.width() as f64,
                                y: 0.5 * self.pixmap.height() as f64,
                            };
                            self.transform =
                                self.transform.then_rotate_about(ROTATION_STEP, center);
                            window.request_redraw();
                        }
                        "R" => {
                            let center = Point {
                                x: 0.5 * self.pixmap.width() as f64,
                                y: 0.5 * self.pixmap.height() as f64,
                            };
                            self.transform =
                                self.transform.then_rotate_about(-ROTATION_STEP, center);
                            window.request_redraw();
                        }
                        "s" if is_cmd => {
                            self.shearing = !self.shearing;
                            window.request_redraw();
                        }
                        "s" => {
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
                        "S" => {
                            let center = Point {
                                x: 0.5 * self.pixmap.width() as f64,
                                y: 0.5 * self.pixmap.height() as f64,
                            };
                            let about_center = Affine::translate((-center.x, -center.y))
                                * Affine::skew(-SHEAR_STEP, 0.0)
                                * Affine::translate((center.x, center.y));
                            self.transform *= about_center;
                            window.request_redraw();
                        }
                        _ => {
                            if let Some(scene) = self.scenes.get_mut(self.current_scene)
                                && scene.handle_key(ch.as_str())
                            {
                                window.request_redraw();
                            }
                        }
                    },
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_down = state == ElementState::Pressed;
                    if !self.mouse_down {
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
                    if let Some(last_pos) = self.last_cursor_position {
                        self.transform = self.transform.then_translate(current_pos - last_pos);
                        window.request_redraw();
                    }
                }

                self.last_cursor_position = Some(current_pos);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(pos) => pos.y / 100.0,
                };

                if let Some(cursor_pos) = self.last_cursor_position {
                    let zoom_factor = (1.0 + delta_y * ZOOM_STEP).max(0.1);
                    self.transform = self.transform.then_scale_about(zoom_factor, cursor_pos);
                    window.request_redraw();
                }
            }
            WindowEvent::PinchGesture { delta, .. } => {
                let zoom_factor = 1.0 + delta * ZOOM_STEP * 5.0;
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
                let now = Instant::now();
                let delta_s = self
                    .last_frame_time
                    .map(|t| now.duration_since(t).as_secs_f64())
                    .unwrap_or(0.0);
                if let Some(last_time) = self.last_frame_time {
                    let frame_time = now.duration_since(last_time).as_secs_f64() * 1000.0;
                    self.accumulated_frame_time += frame_time;
                    self.frame_count += 1;

                    if now.duration_since(self.fps_update_time).as_secs_f64() >= 1.0 {
                        let avg_frame_time = self.accumulated_frame_time / self.frame_count as f64;
                        let avg_fps = 1000.0 / avg_frame_time;
                        let avg_render_time =
                            self.accumulated_render_time / self.frame_count as f64;
                        let status = self.scenes[self.current_scene]
                            .status()
                            .map(|s| format!(" - {s}"))
                            .unwrap_or_default();
                        println!(
                            "FPS: {avg_fps:.1} | render: {avg_render_time:.2}ms | frame: {avg_frame_time:.2}ms{status}"
                        );
                        window.set_title(&format!(
                            "Vello CPU - Scene {} - {:.1} FPS (render {:.2}ms){status}",
                            self.current_scene, avg_fps, avg_render_time
                        ));

                        self.frame_count = 0;
                        self.accumulated_frame_time = 0.0;
                        self.accumulated_render_time = 0.0;
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

                // Apply shear oscillation if enabled
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
                        let about_center = Affine::translate((-center.x, -center.y))
                            * Affine::skew(delta_shear, 0.0)
                            * Affine::translate((center.x, center.y));
                        self.transform *= about_center;
                        self.current_shear = new;
                    }
                }

                let render_start = Instant::now();

                // Render the scene
                self.renderer.reset();
                self.renderer.set_transform(self.transform);
                self.scenes[self.current_scene].render(
                    &mut self.renderer,
                    self.transform,
                    &mut self.glyph_caches,
                    &mut self.image_cache,
                    &self.text_config,
                );

                // Replay outline/COLR draw commands into each atlas page's pixmap.
                for mut recorder in self.glyph_caches.glyph_atlas.take_pending_atlas_commands() {
                    self.glyph_renderer.reset();
                    replay_atlas_commands(&mut recorder.commands, &mut self.glyph_renderer);
                    self.glyph_renderer.flush();
                    if let Some(atlas_pixmap) = self
                        .glyph_caches
                        .glyph_atlas
                        .page_pixmap_mut(recorder.page_index as usize)
                    {
                        self.glyph_renderer
                            .composite_to_pixmap_at_offset(atlas_pixmap, 0, 0);
                    }
                }

                // Upload bitmap glyphs to atlas pages.
                for upload in self.glyph_caches.glyph_atlas.take_pending_uploads() {
                    let page_index = upload.atlas_slot.page_index as usize;
                    if let Some(atlas_pixmap) =
                        self.glyph_caches.glyph_atlas.page_pixmap_mut(page_index)
                    {
                        copy_pixmap_to_atlas(
                            &upload.pixmap,
                            atlas_pixmap,
                            upload.atlas_slot.x,
                            upload.atlas_slot.y,
                            upload.atlas_slot.width,
                            upload.atlas_slot.height,
                        );
                    }
                }

                // Share atlas page pixmaps with the renderer.
                let page_count = self.glyph_caches.glyph_atlas.page_count();
                for page_index in 0..page_count {
                    if let Some(page_pixmap) = self.glyph_caches.glyph_atlas.page_pixmap(page_index)
                    {
                        self.renderer.register_image(page_pixmap.clone());
                    }
                }

                self.renderer.flush();
                self.renderer.render_to_pixmap(&mut self.pixmap);
                self.renderer.clear_images();

                // Maintain caches (eviction, etc.)
                self.glyph_caches.maintain(&mut self.image_cache);

                // Clear stale atlas regions after eviction.
                for rect in self.glyph_caches.glyph_atlas.take_pending_clear_rects() {
                    if let Some(atlas_pixmap) = self
                        .glyph_caches
                        .glyph_atlas
                        .page_pixmap_mut(rect.page_index as usize)
                    {
                        clear_pixmap_region(atlas_pixmap, &rect);
                    }
                }

                self.accumulated_render_time += render_start.elapsed().as_secs_f64() * 1000.0;

                // Copy pixmap to window surface
                let mut buffer = surface.buffer_mut().unwrap();
                let pixmap_data = self.pixmap.data();

                for (buffer_pixel, pixel) in buffer.iter_mut().zip(pixmap_data.iter()) {
                    *buffer_pixel = u32::from_le_bytes([pixel.b, pixel.g, pixel.r, 0]);
                }

                buffer.present().unwrap();

                // Request continuous redraw for FPS measurement
                window.request_redraw();
            }
            _ => {}
        }
    }
}

/// Zero out a rectangular region in the atlas pixmap.
fn clear_pixmap_region(dst: &mut Pixmap, rect: &PendingClearRect) {
    let dst_stride = dst.width() as usize;
    let dst_data = dst.data_as_u8_slice_mut();
    let clear_width = rect.width as usize;
    let clear_height = rect.height as usize;

    for y in 0..clear_height {
        let row_start = ((rect.y as usize + y) * dst_stride + rect.x as usize) * 4;
        let row_end = row_start + clear_width * 4;
        dst_data[row_start..row_end].fill(0);
    }
}

/// Copy bitmap glyph pixels into a rectangular region of an atlas page.
fn copy_pixmap_to_atlas(
    src: &Pixmap,
    dst: &mut Pixmap,
    dst_x: u16,
    dst_y: u16,
    width: u16,
    height: u16,
) {
    let copy_width = width as usize;
    let copy_height = height as usize;
    let src_stride = src.width() as usize;
    let dst_stride = dst.width() as usize;

    let src_data = src.data_as_u8_slice();
    let dst_data = dst.data_as_u8_slice_mut();

    for y in 0..copy_height {
        let src_row_start = y * src_stride * 4;
        let src_row_end = src_row_start + copy_width * 4;
        let dst_row_start = ((dst_y as usize + y) * dst_stride + dst_x as usize) * 4;
        let dst_row_end = dst_row_start + copy_width * 4;

        dst_data[dst_row_start..dst_row_end].copy_from_slice(&src_data[src_row_start..src_row_end]);
    }
}
