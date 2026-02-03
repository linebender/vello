// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Demonstrates using Vello Hybrid using a WebGL2 backend in the browser.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this demo"
)]
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;
use vello_common::{
    fearless_simd::Level,
    kurbo::{Affine, Point},
    paint::{ImageId, ImageSource},
};
use vello_example_scenes::{AnyScene, image::ImageScene};
use vello_hybrid::{AtlasConfig, Pixmap, RenderSettings, RenderTargetConfig, Renderer, Scene};
use wasm_bindgen::prelude::*;
use web_sys::{Event, HtmlCanvasElement, KeyboardEvent, MouseEvent, WheelEvent};

struct RendererWrapper {
    renderer: Renderer,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
}

impl RendererWrapper {
    async fn new(canvas: HtmlCanvasElement) -> Self {
        let width = canvas.width();
        let height = canvas.height();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .expect("Canvas surface to be valid");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("Adapter to be valid");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_texture_dimension_2d: adapter.limits().max_texture_dimension_2d,
                    max_buffer_size: adapter.limits().max_buffer_size,
                    ..wgpu::Limits::downlevel_webgl2_defaults()
                },
                ..Default::default()
            })
            .await
            .expect("Device to be valid");

        // Configure the surface
        let surface_format = wgpu::TextureFormat::Rgba8Unorm;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let max_texture_dimension_2d = device.limits().max_texture_dimension_2d;
        let renderer = Renderer::new_with(
            &device,
            &RenderTargetConfig {
                format: surface_format,
                width,
                height,
            },
            RenderSettings {
                level: Level::try_detect().unwrap_or(Level::fallback()),
                atlas_config: AtlasConfig {
                    atlas_size: (max_texture_dimension_2d, max_texture_dimension_2d),
                    ..AtlasConfig::default()
                },
            },
        );

        Self {
            renderer,
            device,
            queue,
            surface,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };
        self.surface.configure(&self.device, &surface_config);
    }
}

/// State that handles scene rendering and interactions
struct AppState {
    scenes: Box<[AnyScene<Scene>]>,
    current_scene: usize,
    scene: Scene,
    transform: Affine,
    mouse_down: bool,
    last_cursor_position: Option<Point>,
    width: u32,
    height: u32,
    renderer_wrapper: RendererWrapper,
    need_render: bool,
    canvas: HtmlCanvasElement,
}

impl AppState {
    async fn new(canvas: HtmlCanvasElement, scenes: Box<[AnyScene<Scene>]>) -> Self {
        let width = canvas.width();
        let height = canvas.height();

        let renderer_wrapper = RendererWrapper::new(canvas.clone()).await;

        let mut app_state = Self {
            scenes,
            current_scene: 0,
            scene: Scene::new(width as u16, height as u16),
            transform: Affine::IDENTITY,
            mouse_down: false,
            last_cursor_position: None,
            width,
            height,
            renderer_wrapper,
            need_render: true,
            canvas,
        };

        // Upload images to the WebGL atlas
        app_state.upload_images_to_atlas();

        app_state
    }

    fn render(&mut self) {
        if !self.need_render {
            return;
        }

        self.scene.reset();

        // Render the current scene with transform
        self.scenes[self.current_scene].render(&mut self.scene, self.transform);

        let render_size = vello_hybrid::RenderSize {
            width: self.width,
            height: self.height,
        };

        let surface_texture = self.renderer_wrapper.surface.get_current_texture().unwrap();
        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .renderer_wrapper
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.renderer_wrapper
            .renderer
            .render(
                &self.scene,
                &self.renderer_wrapper.device,
                &self.renderer_wrapper.queue,
                &mut encoder,
                &render_size,
                &surface_texture_view,
            )
            .unwrap();

        self.renderer_wrapper.queue.submit([encoder.finish()]);
        surface_texture.present();

        self.need_render = false;
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.width = width;
        self.height = height;

        self.scene = Scene::new(width as u16, height as u16);
        self.renderer_wrapper.resize(width, height);

        self.need_render = true;
    }

    fn next_scene(&mut self) {
        self.current_scene = (self.current_scene + 1) % self.scenes.len();
        self.transform = Affine::IDENTITY;
        self.need_render = true;
    }

    fn prev_scene(&mut self) {
        self.current_scene = if self.current_scene == 0 {
            self.scenes.len() - 1
        } else {
            self.current_scene - 1
        };
        self.transform = Affine::IDENTITY;
        self.need_render = true;
    }

    fn reset_transform(&mut self) {
        self.transform = Affine::IDENTITY;
        self.need_render = true;
    }

    fn handle_mouse_down(&mut self, x: f64, y: f64) {
        self.mouse_down = true;
        self.last_cursor_position = Some(Point { x, y });
    }

    fn handle_mouse_up(&mut self) {
        self.mouse_down = false;
        self.last_cursor_position = None;
    }

    fn handle_mouse_move(&mut self, x: f64, y: f64) {
        let current_pos = Point { x, y };

        if self.mouse_down
            && let Some(last_pos) = self.last_cursor_position
        {
            self.transform = self.transform.then_translate(current_pos - last_pos);
            self.need_render = true;
        }

        self.last_cursor_position = Some(current_pos);
    }

    fn handle_wheel(&mut self, delta_y: f64) {
        const ZOOM_STEP: f64 = 0.1;
        let zoom_factor = (1.0 + delta_y * ZOOM_STEP).max(0.1);

        // Zoom centered at cursor position, or the center if no position is set.
        self.transform = self.transform.then_scale_about(
            zoom_factor,
            self.last_cursor_position.unwrap_or(Point {
                x: 0.5 * self.width as f64,
                y: 0.5 * self.height as f64,
            }),
        );

        self.need_render = true;
    }

    fn upload_images_to_atlas(&mut self) {
        let mut encoder =
            self.renderer_wrapper
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Upload Image pass"),
                });

        // 1st example — uploading pixmap directly to WebGL atlas
        let pixmap1 = ImageScene::read_flower_image();
        self.renderer_wrapper.renderer.upload_image(
            &self.renderer_wrapper.device,
            &self.renderer_wrapper.queue,
            &mut encoder,
            &pixmap1,
        );

        // 2nd example — uploading from a WebGL texture
        let pixmap2 = ImageScene::read_cowboy_image();
        let texture2 = self.upload_image_to_texture(
            &self.renderer_wrapper.device,
            &self.renderer_wrapper.queue,
            &pixmap2,
        );
        self.renderer_wrapper.renderer.upload_image(
            &self.renderer_wrapper.device,
            &self.renderer_wrapper.queue,
            &mut encoder,
            &texture2,
        );

        self.renderer_wrapper.queue.submit([encoder.finish()]);
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

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = requestAnimationFrame)]
    fn request_animation_frame(f: &Closure<dyn FnMut()>);
}

/// Creates a `HTMLCanvasElement` of the given dimensions and renders the given scenes into it,
/// with interactive controls for panning, zooming, and switching between scenes.
pub async fn run_interactive(canvas_width: u16, canvas_height: u16) {
    let canvas = web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(canvas_width as u32);
    canvas.set_height(canvas_height as u32);
    canvas.style().set_property("width", "100%").unwrap();
    canvas.style().set_property("height", "100%").unwrap();

    let body = web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .body()
        .unwrap();
    // Apply background color so white text can be seen.
    body.style()
        .set_property("background-color", "#111")
        .unwrap();

    // Add canvas to body
    web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .body()
        .unwrap()
        .append_child(&canvas)
        .unwrap();

    let scenes = vello_example_scenes::get_example_scenes(vec![
        ImageSource::OpaqueId(ImageId::new(0)),
        ImageSource::OpaqueId(ImageId::new(1)),
    ]);

    let app_state = Rc::new(RefCell::new(AppState::new(canvas.clone(), scenes).await));

    // Set up animation frame loop
    {
        let f = Rc::new(RefCell::new(None));
        let g = f.clone();
        let app_state = app_state.clone();

        *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            app_state.borrow_mut().render();
            request_animation_frame(f.borrow().as_ref().unwrap());
        }) as Box<dyn FnMut()>));

        request_animation_frame(g.borrow().as_ref().unwrap());
    }

    // Set up window resize event handler
    {
        let app_state = app_state.clone();
        let closure = Closure::wrap(Box::new(move |_: Event| {
            let window = web_sys::window().unwrap();
            let dpr = window.device_pixel_ratio();

            let width = window.inner_width().unwrap().as_f64().unwrap() as u32 * dpr as u32;
            let height = window.inner_height().unwrap().as_f64().unwrap() as u32 * dpr as u32;

            app_state.borrow_mut().resize(width, height);
        }) as Box<dyn FnMut(_)>);

        let window = web_sys::window().unwrap();
        window
            .add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Set up event handlers

    // Mouse down
    {
        let app_state = app_state.clone();
        let closure = Closure::wrap(Box::new(move |event: MouseEvent| {
            app_state
                .borrow_mut()
                .handle_mouse_down(event.client_x() as f64, event.client_y() as f64);
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Mouse up
    {
        let app_state = app_state.clone();
        let closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
            app_state.borrow_mut().handle_mouse_up();
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("mouseup", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Mouse move
    {
        let app_state = app_state.clone();
        let closure = Closure::wrap(Box::new(move |event: MouseEvent| {
            app_state
                .borrow_mut()
                .handle_mouse_move(event.client_x() as f64, event.client_y() as f64);
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Mouse wheel
    {
        let app_state = app_state.clone();
        let closure = Closure::wrap(Box::new(move |event: WheelEvent| {
            event.prevent_default();
            let delta = -event.delta_y() / 100.0; // Normalize and invert
            app_state.borrow_mut().handle_wheel(delta);
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("wheel", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Keyboard events (document level)
    {
        let app_state = app_state.clone();
        let document = web_sys::window().unwrap().document().unwrap();
        let closure =
            Closure::wrap(
                Box::new(move |event: KeyboardEvent| match event.key().as_str() {
                    "ArrowRight" => app_state.borrow_mut().next_scene(),
                    "ArrowLeft" => app_state.borrow_mut().prev_scene(),
                    " " => app_state.borrow_mut().reset_transform(),
                    _ => {}
                }) as Box<dyn FnMut(_)>,
            );
        document
            .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Create instructions element
    let document = web_sys::window().unwrap().document().unwrap();
    let instructions = document.create_element("div").unwrap();
    instructions.set_inner_html(
        "Left/Right Arrow: Change scene | Space: Reset view | Mouse Drag: Pan | Mouse Wheel: Zoom",
    );
    let style = instructions
        .dyn_ref::<web_sys::HtmlElement>()
        .unwrap()
        .style();
    style.set_property("position", "fixed").unwrap();
    style.set_property("bottom", "10px").unwrap();
    style.set_property("left", "10px").unwrap();
    style
        .set_property("background", "rgba(0, 0, 0, 0.5)")
        .unwrap();
    style.set_property("color", "white").unwrap();
    style.set_property("padding", "5px 10px").unwrap();
    style.set_property("border-radius", "5px").unwrap();
    style.set_property("font-family", "sans-serif").unwrap();
    style.set_property("pointer-events", "none").unwrap();

    document
        .body()
        .unwrap()
        .append_child(&instructions)
        .unwrap();
}

/// Creates a `HTMLCanvasElement` and renders a single scene into it
pub async fn render_scene(scene: Scene, width: u16, height: u16) {
    let canvas = web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(width as u32);
    canvas.set_height(height as u32);
    canvas.style().set_property("width", "100%").unwrap();
    canvas.style().set_property("height", "100%").unwrap();

    // Add canvas to body
    web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .body()
        .unwrap()
        .append_child(&canvas)
        .unwrap();

    let RendererWrapper {
        mut renderer,
        device,
        queue,
        surface,
    } = RendererWrapper::new(canvas).await;

    let render_size = vello_hybrid::RenderSize {
        width: width as u32,
        height: height as u32,
    };
    let surface_texture = surface.get_current_texture().unwrap();
    let surface_texture_view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    renderer
        .render(
            &scene,
            &device,
            &queue,
            &mut encoder,
            &render_size,
            &surface_texture_view,
        )
        .unwrap();

    queue.submit([encoder.finish()]);
    surface_texture.present();
}
