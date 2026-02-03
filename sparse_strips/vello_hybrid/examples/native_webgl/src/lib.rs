// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Demonstrates using Vello with the WebGL2 renderer in the browser.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this demo"
)]
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;
use vello_common::paint::ImageId;
use vello_common::{
    kurbo::{Affine, Vec2},
    paint::ImageSource,
};
use vello_example_scenes::AnyScene;
use vello_example_scenes::image::ImageScene;
use vello_hybrid::Scene;
use wasm_bindgen::prelude::*;
use web_sys::{Event, HtmlCanvasElement, KeyboardEvent, MouseEvent, WheelEvent};

struct RendererWrapper {
    renderer: vello_hybrid::WebGlRenderer,
}

impl RendererWrapper {
    fn new(canvas: HtmlCanvasElement) -> Self {
        let renderer = vello_hybrid::WebGlRenderer::new(&canvas);

        Self { renderer }
    }
}

/// State that handles scene rendering and interactions
struct AppState {
    scenes: Box<[AnyScene<Scene>]>,
    current_scene: usize,
    scene: Scene,
    transform: Affine,
    mouse_down: bool,
    last_cursor_position: Option<Vec2>,
    width: u32,
    height: u32,
    renderer_wrapper: RendererWrapper,
    need_render: bool,
    canvas: HtmlCanvasElement,
}

impl AppState {
    fn new(canvas: HtmlCanvasElement, scenes: Box<[AnyScene<Scene>]>) -> Self {
        let width = canvas.width();
        let height = canvas.height();

        let renderer_wrapper = RendererWrapper::new(canvas.clone());

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

        self.renderer_wrapper
            .renderer
            .render(&self.scene, &render_size)
            .unwrap();
        self.need_render = false;
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.width = width;
        self.height = height;

        self.scene = Scene::new(width as u16, height as u16);

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
        self.last_cursor_position = Some(Vec2::new(x, y));
    }

    fn handle_mouse_up(&mut self) {
        self.mouse_down = false;
        self.last_cursor_position = None;
    }

    fn handle_mouse_move(&mut self, x: f64, y: f64) {
        let current_pos = Vec2::new(x, y);

        if self.mouse_down
            && let Some(last_pos) = self.last_cursor_position
        {
            let delta = current_pos - last_pos;
            self.transform = Affine::translate(delta) * self.transform;
            self.need_render = true;
        }

        self.last_cursor_position = Some(current_pos);
    }

    fn handle_wheel(&mut self, delta_y: f64) {
        const ZOOM_STEP: f64 = 0.1;

        if let Some(cursor_pos) = self.last_cursor_position {
            let zoom_factor = (1.0 + delta_y * ZOOM_STEP).max(0.1);

            // Zoom centered at cursor position
            self.transform = Affine::translate(cursor_pos)
                * Affine::scale(zoom_factor)
                * Affine::translate(-cursor_pos)
                * self.transform;

            self.need_render = true;
        } else {
            // If no cursor position is known, zoom centered on screen
            let center = Vec2::new(self.width as f64 / 2.0, self.height as f64 / 2.0);

            let zoom_factor = (1.0 + delta_y * ZOOM_STEP).max(0.1);

            self.transform = Affine::translate(center)
                * Affine::scale(zoom_factor)
                * Affine::translate(-center)
                * self.transform;

            self.need_render = true;
        }
    }

    /// Upload images to the WebGL atlas texture
    /// This is the WebGL analogue of the winit example's `upload_images_to_atlas` function
    fn upload_images_to_atlas(&mut self) {
        // 1st example — uploading pixmap directly to WebGL atlas
        let pixmap1 = ImageScene::read_flower_image();
        self.renderer_wrapper.renderer.upload_image(&pixmap1);

        // 2nd example — uploading from a WebGL texture
        let pixmap2 = ImageScene::read_cowboy_image();
        let texture2 = self.pixmap_to_webgl_texture(&pixmap2);
        self.renderer_wrapper.renderer.upload_image(&texture2);
    }

    /// Convert a pixmap to WebGL texture
    fn pixmap_to_webgl_texture(
        &self,
        pixmap: &vello_hybrid::Pixmap,
    ) -> vello_hybrid::WebGlTextureWithDimensions {
        let width = pixmap.width() as u32;
        let height = pixmap.height() as u32;
        let rgba_data = pixmap.data_as_u8_slice();

        let gl = &self.renderer_wrapper.renderer.gl_context();

        let texture = gl.create_texture().unwrap();
        gl.active_texture(web_sys::WebGl2RenderingContext::TEXTURE0);
        gl.bind_texture(web_sys::WebGl2RenderingContext::TEXTURE_2D, Some(&texture));

        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
            web_sys::WebGl2RenderingContext::TEXTURE_2D,
            0,
            web_sys::WebGl2RenderingContext::RGBA as i32,
            width as i32,
            height as i32,
            0,
            web_sys::WebGl2RenderingContext::RGBA,
            web_sys::WebGl2RenderingContext::UNSIGNED_BYTE,
            Some(rgba_data),
        )
        .unwrap();

        gl.tex_parameteri(
            web_sys::WebGl2RenderingContext::TEXTURE_2D,
            web_sys::WebGl2RenderingContext::TEXTURE_MIN_FILTER,
            web_sys::WebGl2RenderingContext::LINEAR as i32,
        );
        gl.tex_parameteri(
            web_sys::WebGl2RenderingContext::TEXTURE_2D,
            web_sys::WebGl2RenderingContext::TEXTURE_MAG_FILTER,
            web_sys::WebGl2RenderingContext::LINEAR as i32,
        );
        gl.tex_parameteri(
            web_sys::WebGl2RenderingContext::TEXTURE_2D,
            web_sys::WebGl2RenderingContext::TEXTURE_WRAP_S,
            web_sys::WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameteri(
            web_sys::WebGl2RenderingContext::TEXTURE_2D,
            web_sys::WebGl2RenderingContext::TEXTURE_WRAP_T,
            web_sys::WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
        );

        vello_hybrid::WebGlTextureWithDimensions {
            texture,
            width,
            height,
        }
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
    body.append_child(&canvas).unwrap();

    let scenes = {
        let v = vello_example_scenes::get_example_scenes(vec![
            ImageSource::OpaqueId(ImageId::new(0)),
            ImageSource::OpaqueId(ImageId::new(1)),
        ])
        .into_vec();
        v.into_boxed_slice()
    };

    let app_state = Rc::new(RefCell::new(AppState::new(canvas.clone(), scenes)));

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

    let mut renderer = vello_hybrid::WebGlRenderer::new(&canvas);

    let render_size = vello_hybrid::RenderSize {
        width: width as u32,
        height: height as u32,
    };

    renderer.render(&scene, &render_size).unwrap();
}
