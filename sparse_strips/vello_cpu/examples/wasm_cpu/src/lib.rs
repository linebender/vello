// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Demonstrates using Vello CPU in the browser.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this demo"
)]
#![cfg(target_arch = "wasm32")]

use std::rc::Rc;
use std::{cell::RefCell, sync::Arc};
use vello_common::kurbo::{Affine, Vec2};
use vello_common::paint::ImageSource;
use vello_cpu::RenderContext;
use vello_example_scenes::{AnyScene, image::ImageScene};
use wasm_bindgen::prelude::*;
use web_sys::{Event, HtmlCanvasElement, KeyboardEvent, MouseEvent, WheelEvent};

/// State that handles scene rendering and interactions
struct AppState {
    scenes: Box<[AnyScene<RenderContext>]>,
    current_scene: usize,
    transform: Affine,
    mouse_down: bool,
    last_cursor_position: Option<Vec2>,
    width: u32,
    height: u32,
    renderer: RenderContext,
    pixmap: vello_common::pixmap::Pixmap,
    need_render: bool,
    canvas: HtmlCanvasElement,
}

impl AppState {
    fn new(canvas: HtmlCanvasElement, scenes: Box<[AnyScene<RenderContext>]>) -> Self {
        let width = canvas.width();
        let height = canvas.height();

        let renderer = RenderContext::new_with(
            width as u16,
            height as u16,
            vello_cpu::RenderSettings {
                num_threads: 0,
                ..Default::default()
            },
        );

        let pixmap = vello_common::pixmap::Pixmap::new(width as u16, height as u16);

        Self {
            scenes,
            current_scene: 0,
            transform: Affine::IDENTITY,
            mouse_down: false,
            last_cursor_position: None,
            width,
            height,
            renderer,
            pixmap,
            need_render: true,
            canvas,
        }
    }

    fn render(&mut self) {
        if !self.need_render {
            return;
        }
        self.renderer.reset();
        self.scenes[self.current_scene].render(&mut self.renderer, self.transform);

        // Render the current scene with transform
        self.renderer.render_to_pixmap(&mut self.pixmap);
        let rgba_bytes = self.pixmap.data_as_u8_slice();
        let image_data = web_sys::ImageData::new_with_u8_clamped_array_and_sh(
            wasm_bindgen::Clamped(rgba_bytes),
            self.width,
            self.height,
        )
        .expect("Failed to create ImageData");
        let ctx_2d = self
            .canvas
            .get_context("2d")
            .expect("2D context should be available")
            .unwrap()
            .dyn_into::<web_sys::CanvasRenderingContext2d>()
            .unwrap();

        ctx_2d
            .put_image_data(&image_data, 0., 0.)
            .expect("Failed to put image data");
        self.need_render = false;
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.width = width;
        self.height = height;

        self.pixmap.resize(width as u16, height as u16);
        self.renderer = RenderContext::new_with(
            width as u16,
            height as u16,
            vello_cpu::RenderSettings {
                num_threads: 0,
                ..Default::default()
            },
        );

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

    let pixmap1 = ImageScene::read_flower_image();
    let pixmap2 = ImageScene::read_cowboy_image();
    let scenes = vello_example_scenes::get_example_scenes::<RenderContext>(vec![
        ImageSource::Pixmap(Arc::new(pixmap1)),
        ImageSource::Pixmap(Arc::new(pixmap2)),
    ]);

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
