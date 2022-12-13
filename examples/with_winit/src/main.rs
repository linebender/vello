// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

mod pico_svg;
mod simple_text;
mod test_scene;

use vello::{util::RenderContext, Renderer, Scene, SceneBuilder};
use winit::{event_loop::EventLoop, window::Window};

async fn run(event_loop: EventLoop<()>, window: Window) {
    use winit::{event::*, event_loop::ControlFlow};
    let mut render_cx = RenderContext::new().unwrap();
    let size = window.inner_size();
    let mut surface = render_cx
        .create_surface(&window, size.width, size.height)
        .await;
    let device_handle = &render_cx.devices[surface.dev_id];
    let mut renderer = Renderer::new(&device_handle.device).unwrap();
    let mut simple_text = simple_text::SimpleText::new();
    let mut current_frame = 0usize;
    let mut scene_ix = 0usize;
    let mut scene = Scene::new();
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::KeyboardInput { input, .. } => {
                if input.state == ElementState::Pressed {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::Left) => scene_ix = scene_ix.saturating_sub(1),
                        Some(VirtualKeyCode::Right) => scene_ix = scene_ix.saturating_add(1),
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                render_cx.resize_surface(&mut surface, size.width, size.height);
                window.request_redraw();
            }
            _ => {}
        },
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            current_frame += 1;
            let width = surface.config.width;
            let height = surface.config.height;
            let device_handle = &render_cx.devices[surface.dev_id];
            let mut builder = SceneBuilder::for_scene(&mut scene);
            const N_SCENES: usize = 6;
            match scene_ix % N_SCENES {
                0 => test_scene::render_anim_frame(&mut builder, &mut simple_text, current_frame),
                1 => test_scene::render_blend_grid(&mut builder),
                2 => test_scene::render_tiger(&mut builder),
                3 => test_scene::render_brush_transform(&mut builder, current_frame),
                4 => test_scene::render_funky_paths(&mut builder),
                _ => test_scene::render_scene(&mut builder),
            }
            builder.finish();
            let surface_texture = surface
                .surface
                .get_current_texture()
                .expect("failed to get surface texture");
            renderer
                .render_to_surface(
                    &device_handle.device,
                    &device_handle.queue,
                    &scene,
                    &surface_texture,
                    width,
                    height,
                )
                .expect("failed to render to surface");
            surface_texture.present();
            device_handle.device.poll(wgpu::Maintain::Wait);
        }
        _ => {}
    });
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use winit::{dpi::LogicalSize, window::WindowBuilder};
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_inner_size(LogicalSize::new(1044, 800))
            .with_resizable(true)
            .build(&event_loop)
            .unwrap();
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        let event_loop = EventLoop::new();
        let window = winit::window::Window::new(&event_loop).unwrap();

        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;

        // On wasm, append the canvas to the document body
        let canvas = window.canvas();
        canvas.set_width(1044);
        canvas.set_height(800);
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| body.append_child(&web_sys::Element::from(canvas)).ok())
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
