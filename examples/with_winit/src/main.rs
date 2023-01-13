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

use std::{borrow::Cow, path::PathBuf, time::Instant};

use clap::Parser;
use vello::{
    block_on_wgpu,
    kurbo::{Affine, Vec2},
    util::RenderContext,
    Renderer, Scene, SceneBuilder,
};
use winit::{
    event_loop::{EventLoop, EventLoopBuilder},
    window::Window,
};

#[cfg(not(target_arch = "wasm32"))]
mod hot_reload;

#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// Path to the svg file to render. If not set, the GhostScript Tiger will be rendered
    #[arg(long)]
    #[cfg(not(target_arch = "wasm32"))]
    svg: Option<std::path::PathBuf>,
    /// When rendering an svg, what scale to use
    #[arg(long)]
    scale: Option<f64>,
    /// Which scene (index) to start on
    /// Switch between scenes with left and right arrow keys
    #[arg(long)]
    scene: Option<i32>,
}

const TIGER: &'static str = include_str!("../../assets/Ghostscript_Tiger.svg");

async fn run(event_loop: EventLoop<UserEvent>, window: Window, args: Args) {
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
    let mut scene = Scene::new();
    let mut cached_svg_scene = None;
    let mut drag = Vec2::default();
    let mut scale = 1f64;
    let mut mouse_down = false;
    let mut prior_position = None;
    let mut svg_static_scale = 1.0;
    // We allow looping left and right through the scenes, so use a signed index
    let mut scene_ix: i32 = 0;
    #[cfg(not(target_arch = "wasm32"))]
    let svg_string: Cow<'static, str> = match args.svg {
        Some(path) => {
            // If an svg file has been specified, show that by default
            scene_ix = 2;
            let start = std::time::Instant::now();
            eprintln!("Reading svg from {path:?}");
            let svg = std::fs::read_to_string(path)
                .expect("Provided path did not point to a file which could be read")
                .into();
            eprintln!("Finished reading svg, took {:?}", start.elapsed());
            svg
        }
        None => {
            svg_static_scale = 6.0;
            TIGER.into()
        }
    };
    #[cfg(target_arch = "wasm32")]
    let svg_string: Cow<'static, str> = TIGER.into();
    // These are set after choosing the svg, as they overwrite the defaults specified there
    if let Some(set_scene) = args.scene {
        scene_ix = set_scene;
    }
    if let Some(set_scale) = args.scale {
        svg_static_scale = set_scale;
    }
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
                        Some(VirtualKeyCode::Escape) => {
                            *control_flow = ControlFlow::Exit;
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                render_cx.resize_surface(&mut surface, size.width, size.height);
                window.request_redraw();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == &MouseButton::Left {
                    mouse_down = state == &ElementState::Pressed;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let MouseScrollDelta::PixelDelta(delta) = delta {
                    scale += delta.y * 0.1;
                    scale = scale.clamp(0.1, 10.0);
                }
                if let MouseScrollDelta::LineDelta(_, y) = delta {
                    scale += *y as f64 * 0.1;
                    scale = scale.clamp(0.1, 10.0);
                }
            }
            WindowEvent::CursorLeft { .. } => {
                prior_position = None;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = Vec2::new(position.x, position.y);
                if mouse_down {
                    if let Some(prior) = prior_position {
                        drag += (position - prior) * (1.0 / scale);
                    }
                }
                prior_position = Some(position);
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

            const N_SCENES: i32 = 6;
            // Allow looping forever
            scene_ix = scene_ix.rem_euclid(N_SCENES);
            // Remainder operation allows negative results, which isn't the right semantics
            match scene_ix {
                0 => test_scene::render_anim_frame(&mut builder, &mut simple_text, current_frame),
                1 => test_scene::render_blend_grid(&mut builder),
                2 => {
                    let transform = Affine::scale(scale) * Affine::translate(drag);
                    test_scene::render_svg_scene(
                        &mut builder,
                        &mut cached_svg_scene,
                        transform,
                        &svg_string,
                        svg_static_scale,
                    )
                }
                3 => test_scene::render_brush_transform(&mut builder, current_frame),
                4 => test_scene::render_funky_paths(&mut builder),
                5 => test_scene::render_scene(&mut builder),
                _ => unreachable!("N_SCENES is too large"),
            }
            builder.finish();
            let surface_texture = surface
                .surface
                .get_current_texture()
                .expect("failed to get surface texture");
            #[cfg(not(target_arch = "wasm32"))]
            {
                block_on_wgpu(
                    &device_handle.device,
                    renderer.render_to_surface_async(
                        &device_handle.device,
                        &device_handle.queue,
                        &scene,
                        &surface_texture,
                        width,
                        height,
                    ),
                )
                .expect("failed to render to surface");
            }
            // Note: in the wasm case, we're currently not running the robust
            // pipeline, as it requires more async wiring for the readback.
            #[cfg(target_arch = "wasm32")]
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
            device_handle.device.poll(wgpu::Maintain::Poll);
        }
        Event::UserEvent(event) => match event {
            #[cfg(not(target_arch = "wasm32"))]
            UserEvent::HotReload => {
                let device_handle = &render_cx.devices[surface.dev_id];
                eprintln!("==============\nReloading shaders");
                let start = Instant::now();
                let result = renderer.reload_shaders(&device_handle.device);
                // We know that the only async here is actually sync, so we just block
                match pollster::block_on(result) {
                    Ok(_) => eprintln!("Reloading took {:?}", start.elapsed()),
                    Err(e) => eprintln!("Failed to reload shaders because of {e}"),
                }
            }
        },
        _ => {}
    });
}

enum UserEvent {
    #[cfg(not(target_arch = "wasm32"))]
    HotReload,
}

fn main() {
    let args = Args::parse();
    // TODO: initializing both env_logger and console_logger fails on wasm.
    // Figure out a more principled approach.
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    #[cfg(not(target_arch = "wasm32"))]
    {
        use winit::{dpi::LogicalSize, window::WindowBuilder};
        let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();

        let proxy = event_loop.create_proxy();
        let _keep =
            hot_reload::hot_reload(move || proxy.send_event(UserEvent::HotReload).ok().map(drop));

        let window = WindowBuilder::new()
            .with_inner_size(LogicalSize::new(1044, 800))
            .with_resizable(true)
            .with_title("Vello demo")
            .build(&event_loop)
            .unwrap();
        pollster::block_on(run(event_loop, window, args));
    }
    #[cfg(target_arch = "wasm32")]
    {
        let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
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
        wasm_bindgen_futures::spawn_local(run(event_loop, window, args));
    }
}
