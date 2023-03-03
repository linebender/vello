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

use std::time::Instant;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use scenes::{SceneParams, SceneSet, SimpleText};
use vello::SceneFragment;
use vello::{
    block_on_wgpu,
    kurbo::{Affine, Vec2},
    peniko::Color,
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
#[command(about, long_about = None, bin_name="cargo run -p with_winit --")]
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
    /// The base color used as the
    #[command(flatten)]
    args: scenes::Arguments,
}

async fn run(
    event_loop: EventLoop<UserEvent>,
    window: Window,
    args: Args,
    base_color_arg: Option<Color>,
    mut scenes: SceneSet,
) {
    use winit::{event::*, event_loop::ControlFlow};
    let mut render_cx = RenderContext::new().unwrap();
    let size = window.inner_size();
    let mut surface = render_cx
        .create_surface(&window, size.width, size.height)
        .await;
    let device_handle = &render_cx.devices[surface.dev_id];
    let mut renderer = Renderer::new(&device_handle.device).unwrap();
    let mut scene = Scene::new();
    let mut fragment = SceneFragment::new();
    let mut simple_text = SimpleText::new();
    let start = Instant::now();

    let mut transform = Affine::IDENTITY;
    let mut mouse_down = false;
    let mut prior_position: Option<Vec2> = None;
    // We allow looping left and right through the scenes, so use a signed index
    let mut scene_ix: i32 = 0;
    if let Some(set_scene) = args.scene {
        scene_ix = set_scene;
    }
    let mut prev_scene_ix = scene_ix - 1;
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
                        Some(key @ VirtualKeyCode::Q) | Some(key @ VirtualKeyCode::E) => {
                            if let Some(prior_position) = prior_position {
                                let is_clockwise = key == VirtualKeyCode::E;
                                let angle = if is_clockwise { -0.05 } else { 0.05 };
                                transform = Affine::translate(prior_position)
                                    * Affine::rotate(angle)
                                    * Affine::translate(-prior_position)
                                    * transform;
                            }
                        }
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
                const BASE: f64 = 1.05;
                const PIXELS_PER_LINE: f64 = 20.0;

                if let Some(prior_position) = prior_position {
                    let exponent = if let MouseScrollDelta::PixelDelta(delta) = delta {
                        delta.y / PIXELS_PER_LINE
                    } else if let MouseScrollDelta::LineDelta(_, y) = delta {
                        *y as f64
                    } else {
                        0.0
                    };
                    transform = Affine::translate(prior_position)
                        * Affine::scale(BASE.powf(exponent))
                        * Affine::translate(-prior_position)
                        * transform;
                } else {
                    eprintln!("Scrolling without mouse in window; this shouldn't be possible");
                }
            }
            WindowEvent::CursorLeft { .. } => {
                prior_position = None;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let position = Vec2::new(position.x, position.y);
                if mouse_down {
                    if let Some(prior) = prior_position {
                        transform = Affine::translate(position - prior) * transform;
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
            let width = surface.config.width;
            let height = surface.config.height;
            let device_handle = &render_cx.devices[surface.dev_id];

            // Allow looping forever
            scene_ix = scene_ix.rem_euclid(scenes.scenes.len() as i32);
            let example_scene = &mut scenes.scenes[scene_ix as usize];
            if prev_scene_ix != scene_ix {
                transform = Affine::IDENTITY;
                prev_scene_ix = scene_ix;
                window.set_title(&format!("Vello demo - {}", example_scene.config.name));
            }
            let mut builder = SceneBuilder::for_fragment(&mut fragment);
            let mut scene_params = SceneParams {
                time: start.elapsed().as_secs_f64(),
                text: &mut simple_text,
                resolution: None,
                base_color: None,
            };
            (example_scene.function)(&mut builder, &mut scene_params);
            builder.finish();

            // If the user specifies a base color in the CLI we use that. Otherwise we use any
            // color specified by the scene. The default is black.
            let render_params = vello::RenderParams {
                base_color: base_color_arg
                    .or(scene_params.base_color)
                    .unwrap_or(Color::BLACK),
                width,
                height,
            };
            let mut builder = SceneBuilder::for_scene(&mut scene);
            let mut transform = transform;
            if let Some(resolution) = scene_params.resolution {
                let factor = Vec2::new(surface.config.width as f64, surface.config.height as f64);
                let scale_factor = (factor.x / resolution.x).min(factor.y / resolution.y);
                transform = transform * Affine::scale(scale_factor);
            }
            builder.append(&fragment, Some(transform));
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
                        &render_params,
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
                    &render_params,
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

fn main() -> Result<()> {
    // TODO: initializing both env_logger and console_logger fails on wasm.
    // Figure out a more principled approach.
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let args = Args::parse();
    let scenes = args.args.select_scene_set(|| Args::command())?;
    let base_color = args.args.get_base_color()?;
    if let Some(scenes) = scenes {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use winit::{dpi::LogicalSize, window::WindowBuilder};
            let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();

            let proxy = event_loop.create_proxy();
            let _keep = hot_reload::hot_reload(move || {
                proxy.send_event(UserEvent::HotReload).ok().map(drop)
            });

            let window = WindowBuilder::new()
                .with_inner_size(LogicalSize::new(1044, 800))
                .with_resizable(true)
                .with_title("Vello demo")
                .build(&event_loop)
                .unwrap();
            pollster::block_on(run(event_loop, window, args, base_color, scenes));
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
            wasm_bindgen_futures::spawn_local(run(event_loop, window, args, base_color, scenes));
        }
    }
    Ok(())
}
