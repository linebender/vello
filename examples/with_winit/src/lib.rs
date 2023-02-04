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

use std::collections::HashSet;
use std::time::Instant;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use scenes::{SceneParams, SceneSet, SimpleText};
use vello::util::RenderSurface;
use vello::SceneFragment;
use vello::{
    kurbo::{Affine, Vec2},
    util::RenderContext,
    Renderer, Scene, SceneBuilder,
};

use winit::{
    event_loop::{EventLoop, EventLoopBuilder},
    window::Window,
};

#[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
mod hot_reload;
mod multi_touch;

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
    #[command(flatten)]
    args: scenes::Arguments,
}

struct RenderState {
    window: Window,
    surface: RenderSurface,
}

fn run(
    event_loop: EventLoop<UserEvent>,
    args: Args,
    mut scenes: SceneSet,
    render_cx: RenderContext,
    #[cfg(target_arch = "wasm32")] render_state: RenderState,
) {
    use winit::{event::*, event_loop::ControlFlow};
    let mut renderers: Vec<Option<Renderer>> = vec![];
    #[cfg(not(target_arch = "wasm32"))]
    let mut render_cx = render_cx;
    #[cfg(not(target_arch = "wasm32"))]
    let mut render_state = None::<RenderState>;
    #[cfg(target_arch = "wasm32")]
    let mut render_state = {
        renderers.resize_with(render_cx.devices.len(), || None);
        let id = render_state.surface.dev_id;
        renderers[id] = Some(
            Renderer::new(&render_cx.devices[id].device, render_state.surface.format)
                .expect("Could create renderer"),
        );
        Some(render_state)
    };
    let mut scene = Scene::new();
    let mut fragment = SceneFragment::new();
    let mut simple_text = SimpleText::new();
    let start = Instant::now();

    let mut touch_state = multi_touch::TouchState::new();
    let mut navigation_fingers = HashSet::new();

    let mut transform = Affine::IDENTITY;
    let mut mouse_down = false;
    let mut prior_position: Option<Vec2> = None;
    // We allow looping left and right through the scenes, so use a signed index
    let mut scene_ix: i32 = 0;
    if let Some(set_scene) = args.scene {
        scene_ix = set_scene;
    }
    let mut prev_scene_ix = scene_ix - 1;
    #[allow(unused)]
    let proxy = event_loop.create_proxy();
    // _event_loop is used on non-wasm platforms
    event_loop.run(move |event, _event_loop, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } => {
            let Some(render_state) = &mut render_state else { return };
            if render_state.window.id() != window_id {
                return;
            }
            match event {
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
                WindowEvent::Touch(touch) => {
                    if touch.location.y > render_state.surface.config.height as f64 - 400. {
                        match touch.phase {
                            TouchPhase::Started => {
                                navigation_fingers.insert(touch.id);
                                if touch.location.x < render_state.surface.config.width as f64 / 3.
                                {
                                    scene_ix = scene_ix.saturating_sub(1);
                                } else if touch.location.x
                                    > 2. * render_state.surface.config.width as f64 / 3.
                                {
                                    scene_ix = scene_ix.saturating_add(1);
                                }
                            }
                            TouchPhase::Ended | TouchPhase::Cancelled => {
                                navigation_fingers.remove(&touch.id);
                            }
                            TouchPhase::Moved => (),
                        }
                    }
                    if !navigation_fingers.contains(&touch.id) {
                        touch_state.add_event(touch);
                    }
                }
                WindowEvent::Resized(size) => {
                    render_cx.resize_surface(&mut render_state.surface, size.width, size.height);
                    render_state.window.request_redraw();
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
            }
        }
        Event::MainEventsCleared => {
            touch_state.end_frame();
            let touch_info = touch_state.info();
            if let Some(touch_info) = touch_info {
                let centre = Vec2::new(touch_info.zoom_centre.x, touch_info.zoom_centre.y);
                transform = Affine::translate(touch_info.translation_delta)
                    * Affine::translate(centre)
                    * Affine::scale(touch_info.zoom_delta)
                    * Affine::rotate(touch_info.rotation_delta)
                    * Affine::translate(-centre)
                    * transform;
            }

            if let Some(render_state) = &mut render_state {
                render_state.window.request_redraw();
            }
        }
        Event::RedrawRequested(_) => {
            let Some(render_state) = &mut render_state else { return };
            let width = render_state.surface.config.width;
            let height = render_state.surface.config.height;
            let device_handle = &render_cx.devices[render_state.surface.dev_id];

            // Allow looping forever
            scene_ix = scene_ix.rem_euclid(scenes.scenes.len() as i32);
            let example_scene = &mut scenes.scenes[scene_ix as usize];
            if prev_scene_ix != scene_ix {
                transform = Affine::IDENTITY;
                prev_scene_ix = scene_ix;
                render_state
                    .window
                    .set_title(&format!("Vello demo - {}", example_scene.config.name));
            }
            let mut builder = SceneBuilder::for_fragment(&mut fragment);
            let mut params = SceneParams {
                time: start.elapsed().as_secs_f64(),
                text: &mut simple_text,
                resolution: None,
            };
            (example_scene.function)(&mut builder, &mut params);
            builder.finish();
            let mut builder = SceneBuilder::for_scene(&mut scene);
            let mut transform = transform;
            if let Some(resolution) = params.resolution {
                let factor = Vec2::new(width as f64, height as f64);
                let scale_factor = (factor.x / resolution.x).min(factor.y / resolution.y);
                transform = transform * Affine::scale(scale_factor);
            }
            builder.append(&fragment, Some(transform));
            builder.finish();
            let surface_texture = render_state
                .surface
                .surface
                .get_current_texture()
                .expect("failed to get surface texture");
            #[cfg(not(target_arch = "wasm32"))]
            {
                vello::block_on_wgpu(
                    &device_handle.device,
                    renderers[render_state.surface.dev_id]
                        .as_mut()
                        .unwrap()
                        .render_to_surface_async(
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
            renderers[render_state.surface.dev_id]
                .as_mut()
                .unwrap()
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
            #[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
            UserEvent::HotReload => {
                let Some(render_state) = &mut render_state else { return };
                let device_handle = &render_cx.devices[render_state.surface.dev_id];
                eprintln!("==============\nReloading shaders");
                let start = Instant::now();
                let result = renderers[render_state.surface.dev_id]
                    .as_mut()
                    .unwrap()
                    .reload_shaders(&device_handle.device);
                // We know that the only async here is actually sync, so we just block
                match pollster::block_on(result) {
                    Ok(_) => eprintln!("Reloading took {:?}", start.elapsed()),
                    Err(e) => eprintln!("Failed to reload shaders because of {e}"),
                }
            }
        },
        Event::Suspended => {
            render_state = None;
            *control_flow = ControlFlow::Wait;
        }
        Event::Resumed => {
            #[cfg(target_arch = "wasm32")]
            {}
            #[cfg(not(target_arch = "wasm32"))]
            {
                let Option::None = render_state else { return };
                let window = create_window(_event_loop);
                let size = window.inner_size();
                let surface_future = render_cx.create_surface(&window, size.width, size.height);
                // We need to block here, in case a Suspended event appeared
                let surface = pollster::block_on(surface_future);
                render_state = {
                    let render_state = RenderState { window, surface };
                    renderers.resize_with(render_cx.devices.len(), || None);
                    let id = render_state.surface.dev_id;
                    renderers[id].get_or_insert_with(|| {
                        eprintln!("Creating renderer {id}");
                        Renderer::new(&render_cx.devices[id].device, render_state.surface.format)
                            .expect("Could create renderer")
                    });
                    Some(render_state)
                };
                *control_flow = ControlFlow::Poll;
            }
        }
        _ => {}
    });
}

fn create_window(event_loop: &winit::event_loop::EventLoopWindowTarget<UserEvent>) -> Window {
    use winit::{dpi::LogicalSize, window::WindowBuilder};
    WindowBuilder::new()
        .with_inner_size(LogicalSize::new(1044, 800))
        .with_resizable(true)
        .with_title("Vello demo")
        .build(&event_loop)
        .unwrap()
}

#[derive(Debug)]
enum UserEvent {
    #[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
    HotReload,
}

pub fn main() -> Result<()> {
    // TODO: initializing both env_logger and console_logger fails on wasm.
    // Figure out a more principled approach.
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let args = Args::parse();
    let scenes = args.args.select_scene_set(|| Args::command())?;
    if let Some(scenes) = scenes {
        let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
        #[allow(unused_mut)]
        let mut render_cx = RenderContext::new().unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        {
            #[cfg(not(target_os = "android"))]
            let proxy = event_loop.create_proxy();
            #[cfg(not(target_os = "android"))]
            let _keep = hot_reload::hot_reload(move || {
                proxy.send_event(UserEvent::HotReload).ok().map(drop)
            });

            run(event_loop, args, scenes, render_cx);
        }
        #[cfg(target_arch = "wasm32")]
        {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init().expect("could not initialize logger");
            use winit::platform::web::WindowExtWebSys;
            let window = create_window(&event_loop);
            // On wasm, append the canvas to the document body
            let canvas = window.canvas();
            canvas.set_width(1044);
            canvas.set_height(800);
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.body())
                .and_then(|body| body.append_child(&web_sys::Element::from(canvas)).ok())
                .expect("couldn't append canvas to document body");
            wasm_bindgen_futures::spawn_local(async move {
                let size = window.inner_size();
                let surface = render_cx
                    .create_surface(&window, size.width, size.height)
                    .await;
                let render_state = RenderState { window, surface };
                // No error handling here; if the event loop has finished, we don't need to send them the surface
                run(event_loop, args, scenes, render_cx, render_state);
            });
        }
    }
    Ok(())
}

#[cfg(target_os = "android")]
use winit::platform::android::activity::AndroidApp;

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;

    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Warn),
    );

    let event_loop = EventLoopBuilder::with_user_event()
        .with_android_app(app)
        .build();
    let args = Args::parse();
    let scenes = args
        .args
        .select_scene_set(|| Args::command())
        .unwrap()
        .unwrap();
    let render_cx = RenderContext::new().unwrap();

    run(event_loop, args, scenes, render_cx);
}
