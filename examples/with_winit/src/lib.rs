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
use scenes::{ImageCache, SceneParams, SceneSet, SimpleText};
use vello::peniko::Color;
use vello::util::RenderSurface;
use vello::{
    kurbo::{Affine, Vec2},
    util::RenderContext,
    Renderer, Scene, SceneBuilder,
};
use vello::{RendererOptions, SceneFragment};

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
    /// Which scene (index) to start on
    /// Switch between scenes with left and right arrow keys
    #[arg(long)]
    scene: Option<i32>,
    #[command(flatten)]
    args: scenes::Arguments,
}

struct RenderState {
    // SAFETY: We MUST drop the surface before the `window`, so the fields
    // must be in this order
    surface: RenderSurface,
    window: Window,
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
    // The design of `RenderContext` forces delayed renderer initialisation to
    // not work on wasm, as WASM futures effectively must be 'static.
    // Otherwise, this could work by sending the result to event_loop.proxy
    // instead of blocking
    #[cfg(target_arch = "wasm32")]
    let mut render_state = {
        renderers.resize_with(render_cx.devices.len(), || None);
        let id = render_state.surface.dev_id;
        renderers[id] = Some(
            Renderer::new(
                &render_cx.devices[id].device,
                &RendererOptions {
                    surface_format: Some(render_state.surface.format),
                },
            )
            .expect("Could create renderer"),
        );
        Some(render_state)
    };
    // Whilst suspended, we drop `render_state`, but need to keep the same window.
    // If render_state exists, we must store the window in it, to maintain drop order
    #[cfg(not(target_arch = "wasm32"))]
    let mut cached_window = None;

    let mut scene = Scene::new();
    let mut fragment = SceneFragment::new();
    let mut simple_text = SimpleText::new();
    let mut images = ImageCache::new();
    let start = Instant::now();

    let mut touch_state = multi_touch::TouchState::new();
    // navigation_fingers are fingers which are used in the navigation 'zone' at the bottom
    // of the screen. This ensures that one press on the screen doesn't have multiple actions
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
    // _event_loop is used on non-wasm platforms to create new windows
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
                WindowEvent::Touch(touch) => {
                    match touch.phase {
                        TouchPhase::Started => {
                            // We reserve the bottom third of the screen for navigation
                            // This also prevents strange effects whilst using the navigation gestures on Android
                            // TODO: How do we know what the client area is? Winit seems to just give us the
                            // full screen
                            // TODO: Render a display of the navigation regions. We don't do
                            // this currently because we haven't researched how to determine when we're
                            // in a touch context (i.e. Windows/Linux/MacOS with a touch screen could
                            // also be using mouse/keyboard controls)
                            // Note that winit's rendering is y-down
                            if touch.location.y
                                > render_state.surface.config.height as f64 * 2. / 3.
                            {
                                navigation_fingers.insert(touch.id);
                                // The left third of the navigation zone navigates backwards
                                if touch.location.x < render_state.surface.config.width as f64 / 3.
                                {
                                    scene_ix = scene_ix.saturating_sub(1);
                                } else if touch.location.x
                                    > 2. * render_state.surface.config.width as f64 / 3.
                                {
                                    scene_ix = scene_ix.saturating_add(1);
                                }
                            }
                        }
                        TouchPhase::Ended | TouchPhase::Cancelled => {
                            // We intentionally ignore the result here
                            navigation_fingers.remove(&touch.id);
                        }
                        TouchPhase::Moved => (),
                    }
                    // See documentation on navigation_fingers
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
            let mut scene_params = SceneParams {
                time: start.elapsed().as_secs_f64(),
                text: &mut simple_text,
                images: &mut images,
                resolution: None,
                base_color: None,
                interactive: true,
            };
            (example_scene.function)(&mut builder, &mut scene_params);

            // If the user specifies a base color in the CLI we use that. Otherwise we use any
            // color specified by the scene. The default is black.
            let render_params = vello::RenderParams {
                base_color: args
                    .args
                    .base_color
                    .or(scene_params.base_color)
                    .unwrap_or(Color::BLACK),
                width,
                height,
            };
            let mut builder = SceneBuilder::for_scene(&mut scene);
            let mut transform = transform;
            if let Some(resolution) = scene_params.resolution {
                // Automatically scale the rendering to fill as much of the window as possible
                // TODO: Apply svg view_box, somehow
                let factor = Vec2::new(width as f64, height as f64);
                let scale_factor = (factor.x / resolution.x).min(factor.y / resolution.y);
                transform = transform * Affine::scale(scale_factor);
            }
            builder.append(&fragment, Some(transform));
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
                            &render_params,
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
                    &render_params,
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
                // We know that the only async here (`pop_error_scope`) is actually sync, so blocking is fine
                match pollster::block_on(result) {
                    Ok(_) => eprintln!("Reloading took {:?}", start.elapsed()),
                    Err(e) => eprintln!("Failed to reload shaders because of {e}"),
                }
            }
        },
        Event::Suspended => {
            eprintln!("Suspending");
            #[cfg(not(target_arch = "wasm32"))]
            // When we suspend, we need to remove the `wgpu` Surface
            if let Some(render_state) = render_state.take() {
                cached_window = Some(render_state.window);
            }
            *control_flow = ControlFlow::Wait;
        }
        Event::Resumed => {
            #[cfg(target_arch = "wasm32")]
            {}
            #[cfg(not(target_arch = "wasm32"))]
            {
                let Option::None = render_state else { return };
                let window = cached_window
                    .take()
                    .unwrap_or_else(|| create_window(_event_loop));
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
                        Renderer::new(
                            &render_cx.devices[id].device,
                            &RendererOptions {
                                surface_format: Some(render_state.surface.format),
                            },
                        )
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
