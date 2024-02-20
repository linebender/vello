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

use instant::{Duration, Instant};
use std::{collections::HashSet, sync::Arc};

use anyhow::Result;
use clap::{CommandFactory, Parser};
use scenes::{ImageCache, SceneParams, SceneSet, SimpleText};
use vello::peniko::Color;
use vello::util::RenderSurface;
use vello::{
    kurbo::{Affine, Vec2},
    util::RenderContext,
    AaConfig, Renderer, Scene,
};
use vello::{BumpAllocators, RendererOptions};

use winit::{
    event_loop::{EventLoop, EventLoopBuilder},
    window::Window,
};

#[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
mod hot_reload;
mod multi_touch;
mod stats;

#[derive(Parser, Debug)]
#[command(about, long_about = None, bin_name="cargo run -p with_winit --")]
struct Args {
    /// Which scene (index) to start on
    /// Switch between scenes with left and right arrow keys
    #[arg(long)]
    scene: Option<i32>,
    #[command(flatten)]
    args: scenes::Arguments,
    #[arg(long)]
    /// Whether to use CPU shaders
    use_cpu: bool,
}

struct RenderState<'s> {
    // SAFETY: We MUST drop the surface before the `window`, so the fields
    // must be in this order
    surface: RenderSurface<'s>,
    window: Arc<Window>,
}

fn run(
    event_loop: EventLoop<UserEvent>,
    args: Args,
    mut scenes: SceneSet,
    render_cx: RenderContext,
    #[cfg(target_arch = "wasm32")] render_state: RenderState,
) {
    use winit::{event::*, event_loop::ControlFlow, keyboard::*};
    let mut renderers: Vec<Option<Renderer>> = vec![];
    #[cfg(not(target_arch = "wasm32"))]
    let mut render_cx = render_cx;
    #[cfg(not(target_arch = "wasm32"))]
    let mut render_state = None::<RenderState>;
    let use_cpu = args.use_cpu;
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
                RendererOptions {
                    surface_format: Some(render_state.surface.format),
                    use_cpu: use_cpu,
                    antialiasing_support: vello::AaSupport::all(),
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
    let mut fragment = Scene::new();
    let mut simple_text = SimpleText::new();
    let mut images = ImageCache::new();
    let mut stats = stats::Stats::new();
    let mut stats_shown = true;
    // Currently not updated in wasm builds
    #[allow(unused_mut)]
    let mut scene_complexity: Option<BumpAllocators> = None;
    let mut complexity_shown = false;
    let mut vsync_on = true;

    const AA_CONFIGS: [AaConfig; 3] = [AaConfig::Area, AaConfig::Msaa8, AaConfig::Msaa16];
    // We allow cycling through AA configs in either direction, so use a signed index
    let mut aa_config_ix: i32 = 0;

    let mut frame_start_time = Instant::now();
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
    let mut complexity: usize = 0;
    if let Some(set_scene) = args.scene {
        scene_ix = set_scene;
    }
    let mut profile_stored = None;
    let mut prev_scene_ix = scene_ix - 1;
    let mut profile_taken = Instant::now();
    let mut modifiers = ModifiersState::default();

    #[cfg(feature = "debug_layers")]
    let mut debug = vello::DebugLayers::none();

    event_loop
        .run(move |event, event_loop| match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } => {
                let Some(render_state) = &mut render_state else {
                    return;
                };
                if render_state.window.id() != window_id {
                    return;
                }
                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::ModifiersChanged(m) => modifiers = m.state(),
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            match event.logical_key.as_ref() {
                                Key::Named(NamedKey::ArrowLeft) => {
                                    scene_ix = scene_ix.saturating_sub(1)
                                }
                                Key::Named(NamedKey::ArrowRight) => {
                                    scene_ix = scene_ix.saturating_add(1)
                                }
                                Key::Named(NamedKey::ArrowUp) => complexity += 1,
                                Key::Named(NamedKey::ArrowDown) => {
                                    complexity = complexity.saturating_sub(1)
                                }
                                Key::Named(NamedKey::Space) => {
                                    transform = Affine::IDENTITY;
                                }
                                Key::Character(char) => {
                                    // TODO: Have a more principled way of handling modifiers on keypress
                                    // see e.g. https://xi.zulipchat.com/#narrow/stream/351333-glazier/topic/Keyboard.20shortcuts
                                    let char = char.to_lowercase();
                                    match char.as_str() {
                                        "q" | "e" => {
                                            if let Some(prior_position) = prior_position {
                                                let is_clockwise = char == "e";
                                                let angle = if is_clockwise { -0.05 } else { 0.05 };
                                                transform = Affine::translate(prior_position)
                                                    * Affine::rotate(angle)
                                                    * Affine::translate(-prior_position)
                                                    * transform;
                                            }
                                        }
                                        "s" => {
                                            stats_shown = !stats_shown;
                                        }
                                        "d" => {
                                            complexity_shown = !complexity_shown;
                                        }
                                        "c" => {
                                            stats.clear_min_and_max();
                                        }
                                        "m" => {
                                            aa_config_ix = if modifiers.shift_key() {
                                                aa_config_ix.saturating_sub(1)
                                            } else {
                                                aa_config_ix.saturating_add(1)
                                            };
                                        }
                                        "p" => {
                                            if let Some(renderer) = &renderers[render_state.surface.dev_id]
                                            {
                                                if let Some(profile_result) = &renderer
                                                  .profile_result
                                                  .as_ref()
                                                  .or(profile_stored.as_ref())
                                                {
                                                    // There can be empty results if the required features aren't supported
                                                    if !profile_result.is_empty() {
                                                        let path = std::path::Path::new("trace.json");
                                                        match wgpu_profiler::chrometrace::write_chrometrace(
                                                            path,
                                                            profile_result,
                                                        ) {
                                                            Ok(()) => {
                                                                println!("Wrote trace to path {path:?}");
                                                            }
                                                            Err(e) => {
                                                                eprintln!("Failed to write trace {e}")
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        "v" => {
                                            vsync_on = !vsync_on;
                                            render_cx.set_present_mode(
                                                &mut render_state.surface,
                                                if vsync_on {
                                                    wgpu::PresentMode::AutoVsync
                                                } else {
                                                    wgpu::PresentMode::AutoNoVsync
                                                },
                                            );
                                        }
                                        #[cfg(feature = "debug_layers")]
                                        "1" => {
                                            debug.toggle(vello::DebugLayers::BOUNDING_BOXES);
                                        }
                                        #[cfg(feature = "debug_layers")]
                                        "2" => {
                                            debug.toggle(vello::DebugLayers::LINESOUP_SEGMENTS);
                                        }
                                        #[cfg(feature = "debug_layers")]
                                        "3" => {
                                            debug.toggle(vello::DebugLayers::LINESOUP_POINTS);
                                        }
                                        #[cfg(feature = "debug_layers")]
                                        "4" => {
                                            debug.toggle(vello::DebugLayers::VALIDATION);
                                        }
                                        _ => {}
                                    }
                                }
                                Key::Named(NamedKey::Escape) => event_loop.exit(),
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
                                    if touch.location.x
                                        < render_state.surface.config.width as f64 / 3.
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
                        render_cx.resize_surface(
                            &mut render_state.surface,
                            size.width,
                            size.height,
                        );
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
                            eprintln!(
                                "Scrolling without mouse in window; this shouldn't be possible"
                            );
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
                    WindowEvent::RedrawRequested => {
                        let width = render_state.surface.config.width;
                        let height = render_state.surface.config.height;
                        let device_handle = &render_cx.devices[render_state.surface.dev_id];
                        let snapshot = stats.snapshot();

                        // Allow looping forever
                        scene_ix = scene_ix.rem_euclid(scenes.scenes.len() as i32);
                        aa_config_ix = aa_config_ix.rem_euclid(AA_CONFIGS.len() as i32);

                        let example_scene = &mut scenes.scenes[scene_ix as usize];
                        if prev_scene_ix != scene_ix {
                            transform = Affine::IDENTITY;
                            prev_scene_ix = scene_ix;
                            render_state
                                .window
                                .set_title(&format!("Vello demo - {}", example_scene.config.name));
                        }
                        fragment.reset();
                        let mut scene_params = SceneParams {
                            time: start.elapsed().as_secs_f64(),
                            text: &mut simple_text,
                            images: &mut images,
                            resolution: None,
                            base_color: None,
                            interactive: true,
                            complexity,
                        };
                        example_scene
                            .function
                            .render(&mut fragment, &mut scene_params);

                        // If the user specifies a base color in the CLI we use that. Otherwise we use any
                        // color specified by the scene. The default is black.
                        let base_color = args
                            .args
                            .base_color
                            .or(scene_params.base_color)
                            .unwrap_or(Color::BLACK);
                        let antialiasing_method = AA_CONFIGS[aa_config_ix as usize];
                        let render_params = vello::RenderParams {
                            base_color,
                            width,
                            height,
                            antialiasing_method,
                            #[cfg(feature = "debug_layers")]
                            debug,
                        };
                        scene.reset();
                        let mut transform = transform;
                        if let Some(resolution) = scene_params.resolution {
                            // Automatically scale the rendering to fill as much of the window as possible
                            // TODO: Apply svg view_box, somehow
                            let factor = Vec2::new(width as f64, height as f64);
                            let scale_factor =
                                (factor.x / resolution.x).min(factor.y / resolution.y);
                            transform *= Affine::scale(scale_factor);
                        }
                        scene.append(&fragment, Some(transform));
                        if stats_shown {
                            snapshot.draw_layer(
                                &mut scene,
                                scene_params.text,
                                width as f64,
                                height as f64,
                                stats.samples(),
                                complexity_shown.then_some(scene_complexity).flatten(),
                                vsync_on,
                                antialiasing_method,
                            );
                            if let Some(profiling_result) = renderers[render_state.surface.dev_id]
                                .as_mut()
                                .and_then(|it| it.profile_result.take())
                            {
                                if profile_stored.is_none()
                                    || profile_taken.elapsed() > Duration::from_secs(1)
                                {
                                    profile_stored = Some(profiling_result);
                                    profile_taken = Instant::now();
                                }
                            }
                            if let Some(profiling_result) = profile_stored.as_ref() {
                                stats::draw_gpu_profiling(
                                    &mut scene,
                                    scene_params.text,
                                    width as f64,
                                    height as f64,
                                    profiling_result,
                                );
                            }
                        }
                        let surface_texture = render_state
                            .surface
                            .surface
                            .get_current_texture()
                            .expect("failed to get surface texture");
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            scene_complexity = vello::block_on_wgpu(
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

                        let new_time = Instant::now();
                        stats.add_sample(stats::Sample {
                            frame_time_us: (new_time - frame_start_time).as_micros() as u64,
                        });
                        frame_start_time = new_time;
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
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
            Event::UserEvent(event) => match event {
                #[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
                UserEvent::HotReload => {
                    let Some(render_state) = &mut render_state else {
                        return;
                    };
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
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            Event::Resumed => {
                #[cfg(target_arch = "wasm32")]
                {}
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let Option::None = render_state else { return };
                    let window = cached_window
                        .take()
                        .unwrap_or_else(|| create_window(event_loop));
                    let size = window.inner_size();
                    let surface_future = render_cx.create_surface(window.clone(), size.width, size.height);
                    // We need to block here, in case a Suspended event appeared
                    let surface =
                        pollster::block_on(surface_future).expect("Error creating surface");
                    render_state = {
                        let render_state = RenderState { window, surface };
                        renderers.resize_with(render_cx.devices.len(), || None);
                        let id = render_state.surface.dev_id;
                        renderers[id].get_or_insert_with(|| {
                            eprintln!("Creating renderer {id}");
                            Renderer::new(
                                &render_cx.devices[id].device,
                                RendererOptions {
                                    surface_format: Some(render_state.surface.format),
                                    use_cpu,
                                    antialiasing_support: vello::AaSupport::all(),
                                },
                            )
                            .expect("Could create renderer")
                        });
                        Some(render_state)
                    };
                    event_loop.set_control_flow(ControlFlow::Poll);
                }
            }
            _ => {}
        })
        .expect("run to completion");
}

fn create_window(event_loop: &winit::event_loop::EventLoopWindowTarget<UserEvent>) -> Arc<Window> {
    use winit::{dpi::LogicalSize, window::WindowBuilder};
    Arc::new(
        WindowBuilder::new()
            .with_inner_size(LogicalSize::new(1044, 800))
            .with_resizable(true)
            .with_title("Vello demo")
            .build(event_loop)
            .unwrap(),
    )
}

#[derive(Debug)]
enum UserEvent {
    #[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
    HotReload,
}

#[cfg(target_arch = "wasm32")]
fn display_error_message() -> Option<()> {
    let window = web_sys::window()?;
    let document = window.document()?;
    let elements = document.get_elements_by_tag_name("body");
    let body = elements.item(0)?;
    body.set_inner_html(
        r#"<style>
        p {
            margin: 2em 10em;
            font-family: sans-serif;
        }
        </style>
        <p><a href="https://caniuse.com/webgpu">WebGPU</a>
        is not enabled. Make sure your browser is updated to
        <a href="https://chromiumdash.appspot.com/schedule">Chrome M113</a> or
        another browser compatible with WebGPU.</p>"#,
    );
    Some(())
}

pub fn main() -> Result<()> {
    // TODO: initializing both env_logger and console_logger fails on wasm.
    // Figure out a more principled approach.
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let args = Args::parse();
    let scenes = args.args.select_scene_set(Args::command)?;
    if let Some(scenes) = scenes {
        let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build()?;
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
            let canvas = window.canvas().unwrap();
            let size = window.inner_size();
            canvas.set_width(size.width);
            canvas.set_height(size.height);
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.body())
                .and_then(|body| body.append_child(canvas.as_ref()).ok())
                .expect("couldn't append canvas to document body");
            _ = web_sys::HtmlElement::from(canvas).focus();
            wasm_bindgen_futures::spawn_local(async move {
                let size = window.inner_size();
                let surface = render_cx
                    .create_surface(window.clone(), size.width, size.height)
                    .await;
                if let Ok(surface) = surface {
                    let render_state = RenderState { window, surface };
                    // No error handling here; if the event loop has finished, we don't need to send them the surface
                    run(event_loop, args, scenes, render_cx, render_state);
                } else {
                    _ = display_error_message();
                }
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
        .build()
        .expect("Required to continue");
    let args = Args::parse();
    let scenes = args
        .args
        .select_scene_set(|| Args::command())
        .unwrap()
        .unwrap();
    let render_cx = RenderContext::new().unwrap();

    run(event_loop, args, scenes, render_cx);
}
