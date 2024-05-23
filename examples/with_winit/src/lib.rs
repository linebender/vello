// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use instant::Instant;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

use clap::Parser;
use scenes::{ImageCache, SceneParams, SceneSet, SimpleText};
use vello::kurbo::{Affine, Vec2};
use vello::peniko::Color;
use vello::util::{RenderContext, RenderSurface};
use vello::{AaConfig, BumpAllocators, Renderer, RendererOptions, Scene};

use winit::dpi::LogicalSize;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowAttributes};

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
    /// Used to disable vsync at startup. Can be toggled with the "V" key.
    ///
    /// This setting is useful for Android, where it might be harder to press this key
    #[arg(long)]
    startup_vsync_off: bool,
    /// Used to enable gpu profiling at startup. Can be toggled with the "G" key
    ///
    /// It is off by default because it has adverse performance characteristics
    #[arg(long)]
    #[cfg(feature = "wgpu-profiler")]
    startup_gpu_profiling_on: bool,
    /// Whether to force initialising the shaders serially (rather than spawning threads)
    /// This has no effect on wasm, and defaults to 1 on macOS for performance reasons
    ///
    /// Use `0` for an automatic choice
    #[arg(long, default_value_t=default_threads())]
    num_init_threads: usize,
    /// Use the asynchronous pipeline (if available) for rendering
    ///
    /// The asynchronous pipeline is one approach for robust memory - see
    /// <https://github.com/linebender/vello/issues/366>
    ///
    /// However, it also has potential latency issues, especially for
    /// accessibility technology, as it (currently) blocks the main thread for
    /// extended periods
    #[arg(long)]
    async_pipeline: bool,
}

fn default_threads() -> usize {
    #[cfg(target_os = "macos")]
    return 1;
    #[cfg(not(target_os = "macos"))]
    return 0;
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
    use winit::event::*;
    use winit::event_loop::ControlFlow;
    use winit::keyboard::*;
    let mut renderers: Vec<Option<Renderer>> = vec![];
    #[cfg(not(target_arch = "wasm32"))]
    let mut render_cx = render_cx;
    #[cfg(not(target_arch = "wasm32"))]
    let mut render_state = None::<RenderState>;
    let use_cpu = args.use_cpu;
    // The available kinds of anti-aliasing
    #[cfg(not(target_os = "android"))]
    // TODO: Make this set configurable through the command line
    // Alternatively, load anti-aliasing shaders on demand/asynchronously
    let aa_configs = [AaConfig::Area, AaConfig::Msaa8, AaConfig::Msaa16];
    #[cfg(target_os = "android")]
    // Hard code to only one on Android whilst we are working on startup speed
    let aa_configs = [AaConfig::Area];

    // The design of `RenderContext` forces delayed renderer initialisation to
    // not work on wasm, as WASM futures effectively must be 'static.
    // Otherwise, this could work by sending the result to event_loop.proxy
    // instead of blocking
    #[cfg(target_arch = "wasm32")]
    let mut render_state = {
        renderers.resize_with(render_cx.devices.len(), || None);
        let id = render_state.surface.dev_id;
        #[allow(unused_mut)]
        let mut renderer = Renderer::new(
            &render_cx.devices[id].device,
            RendererOptions {
                surface_format: Some(render_state.surface.format),
                use_cpu,
                antialiasing_support: aa_configs.iter().copied().collect(),
                // We currently initialise on one thread on WASM, but mark this here
                // anyway
                num_init_threads: NonZeroUsize::new(1),
            },
        )
        .expect("Could create renderer");
        #[cfg(feature = "wgpu-profiler")]
        renderer
            .profiler
            .change_settings(wgpu_profiler::GpuProfilerSettings {
                enable_timer_queries: args.startup_gpu_profiling_on,
                enable_debug_groups: args.startup_gpu_profiling_on,
                ..Default::default()
            })
            .expect("Not setting max_num_pending_frames");
        renderers[id] = Some(renderer);
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
    let mut vsync_on = !args.startup_vsync_off;

    #[cfg(feature = "wgpu-profiler")]
    let mut gpu_profiling_on = args.startup_gpu_profiling_on;
    #[cfg(feature = "wgpu-profiler")]
    let mut profile_stored = None;
    #[cfg(feature = "wgpu-profiler")]
    let mut profile_taken = Instant::now();

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
    let mut prev_scene_ix = scene_ix - 1;
    let mut modifiers = ModifiersState::default();
    #[allow(deprecated)]
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
                let _span = if !matches!(event, WindowEvent::RedrawRequested) {
                    Some(tracing::trace_span!("Handling window event", ?event).entered())
                } else {
                    None
                };
                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::ModifiersChanged(m) => modifiers = m.state(),
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            match event.logical_key.as_ref() {
                                Key::Named(NamedKey::ArrowLeft) => {
                                    scene_ix = scene_ix.saturating_sub(1);
                                }
                                Key::Named(NamedKey::ArrowRight) => {
                                    scene_ix = scene_ix.saturating_add(1);
                                }
                                Key::Named(NamedKey::ArrowUp) => complexity += 1,
                                Key::Named(NamedKey::ArrowDown) => {
                                    complexity = complexity.saturating_sub(1);
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
                                        #[cfg(feature = "wgpu-profiler")]
                                        "p" => {
                                            if let Some(renderer) =
                                                &renderers[render_state.surface.dev_id]
                                            {
                                                store_profiling(renderer, &profile_stored);
                                            }
                                        }
                                        #[cfg(feature = "wgpu-profiler")]
                                        "g" => {
                                            gpu_profiling_on = !gpu_profiling_on;
                                            if let Some(renderer) =
                                                &mut renderers[render_state.surface.dev_id]
                                            {
                                                renderer
                                                    .profiler
                                                    .change_settings(
                                                        wgpu_profiler::GpuProfilerSettings {
                                                            enable_timer_queries: gpu_profiling_on,
                                                            enable_debug_groups: gpu_profiling_on,
                                                            ..Default::default()
                                                        },
                                                    )
                                                    .expect("Not setting max_num_pending_frames");
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
                            log::warn!(
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
                        let _rendering_span =
                            tracing::trace_span!("Actioning Requested Redraw").entered();
                        let encoding_span = tracing::trace_span!("Encoding scene").entered();

                        let width = render_state.surface.config.width;
                        let height = render_state.surface.config.height;
                        let device_handle = &render_cx.devices[render_state.surface.dev_id];
                        let snapshot = stats.snapshot();

                        // Allow looping forever
                        scene_ix = scene_ix.rem_euclid(scenes.scenes.len() as i32);
                        aa_config_ix = aa_config_ix.rem_euclid(aa_configs.len() as i32);

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
                        let antialiasing_method = aa_configs[aa_config_ix as usize];
                        let render_params = vello::RenderParams {
                            base_color,
                            width,
                            height,
                            antialiasing_method,
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
                            #[cfg(feature = "wgpu-profiler")]
                            if let Some(profiling_result) = renderers[render_state.surface.dev_id]
                                .as_mut()
                                .and_then(|it| it.profile_result.take())
                            {
                                if profile_stored.is_none()
                                    || profile_taken.elapsed() > instant::Duration::from_secs(1)
                                {
                                    profile_stored = Some(profiling_result);
                                    profile_taken = Instant::now();
                                }
                            }
                            #[cfg(feature = "wgpu-profiler")]
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
                        drop(encoding_span);
                        let texture_span = tracing::trace_span!("Getting texture").entered();
                        let surface_texture = render_state
                            .surface
                            .surface
                            .get_current_texture()
                            .expect("failed to get surface texture");
                        drop(texture_span);
                        let render_span = tracing::trace_span!("Dispatching render").entered();
                        // Note: we don't run the async/"robust" pipeline, as
                        // it requires more async wiring for the readback. See
                        // [#gpu > async on wasm](https://xi.zulipchat.com/#narrow/stream/197075-gpu/topic/async.20on.20wasm)
                        if args.async_pipeline && cfg!(not(target_arch = "wasm32")) {
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
                        } else {
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
                        }
                        surface_texture.present();
                        drop(render_span);
                        {
                            let _poll_aspan = tracing::trace_span!("Polling wgpu device").entered();
                            device_handle.device.poll(wgpu::Maintain::Poll);
                        }
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
                    log::info!("==============\nReloading shaders");
                    let start = Instant::now();
                    let result = renderers[render_state.surface.dev_id]
                        .as_mut()
                        .unwrap()
                        .reload_shaders(&device_handle.device);
                    // We know that the only async here (`pop_error_scope`) is actually sync, so blocking is fine
                    match pollster::block_on(result) {
                        Ok(_) => log::info!("Reloading took {:?}", start.elapsed()),
                        Err(e) => log::warn!("Failed to reload shaders because of {e}"),
                    }
                }
            },
            Event::Suspended => {
                log::info!("Suspending");
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
                    let window = cached_window.take().unwrap_or_else(|| {
                        Arc::new(event_loop.create_window(window_attributes()).unwrap())
                    });
                    let size = window.inner_size();
                    let present_mode = if vsync_on {
                        wgpu::PresentMode::AutoVsync
                    } else {
                        wgpu::PresentMode::AutoNoVsync
                    };
                    let surface_future = render_cx.create_surface(
                        window.clone(),
                        size.width,
                        size.height,
                        present_mode,
                    );
                    // We need to block here, in case a Suspended event appeared
                    let surface =
                        pollster::block_on(surface_future).expect("Error creating surface");
                    render_state = {
                        let render_state = RenderState { window, surface };
                        renderers.resize_with(render_cx.devices.len(), || None);
                        let id = render_state.surface.dev_id;
                        renderers[id].get_or_insert_with(|| {
                            let start = Instant::now();
                            #[allow(unused_mut)]
                            let mut renderer = Renderer::new(
                                &render_cx.devices[id].device,
                                RendererOptions {
                                    surface_format: Some(render_state.surface.format),
                                    use_cpu,
                                    antialiasing_support: aa_configs.iter().copied().collect(),
                                    num_init_threads: NonZeroUsize::new(args.num_init_threads),
                                },
                            )
                            .expect("Could create renderer");
                            log::info!("Creating renderer {id} took {:?}", start.elapsed());
                            #[cfg(feature = "wgpu-profiler")]
                            renderer
                                .profiler
                                .change_settings(wgpu_profiler::GpuProfilerSettings {
                                    enable_timer_queries: gpu_profiling_on,
                                    enable_debug_groups: gpu_profiling_on,
                                    ..Default::default()
                                })
                                .expect("Not setting max_num_pending_frames");
                            renderer
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

#[cfg(feature = "wgpu-profiler")]
/// A function extracted to fix rustfmt
fn store_profiling(
    renderer: &Renderer,
    profile_stored: &Option<Vec<wgpu_profiler::GpuTimerQueryResult>>,
) {
    if let Some(profile_result) = &renderer.profile_result.as_ref().or(profile_stored.as_ref()) {
        // There can be empty results if the required features aren't supported
        if !profile_result.is_empty() {
            let path = std::path::Path::new("trace.json");
            match wgpu_profiler::chrometrace::write_chrometrace(path, profile_result) {
                Ok(()) => {
                    println!("Wrote trace to path {path:?}");
                }
                Err(e) => {
                    log::warn!("Failed to write trace {e}");
                }
            }
        }
    }
}

fn window_attributes() -> WindowAttributes {
    Window::default_attributes()
        .with_inner_size(LogicalSize::new(1044, 800))
        .with_resizable(true)
        .with_title("Vello demo")
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

#[cfg(not(target_os = "android"))]
pub fn main() -> anyhow::Result<()> {
    // TODO: initializing both env_logger and console_logger fails on wasm.
    // Figure out a more principled approach.
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::builder()
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .init();
    let args = parse_arguments();
    let scenes = args.args.select_scene_set()?;
    if let Some(scenes) = scenes {
        let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
        #[allow(unused_mut)]
        let mut render_cx = RenderContext::new();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let proxy = event_loop.create_proxy();
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
            #[allow(deprecated)]
            let window = Arc::new(event_loop.create_window(window_attributes()).unwrap());
            // On wasm, append the canvas to the document body
            let canvas = window.canvas().unwrap();
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.body())
                .and_then(|body| body.append_child(canvas.as_ref()).ok())
                .expect("couldn't append canvas to document body");
            // Best effort to start with the canvas focused, taking input
            _ = web_sys::HtmlElement::from(canvas).focus();
            wasm_bindgen_futures::spawn_local(async move {
                let (width, height, scale_factor) = web_sys::window()
                    .map(|w| {
                        (
                            w.inner_width().unwrap().as_f64().unwrap(),
                            w.inner_height().unwrap().as_f64().unwrap(),
                            w.device_pixel_ratio(),
                        )
                    })
                    .unwrap();
                let size =
                    winit::dpi::PhysicalSize::from_logical::<_, f64>((width, height), scale_factor);
                _ = window.request_inner_size(size);
                let surface = render_cx
                    .create_surface(
                        window.clone(),
                        size.width,
                        size.height,
                        wgpu::PresentMode::AutoVsync,
                    )
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

fn parse_arguments() -> Args {
    // We allow baking in arguments at compile time. This is especially useful for
    // Android and WASM.
    // This is used on desktop platforms to allow debugging the same settings
    let args = if let Some(args) = option_env!("VELLO_STATIC_ARGS") {
        // We split by whitespace here to allow passing multiple arguments
        // In theory, we could do more advanced parsing/splitting (e.g. using quotes),
        // but that would require a lot more effort

        // We `chain` in a fake binary name, because clap ignores the first argument otherwise
        // Ideally, we'd use the `no_binary_name` argument, but setting that at runtime would
        // require globals or some worse hacks
        Args::parse_from(std::iter::once("with_winit").chain(args.split_ascii_whitespace()))
    } else {
        Args::parse()
    };
    args
}

#[cfg(target_os = "android")]
use winit::platform::android::activity::AndroidApp;

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;
    let config = android_logger::Config::default();
    // We allow configuring the Android logging with an environment variable at build time
    let config = if let Some(logging_config) = option_env!("VELLO_STATIC_LOG") {
        let mut filter = android_logger::FilterBuilder::new();
        filter.filter_level(log::LevelFilter::Warn);
        filter.parse(logging_config);
        let filter = filter.build();
        // This shouldn't be needed in theory, but without this the max
        // level is set to 0 (i.e. Off)
        let config = config.with_max_level(filter.filter());
        config.with_filter(filter)
    } else {
        config.with_max_level(log::LevelFilter::Warn)
    };
    android_logger::init_once(config);

    // Send tracing events to Android Trace
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    tracing_subscriber::registry()
        .with(tracing_android_trace::AndroidTraceLayer::new())
        .try_init()
        .unwrap();
    log::info!(
        "Max level: {}",
        tracing::level_filters::LevelFilter::current()
    );

    let event_loop = EventLoop::with_user_event()
        .with_android_app(app)
        .build()
        .expect("Required to continue");
    let args = parse_arguments();
    let scenes = args.args.select_scene_set().unwrap().unwrap();
    let render_cx = RenderContext::new();

    run(event_loop, args, scenes, render_cx);
}
