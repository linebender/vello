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
use vello::kurbo::Circle;

use anyhow::Result;
use vello::peniko::{Brush, Color};
use vello::util::RenderSurface;
use vello::{kurbo::Affine, util::RenderContext, AaConfig, Renderer, Scene, SceneBuilder};
use vello::{BumpAllocators, RendererOptions};

use winit::{event_loop::EventLoop, window::Window};

use crate::simple_text::SimpleText;

pub mod simple_text;
mod stats;

struct RenderState {
    // SAFETY: We MUST drop the surface before the `window`, so the fields
    // must be in this order
    surface: RenderSurface,
    window: Window,
}

fn run(event_loop: EventLoop<()>) {
    use winit::{event::*, event_loop::ControlFlow};
    let mut renderers: Vec<Option<Renderer>> = vec![];
    let mut render_cx = RenderContext::new().unwrap();
    let mut render_state = None::<RenderState>;
    let use_cpu = false;
    // Whilst suspended, we drop `render_state`, but need to keep the same window.
    // If render_state exists, we must store the window in it, to maintain drop order
    let mut cached_window = None;

    let mut scene = Scene::new();
    let mut simple_text = SimpleText::new();
    let mut stats = stats::Stats::new();
    let mut stats_shown = false;
    let mut scene_complexity: Option<BumpAllocators> = None;
    let mut complexity_shown = false;
    let mut vsync_on = true;

    let mut frame_start_time = Instant::now();

    #[allow(unused)]
    let start = Instant::now();

    let mut profile_stored = None;
    let mut profile_taken = Instant::now();

    let mut modifiers = ModifiersState::default();
    // _event_loop is used on non-wasm platforms to create new windows
    event_loop.run(move |event, _event_loop, control_flow| match event {
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
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::ModifiersChanged(m) => modifiers = *m,
                WindowEvent::KeyboardInput { input, .. } => {
                    if input.state == ElementState::Pressed {
                        match input.virtual_keycode {
                            Some(VirtualKeyCode::S) => {
                                // Toggle showing the stats screen
                                stats_shown = !stats_shown;
                            }
                            Some(VirtualKeyCode::D) => {
                                // Toggle showing the complexity menu
                                complexity_shown = !complexity_shown;
                            }
                            Some(VirtualKeyCode::C) => {
                                stats.clear_min_and_max();
                            }
                            Some(VirtualKeyCode::V) => {
                                // Toggle vsync
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
                            Some(VirtualKeyCode::Escape) => {
                                *control_flow = ControlFlow::Exit;
                            }
                            _ => {}
                        }
                    }
                }
                WindowEvent::Resized(size) => {
                    render_cx.resize_surface(&mut render_state.surface, size.width, size.height);
                    render_state.window.request_redraw();
                }
                _ => {}
            }
        }
        Event::MainEventsCleared => {
            if let Some(render_state) = &mut render_state {
                render_state.window.request_redraw();
            }
        }
        Event::RedrawRequested(_) => {
            let Some(render_state) = &mut render_state else {
                return;
            };
            let width = render_state.surface.config.width;
            let height = render_state.surface.config.height;
            let device_handle = &render_cx.devices[render_state.surface.dev_id];
            let snapshot = stats.snapshot();

            let mut builder = SceneBuilder::for_scene(&mut scene);
            {
                // Draw your scene here
                builder.fill(
                    vello::peniko::Fill::NonZero,
                    Affine::skew(1.0, 1.5),
                    &Brush::Solid(Color::rgb(255., 100., 0.)),
                    None,
                    &Circle::new((300., 300.), 200.),
                )
            }

            // The base colour is the background colour
            let base_color = Color::BLACK;
            let render_params = vello::RenderParams {
                base_color,
                width,
                height,
                antialiasing_method: AaConfig::Area,
            };

            if stats_shown {
                snapshot.draw_layer(
                    &mut builder,
                    &mut simple_text,
                    width as f64,
                    height as f64,
                    stats.samples(),
                    complexity_shown.then_some(scene_complexity).flatten(),
                    vsync_on,
                    AaConfig::Area,
                );
                if let Some(profiling_result) = renderers[render_state.surface.dev_id]
                    .as_mut()
                    .and_then(|it| it.profile_result.take())
                {
                    if profile_stored.is_none() || profile_taken.elapsed() > Duration::from_secs(1)
                    {
                        profile_stored = Some(profiling_result);
                        profile_taken = Instant::now();
                    }
                }
                if let Some(profiling_result) = profile_stored.as_ref() {
                    stats::draw_gpu_profiling(
                        &mut builder,
                        &mut simple_text,
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

            surface_texture.present();
            device_handle.device.poll(wgpu::Maintain::Poll);

            let new_time = Instant::now();
            stats.add_sample(stats::Sample {
                frame_time_us: (new_time - frame_start_time).as_micros() as u64,
            });
            frame_start_time = new_time;
        }
        Event::Suspended => {
            // When we suspend, we need to remove the `wgpu` Surface
            if let Some(render_state) = render_state.take() {
                cached_window = Some(render_state.window);
            }
            *control_flow = ControlFlow::Wait;
        }
        Event::Resumed => {
            let Option::None = render_state else { return };
            let window = cached_window
                .take()
                .unwrap_or_else(|| create_window(_event_loop));
            let size = window.inner_size();
            let surface_future = render_cx.create_surface(&window, size.width, size.height);
            // We need to block here, in case a Suspended event appeared
            let surface = pollster::block_on(surface_future).expect("Error creating surface");
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
                            timestamp_period: render_cx.devices[id].queue.get_timestamp_period(),
                            use_cpu,
                            antialiasing_support: vello::AaSupport::all(),
                        },
                    )
                    .expect("Could create renderer")
                });
                Some(render_state)
            };
            *control_flow = ControlFlow::Poll;
        }
        _ => {}
    });
}

fn create_window(event_loop: &winit::event_loop::EventLoopWindowTarget<()>) -> Window {
    use winit::{dpi::LogicalSize, window::WindowBuilder};
    WindowBuilder::new()
        .with_inner_size(LogicalSize::new(1044, 800))
        .with_resizable(true)
        .with_title("Vello demo")
        .build(event_loop)
        .unwrap()
}

pub fn main() -> Result<()> {
    // TODO: initializing both env_logger and console_logger fails on wasm.
    // Figure out a more principled approach.
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    let event_loop = EventLoop::new();
    run(event_loop);

    Ok(())
}
