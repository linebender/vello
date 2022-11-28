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

use piet_scene::{Scene, SceneBuilder};
use piet_wgsl::{util::RenderContext, Renderer, Result};

async fn run() -> Result<()> {
    use winit::{
        dpi::LogicalSize,
        event::*,
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
    };
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(LogicalSize::new(1044, 800))
        .with_resizable(true)
        .build(&event_loop)
        .unwrap();
    let render_cx = RenderContext::new().await?;
    let size = window.inner_size();
    let mut surface = render_cx.create_surface(&window, size.width, size.height);
    let mut renderer = Renderer::new(&render_cx.device)?;
    let mut simple_text = simple_text::SimpleText::new();
    let mut current_frame = 0usize;
    let mut scene_ix = 0usize;
    let mut scene = Scene::default();
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
            let mut builder = SceneBuilder::for_scene(&mut scene);
            const N_SCENES: usize = 6;
            match scene_ix % N_SCENES {
                0 => test_scene::render_anim_frame(&mut builder, &mut simple_text, current_frame),
                1 => test_scene::render_blend_grid(&mut builder),
                2 => test_scene::render_tiger(&mut builder, false),
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
                    &render_cx.device,
                    &render_cx.queue,
                    &scene,
                    &surface_texture,
                    width,
                    height,
                )
                .expect("failed to render to surface");
            surface_texture.present();
            render_cx.device.poll(wgpu::Maintain::Wait);
        }
        _ => {}
    });
}

fn main() {
    pollster::block_on(run()).unwrap();
}
