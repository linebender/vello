mod asset;
mod render;

use anyhow::Result;
use asset::ASSETS;
use byte_unit::Byte;
use clap::Parser;
use dialoguer::Confirm;
use render::render_svg;
use std::path::PathBuf;
use std::time::Instant;
use vello::{
    kurbo::{Affine, Vec2},
    util::RenderContext,
    Renderer, Scene, SceneBuilder,
};
use winit::{event_loop::EventLoop, window::Window};

#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// Input files for rendering. Will use builtin SVGs if empty.
    files: Vec<PathBuf>,
}

// Check if all the known assets have been downloaded.
// If some haven't been downloaded (or if the checksums don't match), find
// their combined size and licenses. Ask the user if they want to download
// the SVG files.
// If yes, download the files and return normally.
// If no, exit with status code -1
fn fetch_missing_assets() -> Result<()> {
    let missing_assets = ASSETS
        .iter()
        .filter(|asset| !asset.fetched())
        .collect::<Vec<_>>();

    if !missing_assets.is_empty() {
        let total_size = Byte::from_bytes(missing_assets.iter().map(|asset| asset.size).sum())
            .get_appropriate_unit(true);
        let mut licenses: Vec<_> = missing_assets.iter().map(|asset| asset.license).collect();
        licenses.dedup();

        println!("Some SVG assets are missing. Let me download them for you.");
        println!(
            "They'll take up {total_size} and are available under these licenses: {}",
            licenses.join(", ")
        );

        if Confirm::new()
            .with_prompt("Do you want to continue?")
            .interact()?
        {
            println!("Looks like you want to continue");
            for missing in missing_assets {
                missing.fetch()?
            }
        } else {
            println!("nevermind then :(");
            std::process::exit(1)
        }
    }
    Ok(())
}

async fn run(event_loop: EventLoop<()>, window: Window, svg_files: Vec<PathBuf>) {
    use winit::{event::*, event_loop::ControlFlow};
    let mut render_cx = RenderContext::new().unwrap();
    let size = window.inner_size();
    let mut surface = render_cx
        .create_surface(&window, size.width, size.height)
        .await;
    let device_handle = &render_cx.devices[surface.dev_id];
    let mut renderer = Renderer::new(&device_handle.device).unwrap();
    let mut current_frame = 0usize;
    let mut scene = Scene::new();
    let mut cached_svg_scene = vec![];
    cached_svg_scene.resize_with(svg_files.len(), || None);
    let mut transform = Affine::IDENTITY;
    let mut mouse_down = false;
    let mut prior_position: Option<Vec2> = None;
    let mut last_title_update = Instant::now();
    // We allow looping left and right through the svgs, so use a signed index
    let mut svg_ix: i32 = 0;
    // These are set after choosing the svg, as they overwrite the defaults specified there
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::KeyboardInput { input, .. } => {
                if input.state == ElementState::Pressed {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::Left) => {
                            svg_ix = svg_ix.saturating_sub(1);
                            transform = Affine::IDENTITY
                        }
                        Some(VirtualKeyCode::Right) => {
                            svg_ix = svg_ix.saturating_add(1);
                            transform = Affine::IDENTITY
                        }
                        Some(VirtualKeyCode::Space) => transform = Affine::IDENTITY,
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
                let modifier = if let MouseScrollDelta::PixelDelta(delta) = delta {
                    delta.y * 0.001
                } else if let MouseScrollDelta::LineDelta(_, y) = delta {
                    *y as f64 * 0.1
                } else {
                    0.0
                };
                transform = Affine::translate(prior_position.unwrap_or_default())
                    * Affine::scale(1.0 + modifier)
                    * Affine::translate(-prior_position.unwrap_or_default())
                    * transform;
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
            current_frame += 1;
            let width = surface.config.width;
            let height = surface.config.height;
            let device_handle = &render_cx.devices[surface.dev_id];
            let mut builder = SceneBuilder::for_scene(&mut scene);

            // Allow looping forever
            let svg_ix = svg_ix.rem_euclid(svg_files.len() as i32) as usize;

            render_svg(
                &mut builder,
                &mut cached_svg_scene[svg_ix],
                transform,
                &svg_files[svg_ix],
            );

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

            if current_frame % 60 == 0 {
                let now = Instant::now();
                let duration = now.duration_since(last_title_update);
                let fps = 60.0 / duration.as_secs_f64();
                window.set_title(&format!("usvg viewer - fps: {:.1}", fps));
                last_title_update = now;
            }
            device_handle.device.poll(wgpu::Maintain::Wait);
        }
        _ => {}
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let paths = if args.files.is_empty() {
        fetch_missing_assets()?;
        ASSETS.iter().map(|asset| asset.local_path()).collect()
    } else {
        args.files
    };
    use winit::{dpi::LogicalSize, window::WindowBuilder};
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(LogicalSize::new(1044, 800))
        .with_resizable(true)
        .with_title("Vello usvg viewer")
        .build(&event_loop)
        .unwrap();
    pollster::block_on(run(event_loop, window, paths));
    Ok(())
}
