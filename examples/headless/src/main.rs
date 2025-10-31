// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Headless

// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(
    clippy::cast_possible_truncation,
    clippy::allow_attributes_without_reason
)]

use std::fs::File;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use scenes::{ImageCache, SceneParams, SceneSet, SimpleText};
use vello::graph::{Canvas, Gallery, OutputSize, PaintingDescriptor, Vello};
use vello::kurbo::{Affine, Vec2};
use vello::peniko::color::palette;
use vello::util::RenderContext;
use vello::wgpu::{
    self, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, TexelCopyBufferInfo,
    TextureUsages,
};
use vello::{RendererOptions, Scene, util::block_on_wgpu};

fn main() -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let args = Args::parse();
    let scenes = args.args.select_scene_set()?;
    if let Some(scenes) = scenes {
        let mut scene_idx = None;
        for (idx, scene) in scenes.scenes.iter().enumerate() {
            if scene.config.name.eq_ignore_ascii_case(&args.scene) {
                if let Some(scene_idx) = scene_idx {
                    eprintln!(
                        "Scene names conflict, skipping scene {idx} (instead rendering {scene_idx})"
                    );
                } else {
                    scene_idx = Some(idx);
                }
            }
        }
        let scene_idx = match scene_idx {
            Some(idx) => idx,
            None => {
                let parsed = args.scene.parse::<usize>().context(format!(
                    "'{}' didn't match any scene, trying to parse as index",
                    args.scene
                ))?;

                if parsed >= scenes.scenes.len() {
                    if scenes.scenes.is_empty() {
                        bail!("Cannot select a scene, as there are no scenes")
                    }
                    bail!(
                        "{parsed} doesn't fit in scenes (len {})",
                        scenes.scenes.len()
                    );
                }
                parsed
            }
        };
        if args.print_scenes {
            println!("Available scenes:");

            for (idx, scene) in scenes.scenes.iter().enumerate() {
                println!(
                    "{idx}: {}{}{}",
                    scene.config.name,
                    if scene.config.animated {
                        " (animated)"
                    } else {
                        ""
                    },
                    if scene_idx == idx { " (selected)" } else { "" }
                );
            }
            return Ok(());
        }
        pollster::block_on(render(scenes, scene_idx, &args))?;
    }
    Ok(())
}

async fn render(mut scenes: SceneSet, index: usize, args: &Args) -> Result<()> {
    let mut context = RenderContext::new();
    let device_id = context
        .device(None)
        .await
        .ok_or_else(|| anyhow!("No compatible device found"))?;
    let device_handle = &mut context.devices[device_id];
    let device = &device_handle.device;
    let queue = &device_handle.queue;
    let mut vello = Vello::new(
        device.clone(),
        RendererOptions {
            use_cpu: args.use_cpu,
            num_init_threads: NonZeroUsize::new(1),
            antialiasing_support: vello::AaSupport::area_only(),
            ..Default::default()
        },
    )?;
    let mut gallery = Gallery::new(device.clone(), "Main Thread");
    let mut fragment = Scene::new();
    let example_scene = &mut scenes.scenes[index];
    let mut text = SimpleText::new();
    let mut images = ImageCache::new();
    let mut scene_params = SceneParams {
        time: args.time.unwrap_or(0.),
        text: &mut text,
        images: &mut images,
        resolution: None,
        base_color: None,
        interactive: false,
        complexity: 0,
    };
    example_scene
        .function
        .render(&mut fragment, &mut scene_params);
    let mut transform = Affine::IDENTITY;
    let (width, height) = if let Some(resolution) = scene_params.resolution {
        let ratio = resolution.x / resolution.y;
        let (new_width, new_height) = match (args.x_resolution, args.y_resolution) {
            (None, None) => (resolution.x.ceil() as u32, resolution.y.ceil() as u32),
            (None, Some(y)) => ((ratio * (y as f64)).ceil() as u32, y),
            (Some(x), None) => (x, ((x as f64) / ratio).ceil() as u32),
            (Some(x), Some(y)) => (x, y),
        };
        let factor = Vec2::new(new_width as f64, new_height as f64);
        let scale_factor = (factor.x / resolution.x).min(factor.y / resolution.y);
        transform *= Affine::scale(scale_factor);
        (new_width, new_height)
    } else {
        match (args.x_resolution, args.y_resolution) {
            (None, None) => (1000, 1000),
            (None, Some(y)) => (y, y),
            (Some(x), None) => (x, x),
            (Some(x), Some(y)) => (x, y),
        }
    };

    let (sub_scene_width, sub_scene_height) = (width / 2, height / 2);
    let _render_params = vello::RenderParams {
        // TODO: Pass in base_color somewhere
        base_color: args
            .args
            .base_color
            .or(scene_params.base_color)
            .unwrap_or(palette::css::BLACK),
        width,
        height,
        antialiasing_method: vello::AaConfig::Area,
    };
    let mut scene = Scene::new();
    scene.append(&fragment, Some(transform.then_scale(0.5)));

    let inner_scene = gallery.create_painting(PaintingDescriptor {
        label: "ExampleScene".into(),
        usages: TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC,
    });
    inner_scene.paint_scene(
        scene.into(),
        OutputSize {
            width: sub_scene_width,
            height: sub_scene_height,
        },
    );

    let blurred = gallery.create_painting(PaintingDescriptor {
        label: "Blurred Result".into(),
        usages: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
    });
    blurred.paint_blur(inner_scene.clone());
    let painting = gallery.create_painting(PaintingDescriptor {
        label: "Main Scene".into(),
        usages: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
    });

    let mut canvas = Canvas::new();
    canvas.draw_painting(
        inner_scene.clone(),
        sub_scene_width.try_into().unwrap(),
        sub_scene_height.try_into().unwrap(),
        Affine::IDENTITY,
    );
    canvas.draw_painting(
        blurred,
        sub_scene_width.try_into().unwrap(),
        sub_scene_height.try_into().unwrap(),
        Affine::translate((0., height as f64 / 2.)),
    );
    painting.paint_scene(canvas, OutputSize { width, height });
    let render_details = vello.prepare_render(painting, &mut gallery);

    let target = vello.render_to_texture(device, queue, render_details);
    let padded_byte_width = (width * 4).next_multiple_of(256);
    let buffer_size = padded_byte_width as u64 * height as u64;
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("val"),
        size: buffer_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Copy out buffer"),
    });

    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_byte_width),
                rows_per_image: None,
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    let buf_slice = buffer.slice(..);

    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    if let Some(recv_result) = block_on_wgpu(device, receiver.receive()) {
        recv_result?;
    } else {
        bail!("channel was closed");
    }

    let data = buf_slice.get_mapped_range();
    let mut result_unpadded = Vec::<u8>::with_capacity((width * height * 4).try_into()?);
    for row in 0..height {
        let start = (row * padded_byte_width).try_into()?;
        result_unpadded.extend(&data[start..start + (width * 4) as usize]);
    }
    let out_path = args
        .out_directory
        .join(&example_scene.config.name)
        .with_extension("png");
    let mut file = File::create(&out_path)?;
    let mut png_encoder = png::Encoder::new(&mut file, width, height);
    png_encoder.set_color(png::ColorType::Rgba);
    png_encoder.set_depth(png::BitDepth::Eight);
    let mut writer = png_encoder.write_header()?;
    writer.write_image_data(&result_unpadded)?;
    writer.finish()?;
    println!("Wrote result ({width}x{height}) to {out_path:?}");
    Ok(())
}

#[derive(Parser, Debug)]
#[command(about, long_about = None, bin_name="cargo run -p headless --")]
struct Args {
    #[arg(long, short, global(false))]
    x_resolution: Option<u32>,
    #[arg(long, short, global(false))]
    y_resolution: Option<u32>,
    /// Which scene (name) to render
    /// If no scenes have that name, an index can be specified instead
    #[arg(long, short, default_value = "0", global(false))]
    scene: String,
    #[arg(long, short, global(false))]
    /// The time in seconds since the frame start, for animated scenes
    time: Option<f64>,
    /// Directory to store the result into
    #[arg(long, default_value_os_t = default_directory())]
    out_directory: PathBuf,
    #[arg(long, short, global(false))]
    /// Display a list of all scene names
    print_scenes: bool,
    #[arg(long)]
    /// Whether to use CPU shaders
    use_cpu: bool,
    #[command(flatten)]
    args: scenes::Arguments,
}

fn default_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("outputs")
}
