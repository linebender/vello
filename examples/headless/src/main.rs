use std::{
    fs::File,
    num::NonZeroU32,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Context, Result};
use clap::{CommandFactory, Parser};
use scenes::{ImageCache, SceneParams, SceneSet, SimpleText};
use vello::{
    block_on_wgpu,
    kurbo::{Affine, Vec2},
    util::RenderContext,
    RendererOptions, Scene, SceneBuilder, SceneFragment,
};
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, ImageCopyBuffer,
    TextureDescriptor, TextureFormat, TextureUsages,
};

fn main() -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let args = Args::parse();
    let scenes = args.args.select_scene_set(|| Args::command())?;
    if let Some(scenes) = scenes {
        let mut scene_idx = None;
        for (idx, scene) in scenes.scenes.iter().enumerate() {
            if scene.config.name.eq_ignore_ascii_case(&args.scene) {
                if let Some(scene_idx) = scene_idx {
                    eprintln!("Scene names conflict, skipping scene {idx} (instead rendering {scene_idx})");
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

                if !(parsed < scenes.scenes.len()) {
                    if scenes.scenes.len() == 0 {
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
    let mut context = RenderContext::new()
        .or_else(|_| bail!("Got non-Send/Sync error from creating render context"))?;
    let device_id = context
        .device(None)
        .await
        .ok_or_else(|| anyhow!("No compatible device found"))?;
    let device_handle = &mut context.devices[device_id];
    let device = &device_handle.device;
    let queue = &device_handle.queue;
    let mut renderer = vello::Renderer::new(
        &device,
        &RendererOptions {
            surface_format: None,
        },
    )
    .or_else(|_| bail!("Got non-Send/Sync error from creating renderer"))?;
    let mut fragment = SceneFragment::new();
    let mut builder = SceneBuilder::for_fragment(&mut fragment);
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
    };
    (example_scene.function)(&mut builder, &mut scene_params);
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
        transform = transform * Affine::scale(scale_factor);
        (new_width, new_height)
    } else {
        match (args.x_resolution, args.y_resolution) {
            (None, None) => (1000, 1000),
            (None, Some(y)) => {
                let y = y.try_into()?;
                (y, y)
            }
            (Some(x), None) => {
                let x = x.try_into()?;
                (x, x)
            }
            (Some(x), Some(y)) => (x.try_into()?, y.try_into()?),
        }
    };
    let render_params = vello::RenderParams {
        base_color: args
            .args
            .base_color
            .or(scene_params.base_color)
            .unwrap_or(vello::peniko::Color::BLACK),
        width,
        height,
    };
    let mut scene = Scene::new();
    let mut builder = SceneBuilder::for_scene(&mut scene);
    builder.append(&fragment, Some(transform));
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let target = device.create_texture(&TextureDescriptor {
        label: Some("Target texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    renderer
        .render_to_texture(&device, &queue, &scene, &view, &render_params)
        .or_else(|_| bail!("Got non-Send/Sync error from rendering"))?;
    // (width * 4).next_multiple_of(256)
    let padded_byte_width = {
        let w = width as u32 * 4;
        match w % 256 {
            0 => w,
            r => w + (256 - r),
        }
    };
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
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(padded_byte_width),
                rows_per_image: None,
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    let buf_slice = buffer.slice(..);

    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    if let Some(recv_result) = block_on_wgpu(&device, receiver.receive()) {
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
    let mut encoder = png::Encoder::new(&mut file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
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
    pub out_directory: PathBuf,
    #[arg(long, short, global(false))]
    /// Display a list of all scene names
    print_scenes: bool,
    #[command(flatten)]
    args: scenes::Arguments,
}

fn default_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("outputs")
}
