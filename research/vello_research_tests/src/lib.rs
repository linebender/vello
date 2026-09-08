// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello tests.

// LINEBENDER LINT SET - lib.rs - v2
// See https://linebender.org/wiki/canonical-lints/
// These lints aren't included in Cargo.toml because they
// shouldn't apply to examples and tests
#![warn(unused_crate_dependencies)]
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(
    missing_debug_implementations,
    unreachable_pub,
    missing_docs,
    clippy::missing_assert_message,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::allow_attributes_without_reason
)]

use std::env;
use std::io::ErrorKind;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use scenes::{ExampleScene, ImageCache, SceneParams, SimpleText};
use vello::kurbo::{Affine, Vec2};
use vello::peniko::{Blob, Color, ImageFormat, color::palette};
use vello::peniko::{ImageAlphaType, ImageData};
use vello::wgpu::{
    self, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, TexelCopyBufferInfo,
    TextureDescriptor, TextureFormat, TextureUsages,
};
use vello::{AaConfig, RendererOptions, Scene, util::RenderContext, util::block_on_wgpu};

mod compare;
mod snapshot;

pub use compare::{GpuCpuComparison, compare_gpu_cpu, compare_gpu_cpu_sync};
pub use snapshot::{
    Snapshot, SnapshotDirectory, smoke_snapshot_test_sync, snapshot_test, snapshot_test_sync,
};

pub struct TestParams {
    pub width: u32,
    pub height: u32,
    pub base_color: Option<Color>,
    pub use_cpu: bool,
    pub name: String,
    pub anti_aliasing: AaConfig,
}

impl TestParams {
    pub fn new(name: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            base_color: None,
            use_cpu: false,
            name: name.into(),
            anti_aliasing: AaConfig::Area,
        }
    }
}

pub fn render_then_debug_sync(scene: &Scene, params: &TestParams) -> Result<ImageData> {
    pollster::block_on(render_then_debug(scene, params))
}

pub async fn render_then_debug(scene: &Scene, params: &TestParams) -> Result<ImageData> {
    let image = get_scene_image(params, scene).await?;
    let suffix = if params.use_cpu { "cpu" } else { "gpu" };
    let name = format!("{}_{suffix}", &params.name);
    let out_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("debug_outputs")
        .join(name)
        .with_extension("png");
    if env_var_relates_to("VELLO_DEBUG_TEST", &params.name, params.use_cpu) {
        write_png_to_file(params, &out_path, &image, None, false)?;
        let (width, height) = (image.width, image.height);
        println!("Wrote debug result ({width}x{height}) to {out_path:?}");
    } else {
        match std::fs::remove_file(&out_path) {
            Ok(()) => (),
            Err(e) if e.kind() == ErrorKind::NotFound => (),
            Err(e) => return Err(e.into()),
        }
    }
    Ok(image)
}

pub async fn get_scene_image(
    params: &TestParams,
    scene: &Scene,
) -> Result<ImageData, anyhow::Error> {
    let mut context = RenderContext::new();
    let device_id = context
        .device(None)
        .await
        .ok_or_else(|| anyhow!("No compatible device found"))?;
    let device_handle = &mut context.devices[device_id];
    let device = &device_handle.device;
    let queue = &device_handle.queue;
    let mut renderer = vello::Renderer::new(
        device,
        RendererOptions {
            use_cpu: params.use_cpu,
            num_init_threads: NonZeroUsize::new(1),
            antialiasing_support: std::iter::once(params.anti_aliasing).collect(),
            pipeline_cache: None,
        },
    )
    .or_else(|_| bail!("Got non-Send/Sync error from creating renderer"))?;
    let width = params.width;
    let height = params.height;
    let render_params = vello::RenderParams {
        base_color: params.base_color.unwrap_or(palette::css::BLACK),
        width,
        height,
        antialiasing_method: params.anti_aliasing,
    };
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
        .render_to_texture(device, queue, scene, &view, &render_params)
        .or_else(|_| bail!("Got non-Send/Sync error from rendering"))?;
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
    let data = Blob::new(Arc::new(result_unpadded));
    let image = ImageData {
        data,
        format: ImageFormat::Rgba8,
        width,
        height,
        // TODO: Confirm
        alpha_type: ImageAlphaType::Alpha,
    };
    Ok(image)
}

pub fn write_png_to_file(
    params: &TestParams,
    out_path: &Path,
    image: &ImageData,
    max_size_in_bytes: Option<u64>,
    optimise: bool,
) -> Result<(), anyhow::Error> {
    if image.format != ImageFormat::Rgba8 {
        unimplemented!();
    }
    if image.alpha_type != ImageAlphaType::Alpha {
        unimplemented!()
    }
    let width = params.width;
    let height = params.height;
    let mut data = Vec::new();
    let mut encoder = png::Encoder::new(&mut data, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(image.data.data())?;
    writer.finish()?;
    if optimise {
        data = oxipng::optimize_from_memory(&data, &oxipng::Options::max_compression()).unwrap();
    }

    let size = data.len();
    std::fs::write(out_path, &data)?;
    let oversized_path = out_path.with_extension("oversized.png");
    if max_size_in_bytes
        .is_some_and(|max_size_in_bytes| u64::try_from(size).unwrap() > max_size_in_bytes)
    {
        std::fs::rename(out_path, &oversized_path)?;
        bail!(
            "File was oversized, expected {} bytes, got {size} bytes. New file written to {to}",
            max_size_in_bytes.unwrap(),
            to = oversized_path.display()
        );
    } else {
        // Intentionally do not handle errors here
        drop(std::fs::remove_file(oversized_path));
    }
    Ok(())
}

/// Determine whether the value of the environment variable `env_var`
/// includes a specific test.
/// This is used when updating tests, or dumping the debug output
fn env_var_relates_to(env_var: &'static str, name: &str, use_cpu: bool) -> bool {
    if let Ok(val) = env::var(env_var) {
        if val.eq_ignore_ascii_case("all")
            || val.eq_ignore_ascii_case("cpu") && use_cpu
            || val.eq_ignore_ascii_case("gpu") && !use_cpu
        {
            return true;
        }
        for test in val.split(',') {
            if use_cpu {
                let test_name = test.trim_end_matches("_cpu");
                if test_name.eq_ignore_ascii_case(name) {
                    return true;
                }
            } else {
                let test_name = test.trim_end_matches("_gpu");
                if test_name.eq_ignore_ascii_case(name) {
                    return true;
                }
            }
        }
    }
    false
}

pub fn encode_test_scene(mut test_scene: ExampleScene, test_params: &mut TestParams) -> Scene {
    let mut inner_scene = Scene::new();
    let mut image_cache = ImageCache::new();
    let mut text = SimpleText::new();
    let mut scene_params = SceneParams {
        base_color: None,
        complexity: 100,
        time: 0.,
        images: &mut image_cache,
        interactive: false,
        resolution: None,
        text: &mut text,
    };
    test_scene
        .function
        .render(&mut inner_scene, &mut scene_params);
    if test_params.base_color.is_none() {
        test_params.base_color = scene_params.base_color;
    }
    if let Some(resolution) = scene_params.resolution {
        // Automatically scale the rendering to fill as much of the window as possible
        let factor = Vec2::new(test_params.width as f64, test_params.height as f64);
        let scale_factor = (factor.x / resolution.x).min(factor.y / resolution.y);
        let mut outer_scene = Scene::new();
        outer_scene.append(&inner_scene, Some(Affine::scale(scale_factor)));
        outer_scene
    } else {
        inner_scene
    }
}
