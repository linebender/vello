// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use std::env;
use std::fs::File;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use vello::peniko::{Blob, Color, Format, Image};
use vello::util::RenderContext;
use vello::wgpu::{
    self, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, ImageCopyBuffer,
    TextureDescriptor, TextureFormat, TextureUsages,
};
use vello::{block_on_wgpu, RendererOptions, Scene};

mod snapshot;

pub use snapshot::{snapshot_test, snapshot_test_sync};

pub struct TestParams {
    pub width: u32,
    pub height: u32,
    pub base_colour: Color,
    pub use_cpu: bool,
    pub name: String,
}

impl TestParams {
    pub fn new(name: impl Into<String>, width: u32, height: u32) -> Self {
        TestParams {
            width,
            height,
            base_colour: Color::BLACK,
            use_cpu: false,
            name: name.into(),
        }
    }
}

pub fn render_sync(scene: Scene, params: &TestParams) -> Result<Image> {
    pollster::block_on(render(scene, params))
}

pub async fn render(scene: Scene, params: &TestParams) -> Result<Image> {
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
            surface_format: None,
            use_cpu: params.use_cpu,
            num_init_threads: NonZeroUsize::new(1),
            antialiasing_support: vello::AaSupport::area_only(),
        },
    )
    .or_else(|_| bail!("Got non-Send/Sync error from creating renderer"))?;

    let width = params.width;
    let height = params.height;
    let render_params = vello::RenderParams {
        base_color: params.base_colour,
        width,
        height,
        antialiasing_method: vello::AaConfig::Area,
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
        .render_to_texture(device, queue, &scene, &view, &render_params)
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
        ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
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
    let image = Image::new(data, Format::Rgba8, width, height);
    if env_var_relates_to("VELLO_DEBUG_TEST", &params.name, params.use_cpu) {
        let suffix = if params.use_cpu { "cpu" } else { "gpu" };
        let name = format!("{}_{suffix}", &params.name);
        let out_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("debug_outputs")
            .join(name)
            .with_extension("png");
        write_png_to_file(params, &out_path, &image)?;
        println!("Wrote debug result ({width}x{height}) to {out_path:?}");
    }
    Ok(image)
}

pub fn write_png_to_file(
    params: &TestParams,
    out_path: &std::path::Path,
    image: &Image,
) -> Result<(), anyhow::Error> {
    let width = params.width;
    let height = params.height;
    let mut file = File::create(&out_path)?;
    let mut encoder = png::Encoder::new(&mut file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(image.data.data())?;
    writer.finish()?;
    Ok(())
}

pub fn env_var_relates_to(env_var: &'static str, name: &str, use_cpu: bool) -> bool {
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
