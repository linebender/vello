// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Renders a biplanar `Y'CbCr` (NV12) frame to a PNG without a window.
//!
//! Hardware video decoders hand out frames as two planes: full-resolution luma and
//! half-resolution interleaved chroma. This example builds such a frame (SMPTE-style colour bars
//! over a gradient) on the CPU, binds the two planes with
//! [`TextureBindings::insert_biplanar`](vello_gpu::TextureBindings::insert_biplanar) and draws
//! them at twice their size; the strip shader converts the samples to RGB while sampling, so no
//! conversion pass runs.
//!
//! ```sh
//! cargo run -p vello_gpu --example render_nv12_to_file -- out.png [bt601|bt709|bt2020] [limited|full]
//! ```

use std::io::BufWriter;
use vello_common::geometry::RectU16;
use vello_common::kurbo::{Affine, Rect};
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::{Extend, ImageAlphaType, ImageQuality, ImageSampler};
use vello_common::pixmap::{PixelMetadata, Pixmap};
use vello_gpu::{Scene, TextureId, YuvFormat, YuvMatrix, YuvRange};

const FRAME_WIDTH: u16 = 256;
const FRAME_HEIGHT: u16 = 144;
const SCALE: u16 = 2;

fn main() {
    let mut args = std::env::args().skip(1);
    let output_filename = args
        .next()
        .expect("output PNG filename is the first argument");
    let matrix = match args.next().as_deref() {
        None | Some("bt709") => YuvMatrix::Bt709,
        Some("bt601") => YuvMatrix::Bt601,
        Some("bt2020") => YuvMatrix::Bt2020Ncl,
        Some(other) => panic!("unknown matrix {other:?}: expected bt601, bt709 or bt2020"),
    };
    let range = match args.next().as_deref() {
        None | Some("limited") => YuvRange::Limited,
        Some("full") => YuvRange::Full,
        Some(other) => panic!("unknown range {other:?}: expected limited or full"),
    };
    let format = YuvFormat {
        matrix,
        range,
        ..YuvFormat::NV12_BT709_LIMITED
    };
    pollster::block_on(run(&output_filename, format));
}

/// The frame as RGB: seven colour bars over the top two thirds, a gradient below.
fn frame_rgb(x: u16, y: u16) -> [f64; 3] {
    const BARS: [[f64; 3]; 7] = [
        [0.75, 0.75, 0.75],
        [0.75, 0.75, 0.0],
        [0.0, 0.75, 0.75],
        [0.0, 0.75, 0.0],
        [0.75, 0.0, 0.75],
        [0.75, 0.0, 0.0],
        [0.0, 0.0, 0.75],
    ];
    if y < FRAME_HEIGHT * 2 / 3 {
        BARS[usize::from(x) * BARS.len() / usize::from(FRAME_WIDTH)]
    } else {
        let t = f64::from(x) / f64::from(FRAME_WIDTH - 1);
        [t, 0.5, 1.0 - t]
    }
}

/// Encodes the frame as NV12 planes: a luma byte per pixel and one (Cb, Cr) pair per 2×2 block,
/// chroma co-sited with the left luma column (the format's `ChromaSiting::Left`).
#[expect(
    clippy::cast_possible_truncation,
    reason = "codes are clamped to the sample range before the cast"
)]
fn nv12_planes(format: YuvFormat) -> (Vec<u8>, Vec<u8>) {
    let (kr, kb) = match format.matrix {
        YuvMatrix::Bt601 => (0.299, 0.114),
        YuvMatrix::Bt709 => (0.2126, 0.0722),
        YuvMatrix::Bt2020Ncl => (0.2627, 0.0593),
    };
    let (luma_black, luma_range, chroma_range) = match format.range {
        YuvRange::Limited => (16.0, 219.0, 224.0),
        YuvRange::Full => (0.0, 255.0, 255.0),
    };
    let ypbpr = |x: u16, y: u16| {
        let [r, g, b] = frame_rgb(x, y);
        let luma = kr * r + (1.0 - kr - kb) * g + kb * b;
        (
            luma,
            (b - luma) / (2.0 * (1.0 - kb)),
            (r - luma) / (2.0 * (1.0 - kr)),
        )
    };
    let code = |value: f64| value.round().clamp(0.0, 255.0) as u8;
    let mut luma = Vec::with_capacity(usize::from(FRAME_WIDTH) * usize::from(FRAME_HEIGHT));
    for y in 0..FRAME_HEIGHT {
        for x in 0..FRAME_WIDTH {
            luma.push(code(luma_black + luma_range * ypbpr(x, y).0));
        }
    }
    let mut chroma = Vec::with_capacity(luma.len() / 2);
    for cy in 0..FRAME_HEIGHT / 2 {
        for cx in 0..FRAME_WIDTH / 2 {
            let (_, pb0, pr0) = ypbpr(2 * cx, 2 * cy);
            let (_, pb1, pr1) = ypbpr(2 * cx, 2 * cy + 1);
            chroma.push(code(128.0 + chroma_range * (pb0 + pb1) / 2.0));
            chroma.push(code(128.0 + chroma_range * (pr0 + pr1) / 2.0));
        }
    }
    (luma, chroma)
}

fn plane_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    bytes_per_texel: u32,
    data: &[u8],
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * bytes_per_texel),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

async fn run(output_filename: &str, format: YuvFormat) {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find an appropriate adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            ..Default::default()
        })
        .await
        .expect("Failed to create device");

    // The decoded frame: two planes, bound under one texture id.
    let (luma, chroma) = nv12_planes(format);
    let luma_view = plane_texture(
        &device,
        &queue,
        "NV12 luma plane",
        wgpu::TextureFormat::R8Unorm,
        FRAME_WIDTH.into(),
        FRAME_HEIGHT.into(),
        1,
        &luma,
    );
    let chroma_view = plane_texture(
        &device,
        &queue,
        "NV12 chroma plane",
        wgpu::TextureFormat::Rg8Unorm,
        u32::from(FRAME_WIDTH) / 2,
        u32::from(FRAME_HEIGHT) / 2,
        2,
        &chroma,
    );
    let frame_id = TextureId(1);
    let mut texture_bindings = vello_gpu::TextureBindings::new();
    texture_bindings.insert_biplanar(frame_id, luma_view, chroma_view, format);

    // A scene that paints the frame at twice its size.
    let width = FRAME_WIDTH * SCALE;
    let height = FRAME_HEIGHT * SCALE;
    let mut scene = Scene::new(width, height);
    scene.set_paint_transform(Affine::scale(f64::from(SCALE)));
    scene.set_paint(Image {
        image: ImageSource::external_texture(
            frame_id,
            RectU16::new(0, 0, FRAME_WIDTH, FRAME_HEIGHT),
            false,
        ),
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality: ImageQuality::Medium,
            alpha: 1.0,
        },
    });
    scene.fill_rect(&Rect::new(0.0, 0.0, width.into(), height.into()));

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Render Target"),
        size: wgpu::Extent3d {
            width: width.into(),
            height: height.into(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let (mut renderer, mut resources) = vello_gpu::Renderer::new(
        &device,
        &vello_gpu::RenderTargetConfig {
            format: texture.format(),
            width,
            height,
        },
    );
    let render_size = vello_gpu::RenderSize { width, height };
    let depth_texture_view = vello_gpu::Renderer::create_depth_texture_view(&device, &render_size);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Vello Render NV12"),
    });
    renderer
        .render(
            &scene,
            &mut resources,
            &device,
            &queue,
            &mut encoder,
            &render_size,
            &texture_view,
            Some(&depth_texture_view),
            &texture_bindings,
            vello_gpu::TargetInit::Clear(vello_gpu::ClearSettings::default()),
        )
        .unwrap();

    // Read the target back and write it as a PNG.
    let bytes_per_row = (u32::from(width) * 4).next_multiple_of(256);
    let texture_copy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &texture_copy_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: width.into(),
            height: height.into(),
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    texture_copy_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| {
            result.expect("Failed to map texture for reading");
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mut img_data = Vec::with_capacity(usize::from(width) * usize::from(height) * 4);
    for row in texture_copy_buffer
        .slice(..)
        .get_mapped_range()
        .unwrap()
        .chunks_exact(bytes_per_row as usize)
    {
        img_data.extend_from_slice(&row[0..usize::from(width) * 4]);
    }
    texture_copy_buffer.unmap();

    let pixmap = Pixmap::from_parts(img_data, width, height, PixelMetadata::default());
    let file = std::fs::File::create(output_filename).unwrap();
    let mut png_encoder = png::Encoder::new(BufWriter::new(file), width.into(), height.into());
    png_encoder.set_color(png::ColorType::Rgba);
    let mut writer = png_encoder.write_header().unwrap();
    writer
        .write_image_data(&pixmap.take_rgba8(ImageAlphaType::Alpha))
        .unwrap();
}
