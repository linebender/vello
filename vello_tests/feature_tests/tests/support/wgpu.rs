// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{Pixmap, ProbeFeature, probe};
use vello_common::{color::palette::css, paint::ImageSource, pixmap::PixelMetadata};
use vello_gpu::{
    ClearSettings, RenderSize, RenderTargetConfig, Renderer, Scene, TargetInit, TextureBindings,
};

pub(crate) fn render(elements: &[ProbeFeature]) -> Pixmap {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("WGPU feature tests require a GPU adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let (width, height) = probe::canvas_size(elements);
    let config = RenderTargetConfig {
        width,
        height,
        format: wgpu::TextureFormat::Rgba8Unorm,
    };
    let (mut renderer, mut resources) = Renderer::new(&device, &config);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let image = probe::probe_image_pixmap();
    let image_id = renderer.upload_image(&mut resources, &device, &queue, &mut encoder, &image);
    let mut scene = Scene::new(width, height);

    probe::draw_scene(
        &mut scene,
        ImageSource::opaque_id_with_transparency_hint(image_id, image.may_have_transparency()),
        elements,
    );

    let size = RenderSize { width, height };
    let extent = wgpu::Extent3d {
        width: width.into(),
        height: height.into(),
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Feature test target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let depth = Renderer::create_depth_texture_view(&device, &size);
    renderer
        .render(
            &scene,
            &mut resources,
            &device,
            &queue,
            &mut encoder,
            &size,
            &texture.create_view(&wgpu::TextureViewDescriptor::default()),
            Some(&depth),
            &TextureBindings::new(),
            TargetInit::Clear(ClearSettings::Viewport { color: css::WHITE }),
        )
        .unwrap();

    let bytes_per_row = (u32::from(width) * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Feature test readback"),
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
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        extent,
    );
    queue.submit([encoder.finish()]);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mut pixels = Vec::with_capacity(usize::from(width) * usize::from(height) * 4);

    {
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        for row in mapped.chunks_exact(usize::try_from(bytes_per_row).unwrap()) {
            pixels.extend_from_slice(&row[..usize::from(width) * 4]);
        }
    }

    buffer.unmap();

    Pixmap::from_parts(pixels, width, height, PixelMetadata::default())
}
