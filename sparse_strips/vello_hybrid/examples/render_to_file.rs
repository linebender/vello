// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SVG rendering example for headless rendering
//!
//! This example demonstrates rendering an SVG file without a window or display.
//! It takes an input SVG file and renders it to a PNG file using the hybrid CPU/GPU renderer.

mod common;

use common::render_svg;
use std::io::BufWriter;
use vello_common::pico_svg::PicoSvg;
use vello_common::pixmap::Pixmap;
use vello_hybrid::{DimensionConstraints, Scene};

/// Main entry point for the headless rendering example.
/// Takes two command line arguments:
/// - Input SVG filename to render
/// - Output PNG filename to save the rendered result
///
/// Renders the SVG using the hybrid CPU/GPU renderer and saves the output as a PNG file.
fn main() {
    pollster::block_on(run());
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "Width and height are expected to fit within u16 range"
)]
async fn run() {
    let mut args = std::env::args().skip(1);
    let svg_filename: String = args.next().expect("svg filename is first arg");
    let output_filename: String = args.next().expect("output filename is second arg");
    let svg = std::fs::read_to_string(svg_filename).expect("error reading file");
    let render_scale = 5.0;
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing SVG");

    let constraints = DimensionConstraints::default();
    let svg_width = parsed.size.width * render_scale;
    let svg_height = parsed.size.height * render_scale;
    let (width, height) = constraints.calculate_dimensions(svg_width, svg_height);

    let mut scene = Scene::new(width as u16, height as u16);
    render_svg(&mut scene, render_scale, &parsed.items);

    // Initialize wgpu device and queue for GPU rendering
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("Failed to find an appropriate adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .expect("Failed to create device");

    // Create a render target texture
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Render Target"),
        size: wgpu::Extent3d {
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Create renderer and render the scene to the texture
    let mut renderer = vello_hybrid::Renderer::new(&device, &vello_hybrid::RendererOptions {});
    let render_params = vello_hybrid::RenderParams {
        base_color: Some(peniko::Color::TRANSPARENT),
        width: width as u32,
        height: height as u32,
        strip_height: vello_common::strip::STRIP_HEIGHT as u32,
    };
    renderer.prepare(&device, &queue, &scene, &render_params);
    renderer.render_to_texture(&device, &queue, &scene, &texture_view, &render_params);

    // Create a buffer to copy the texture data
    let bytes_per_row = (width as u32 * 4).next_multiple_of(256);
    let buffer_size = (bytes_per_row * height as u32) as u64;
    let texture_copy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy texture to buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copy Encoder"),
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
            width: width as u32,
            height: height as u32,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));

    // Map the buffer for reading
    texture_copy_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            if result.is_err() {
                panic!("Failed to map texture for reading");
            }
        });
    device.poll(wgpu::Maintain::Wait);

    // Read back the pixel data
    let mut img_data = Vec::with_capacity((width as u32 * height as u32 * 4) as usize);
    for row in texture_copy_buffer
        .slice(..)
        .get_mapped_range()
        .chunks_exact(bytes_per_row as usize)
    {
        img_data.extend_from_slice(&row[0..width as usize * 4]);
    }
    texture_copy_buffer.unmap();

    // Convert BGRA to RGBA
    let mut rgba_buffer = Vec::with_capacity(img_data.len());
    for chunk in img_data.chunks_exact(4) {
        rgba_buffer.push(chunk[2]); // R (was B)
        rgba_buffer.push(chunk[1]); // G (unchanged)
        rgba_buffer.push(chunk[0]); // B (was R)
        rgba_buffer.push(chunk[3]); // A (unchanged)
    }

    // Create a pixmap and set the buffer
    let mut pixmap = Pixmap::new(width as u16, height as u16);
    pixmap.buf = rgba_buffer;
    pixmap.unpremultiply();

    // Write the pixmap to a file
    let file = std::fs::File::create(output_filename).unwrap();
    let w = BufWriter::new(file);
    let mut png_encoder = png::Encoder::new(w, width as u32, height as u32);
    png_encoder.set_color(png::ColorType::Rgba);
    let mut writer = png_encoder.write_header().unwrap();
    writer.write_image_data(&pixmap.buf).unwrap();
}
