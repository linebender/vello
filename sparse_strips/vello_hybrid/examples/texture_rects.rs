// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Example demonstrating external texture sources.
//!
//! Usage:
//! ```shell
//! cargo run -p vello_hybrid --example texture_rects -- output.png
//! ```

use std::io::{BufWriter, Cursor};
use vello_common::color::palette::css::WHITE;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, Circle, Rect, Shape};
use vello_common::peniko::color::palette::css;
use vello_common::peniko::color::{AlphaColor, Srgb};
use vello_common::peniko::{Extend, ImageQuality};
use vello_hybrid::{
    RenderSize, RenderTargetConfig, Scene, TextureBindings, TextureId, TextureRect,
};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const GLYPH_ATLAS_PNG: &[u8] = include_bytes!("assets/glyphs_colr_noto.png");

fn load_rgba_png(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::ALPHA | png::Transformations::STRIP_16);

    let mut reader = decoder.read_info().expect("Failed to decode atlas PNG");
    assert_eq!(
        reader.output_color_type(),
        (png::ColorType::Rgba, png::BitDepth::Eight),
        "The texture atlas example expects an RGBA8 image",
    );

    let (width, height) = {
        let info = reader.info();
        (info.width, info.height)
    };
    let mut rgba = vec![0; reader.output_buffer_size().unwrap_or_default()];
    reader
        .next_frame(&mut rgba)
        .expect("Failed to read atlas PNG frame");

    // The blend pipeline expects color channels to be premultiplied by the alpha channel.
    for pixel in rgba.chunks_exact_mut(4) {
        let a = pixel[3] as u16;
        pixel[0] = ((pixel[0] as u16 * a + 128) / 255) as u8;
        pixel[1] = ((pixel[1] as u16 * a + 128) / 255) as u8;
        pixel[2] = ((pixel[2] as u16 * a + 128) / 255) as u8;
    }

    (rgba, width, height)
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let output_filename: String = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "texture_rects_output.png".into());

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

    let render_size = RenderSize {
        width: WIDTH,
        height: HEIGHT,
    };

    let target_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Render target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut renderer = vello_hybrid::Renderer::new(
        &device,
        &RenderTargetConfig {
            format: target_texture.format(),
            width: WIDTH,
            height: HEIGHT,
        },
    );

    // Create the source texture from a bundled sprite atlas.
    let (atlas_rgba, atlas_width, atlas_height) = load_rgba_png(GLYPH_ATLAS_PNG);
    let source_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Glyph atlas"),
        size: wgpu::Extent3d {
            width: atlas_width,
            height: atlas_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &source_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &atlas_rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(atlas_width * 4),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: atlas_width,
            height: atlas_height,
            depth_or_array_layers: 1,
        },
    );
    let source_view = source_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let width_u16 = u16::try_from(WIDTH).unwrap();
    let height_u16 = u16::try_from(HEIGHT).unwrap();
    let mut scene = Scene::new(width_u16, height_u16);
    let src_full = Rect::new(0.0, 0.0, atlas_width as f64, atlas_height as f64);
    let sprite_cells = [
        Rect::new(0.0, 3.0, 62.0, 61.0),
        Rect::new(62.0, 3.0, 124.0, 61.0),
        Rect::new(124.0, 3.0, 185.0, 61.0),
        Rect::new(185.0, 3.0, 250.0, 61.0),
    ];

    scene.set_paint(WHITE);
    scene.fill_rect(&Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64));
    scene.set_paint(AlphaColor::<Srgb>::new([0.97, 0.92, 0.76, 1.]));
    for panel in [
        Rect::new(20.0, 20.0, 308.0, 132.0),
        Rect::new(328.0, 20.0, 492.0, 132.0),
        Rect::new(20.0, 188.0, 156.0, 320.0),
        Rect::new(178.0, 188.0, 334.0, 320.0),
        Rect::new(356.0, 188.0, 492.0, 320.0),
        Rect::new(20.0, 356.0, 156.0, 492.0),
        Rect::new(178.0, 356.0, 334.0, 492.0),
        Rect::new(356.0, 356.0, 492.0, 492.0),
    ] {
        scene.fill_rect(&panel);
    }

    // Top-left: four texel-aligned draws.
    scene.fill_texture_rects(
        TextureId(0),
        ImageQuality::Low,
        Extend::Pad,
        Extend::Pad,
        sprite_cells
            .iter()
            .copied()
            .zip([30., 98., 166., 234.])
            .map(|(src, x)| TextureRect {
                src,
                transform: Affine::translate((x, 48.)),
            }),
    );

    // Top-right: draw the full texture, but scaled down.
    scene.fill_texture_rects(
        TextureId(0),
        ImageQuality::Medium,
        Extend::Pad,
        Extend::Pad,
        [TextureRect {
            src: src_full,
            transform: Affine::scale(148. / src_full.width()).then_translate((336., 52.).into()),
        }],
    );

    // Middle-left: half-pixel offset on both axes.
    scene.fill_texture_rects(
        TextureId(0),
        ImageQuality::Medium,
        Extend::Pad,
        Extend::Pad,
        [TextureRect {
            src: sprite_cells[1],
            transform: Affine::translate((50.5, 223.5)),
        }],
    );

    // Middle-center: Clip with a circle.
    let clip_circle = Circle::new((256.0, 252.0), 25.0).to_path(0.1);
    scene.push_clip_layer(&clip_circle);
    scene.fill_texture_rects(
        TextureId(0),
        ImageQuality::Low,
        Extend::Pad,
        Extend::Pad,
        [
            TextureRect {
                src: sprite_cells[1],
                transform: Affine::translate((214., 223.)),
            },
            TextureRect {
                src: sprite_cells[3],
                transform: Affine::translate((244., 214.)),
            },
        ],
    );
    scene.pop_layer();

    // Middle-right: blur filtered.
    scene.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 4.0,
        edge_mode: EdgeMode::None,
    }));
    scene.fill_texture_rects(
        TextureId(0),
        ImageQuality::Low,
        Extend::Pad,
        Extend::Pad,
        [TextureRect {
            src: sprite_cells[0],
            transform: Affine::translate((392.0, 224.0)),
        }],
    );
    scene.pop_layer();

    // Bottom: occlude left one with geometry, skew middle one, scale right one.
    scene.set_paint(css::GOLD);
    scene.fill_path(&Circle::new((53.0, 427.0), 28.0).to_path(0.1));
    scene.fill_texture_rects(
        TextureId(0),
        ImageQuality::Medium,
        Extend::Pad,
        Extend::Pad,
        [
            TextureRect {
                src: sprite_cells[2],
                transform: Affine::translate((48., 394.)),
            },
            TextureRect {
                src: sprite_cells[2],
                transform: Affine::skew(1., 0.).then_translate((195., 394.).into()),
            },
            TextureRect {
                src: sprite_cells[2],
                transform: Affine::scale(1.75).then_translate((372., 374.).into()),
            },
        ],
    );
    scene.set_paint(css::ORANGE_RED);
    scene.fill_rect(&Rect::new(49., 430., 109., 468.));

    // We bind the texture when it's time for rendering.
    let mut bindings = TextureBindings::new();
    bindings.insert(TextureId(0), &source_view);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Scene Render"),
    });
    renderer
        .render_with_textures(
            &scene,
            &device,
            &queue,
            &mut encoder,
            &render_size,
            &target_view,
            &bindings,
        )
        .unwrap();

    // Read back and write PNG.
    let bytes_per_row = (WIDTH * 4).next_multiple_of(256);
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: u64::from(bytes_per_row) * u64::from(HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    readback_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| {
            result.expect("Failed to map buffer");
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mut img_data = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for row in readback_buffer
        .slice(..)
        .get_mapped_range()
        .chunks_exact(bytes_per_row as usize)
    {
        img_data.extend_from_slice(&row[..WIDTH as usize * 4]);
    }
    readback_buffer.unmap();

    let file = std::fs::File::create(&output_filename).unwrap();
    let w = BufWriter::new(file);
    let mut png_encoder = png::Encoder::new(w, WIDTH, HEIGHT);
    png_encoder.set_color(png::ColorType::Rgba);
    let mut writer = png_encoder.write_header().unwrap();
    writer.write_image_data(&img_data).unwrap();

    println!("Wrote {output_filename}");
}
