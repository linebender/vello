// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests to ensure that certain issues which don't deserve a test scene don't regress

use std::sync::Arc;

use scenes::ImageCache;
use scenes::SimpleText;
use vello::{
    AaConfig, RendererOptions, Scene,
    kurbo::{Affine, Rect, RoundedRect, Stroke},
    peniko::{
        Blob, Brush, Color, ColorStop, Extend, Gradient, ImageAlphaType, ImageBrush, ImageData,
        ImageFormat, ImageQuality, InterpolationAlphaSpace, color::palette,
    },
    util::{RenderContext, block_on_wgpu},
    wgpu::{
        self, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d,
        TexelCopyBufferInfo, TextureDescriptor, TextureFormat, TextureUsages,
    },
};
use vello_tests::{TestParams, smoke_snapshot_test_sync, snapshot_test_sync};

/// Test created from <https://github.com/linebender/vello/issues/616>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn rounded_rectangle_watertight() {
    let mut scene = Scene::new();
    let rect = RoundedRect::new(60.0, 10.0, 80.0, 30.0, 10.0);
    let stroke = Stroke::new(2.0);
    scene.stroke(&stroke, Affine::IDENTITY, palette::css::WHITE, None, &rect);
    let mut params = TestParams::new("rounded_rectangle_watertight", 70, 30);
    params.anti_aliasing = AaConfig::Msaa16;
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

const DATA_IMAGE_PNG: &[u8] = include_bytes!("../snapshots/smoke/data_image_roundtrip.png");

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_data_image_roundtrip_extend_pad() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Pad);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new(
        "data_image_roundtrip",
        image.image.width,
        image.image.height,
    );
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_data_image_roundtrip_extend_reflect() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Reflect);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new(
        "data_image_roundtrip",
        image.image.width,
        image.image.height,
    );
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Test for <https://github.com/linebender/vello/issues/972>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_data_image_roundtrip_extend_repeat() {
    let mut scene = Scene::new();
    let mut images = ImageCache::new();
    let image = images
        .from_bytes(0, DATA_IMAGE_PNG)
        .unwrap()
        .with_quality(ImageQuality::Low)
        .with_extend(Extend::Repeat);
    scene.draw_image(&image, Affine::IDENTITY);
    let mut params = TestParams::new(
        "data_image_roundtrip",
        image.image.width,
        image.image.height,
    );
    params.anti_aliasing = AaConfig::Area;
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// Test created from <https://github.com/linebender/vello/issues/662>
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn stroke_width_zero() {
    let mut scene = Scene::new();
    let stroke = Stroke::new(0.0);
    let rect = Rect::new(10.0, 10.0, 40.0, 40.0);
    let rect_stroke_color = palette::css::PEACH_PUFF;
    scene.stroke(&stroke, Affine::IDENTITY, rect_stroke_color, None, &rect);
    let mut params = TestParams::new("stroke_width_zero", 50, 50);
    params.anti_aliasing = AaConfig::Msaa16;
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
#[expect(clippy::cast_possible_truncation, reason = "Test code")]
fn text_stroke_width_zero() {
    let font_size = 12.;
    let mut scene = Scene::new();
    let mut simple_text = SimpleText::new();
    simple_text.add_run(
        &mut scene,
        None,
        font_size,
        palette::css::WHITE,
        Affine::translate((0., f64::from(font_size))),
        None,
        None,
        &Stroke::new(0.),
        "Testing text",
    );
    let params = TestParams::new(
        "text_stroke_width_zero",
        (font_size * 6.) as _,
        (font_size * 1.25).ceil() as _,
    );
    snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

const GLYPH_IMAGE_BACKGROUND: [u8; 4] = [247, 243, 236, 255];

fn glyph_image_background() -> Color {
    Color::from_rgba8(
        GLYPH_IMAGE_BACKGROUND[0],
        GLYPH_IMAGE_BACKGROUND[1],
        GLYPH_IMAGE_BACKGROUND[2],
        GLYPH_IMAGE_BACKGROUND[3],
    )
}

fn glyph_image_data() -> ImageData {
    let width = 96_u32;
    let height = 96_u32;
    let mut bytes = Vec::with_capacity(usize::try_from(width * height * 4).unwrap());
    for y in 0..height {
        for x in 0..width {
            let r = 32 + u8::try_from((223 * x) / (width - 1)).unwrap();
            let g = 36 + u8::try_from((170 * y) / (height - 1)).unwrap();
            let stripe = if ((x / 8) + (y / 8)) % 2 == 0 { 34 } else { 0 };
            let b = 82 + stripe;
            bytes.extend_from_slice(&[r, g, b, 255]);
        }
    }
    ImageData {
        data: Blob::new(Arc::new(bytes)),
        format: ImageFormat::Rgba8,
        width,
        height,
        alpha_type: ImageAlphaType::Alpha,
    }
}

fn glyph_image_brush(image: &ImageData) -> Brush {
    Brush::Image(
        ImageBrush::new(image.clone())
            .with_quality(ImageQuality::Medium)
            .with_extend(Extend::Repeat),
    )
}

fn glyph_image_brush_scene(image: &ImageData) -> Scene {
    let mut scene = Scene::new();
    let mut text = SimpleText::new();
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        glyph_image_background(),
        None,
        &Rect::new(0.0, 0.0, 256.0, 256.0),
    );
    let brush = glyph_image_brush(image);
    text.add_var_run(
        &mut scene,
        None,
        42.0,
        &[],
        &brush,
        Affine::translate((14.0, 116.0)),
        None,
        None,
        vello::peniko::Fill::NonZero,
        "texture",
        true,
    );
    scene
}

fn image_free_scene() -> Scene {
    let mut scene = Scene::new();
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(30, 30, 34),
        None,
        &Rect::new(0.0, 0.0, 256.0, 256.0),
    );
    scene
}

#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn glyph_image_brush_survives_image_free_render() {
    pollster::block_on(async {
        let width = 256;
        let height = 256;
        let image = glyph_image_data();
        let mut context = RenderContext::new();
        let device_id = context.device(None).await.expect("compatible device");
        let device_handle = &mut context.devices[device_id];
        let device = &device_handle.device;
        let queue = &device_handle.queue;
        let mut renderer = vello::Renderer::new(
            device,
            RendererOptions {
                num_init_threads: std::num::NonZeroUsize::new(1),
                antialiasing_support: std::iter::once(AaConfig::Area).collect(),
                ..RendererOptions::default()
            },
        )
        .expect("create renderer");
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let target = device.create_texture(&TextureDescriptor {
            label: Some("glyph image brush target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let params = vello::RenderParams {
            base_color: Color::from_rgba8(0, 0, 0, 0),
            width,
            height,
            antialiasing_method: AaConfig::Area,
        };

        for scene in [
            glyph_image_brush_scene(&image),
            image_free_scene(),
            glyph_image_brush_scene(&image),
        ] {
            renderer
                .render_to_texture(device, queue, &scene, &view, &params)
                .expect("render scene");
        }

        let padded_byte_width = (width * 4).next_multiple_of(256);
        let buffer_size = u64::from(padded_byte_width) * u64::from(height);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("glyph image brush readback"),
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("glyph image brush copy"),
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
        block_on_wgpu(device, receiver.receive())
            .expect("map callback")
            .expect("map readback");
        let data = buf_slice.get_mapped_range();
        let unpadded_byte_width = usize::try_from(width * 4).unwrap();
        let padded_byte_width = usize::try_from(padded_byte_width).unwrap();
        let non_background_pixels = data
            .chunks(padded_byte_width)
            .flat_map(|row| row[..unpadded_byte_width].chunks_exact(4))
            .filter(|pixel| **pixel != GLYPH_IMAGE_BACKGROUND)
            .count();
        assert!(
            non_background_pixels > 0,
            "image brush glyph run rendered as a blank image after an image-free render"
        );
    });
}

/// <https://github.com/web-platform-tests/wpt/blob/18c64a74b1/html/canvas/element/fill-and-stroke-styles/2d.gradient.interpolate.coloralpha.html>
/// See <https://github.com/linebender/vello/issues/1056>.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_gradient_color_alpha_premultiplied() {
    let mut scene = Scene::new();
    let viewport = Rect::new(0., 0., 100., 50.);
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Gradient::new_linear((0., 0.), (100., 0.))
            .with_stops([
                ColorStop {
                    offset: 0.,
                    color: Color::from_rgba8(255, 255, 0, 0).into(),
                },
                ColorStop {
                    offset: 1.,
                    color: Color::from_rgba8(0, 0, 255, 255).into(),
                },
            ])
            .with_interpolation_alpha_space(InterpolationAlphaSpace::Premultiplied),
        None,
        &viewport,
    );
    let mut params = TestParams::new("gradient_color_alpha_premultiplied", 100, 50);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}

/// <https://github.com/web-platform-tests/wpt/blob/18c64a74b1/html/canvas/element/fill-and-stroke-styles/2d.gradient.interpolate.coloralpha.html>
/// See <https://github.com/linebender/vello/issues/1056>.
#[test]
#[cfg_attr(skip_gpu_tests, ignore)]
fn test_gradient_color_alpha_unpremultiplied() {
    let mut scene = Scene::new();
    let viewport = Rect::new(0., 0., 100., 50.);
    scene.fill(
        vello::peniko::Fill::NonZero,
        Affine::IDENTITY,
        &Gradient::new_linear((0., 0.), (100., 0.))
            .with_stops([
                ColorStop {
                    offset: 0.,
                    color: Color::from_rgba8(255, 255, 0, 0).into(),
                },
                ColorStop {
                    offset: 1.,
                    color: Color::from_rgba8(0, 0, 255, 255).into(),
                },
            ])
            .with_interpolation_alpha_space(InterpolationAlphaSpace::Unpremultiplied),
        None,
        &viewport,
    );
    let mut params = TestParams::new("gradient_color_alpha_unpremultiplied", 100, 50);
    params.base_color = Some(palette::css::WHITE);
    smoke_snapshot_test_sync(scene, &params)
        .unwrap()
        .assert_mean_less_than(0.001);
}
