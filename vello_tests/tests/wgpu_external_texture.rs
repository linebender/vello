// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for image paints backed by a native `wgpu::ExternalTexture`.
//!
//! These bypass the CPU/GPU snapshot harness because only wgpu can bind external textures; each
//! test returns early when the adapter does not support `wgpu::Features::EXTERNAL_TEXTURE`.

use vello_common::kurbo::Rect;
use vello_common::paint::{Image, ImageSource, TextureRegion};
use vello_common::peniko::{Extend, ImageQuality, ImageSampler};
use vello_cpu::{Level, RenderMode};

use crate::renderer::{GpuRenderer, Renderer};

/// Distinct luma per pixel of a 4×2 NV12 texture.
const NV12_Y_PLANE: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
/// One interleaved Cb/Cr pair per 2×2 luma block.
const NV12_UV_PLANE: [u8; 4] = [100, 110, 200, 210];

/// With an identity conversion matrix, each pixel is its raw `[Y, Cb, Cr]` sample, which checks
/// that both planes are bound and the chroma plane is subsampled.
///
/// Nearest sampling is used, since linear sampling blends neighboring chroma samples.
#[test]
fn wgpu_external_texture_nv12_planes() {
    #[rustfmt::skip]
    let identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let Some(pixels) = render_nv12(identity, ImageQuality::Low) else {
        return;
    };
    assert_pixels_near(&pixels, |i| {
        let uv = (i % 4) / 2 * 2;
        [
            NV12_Y_PLANE[i],
            NV12_UV_PLANE[uv],
            NV12_UV_PLANE[uv + 1],
            255,
        ]
    });
}

/// wgpu does not clamp the converted color, so Vello clamps it to the premultiplied range.
#[test]
fn wgpu_external_texture_nv12_out_of_range_is_clamped() {
    // R = 4Y overflows for Y > 63, G = -Y underflows, B = Y stays in range.
    #[rustfmt::skip]
    let matrix = [
        4.0, -1.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    for quality in [ImageQuality::Low, ImageQuality::Medium] {
        let Some(pixels) = render_nv12(matrix, quality) else {
            return;
        };
        assert_pixels_near(&pixels, |i| {
            let y = NV12_Y_PLANE[i];
            [y.saturating_mul(4), 0, y, 255]
        });
    }
}

/// Renders [`NV12_Y_PLANE`] and [`NV12_UV_PLANE`] 1:1 as a `wgpu::ExternalTexture`, or returns
/// `None` if the device does not support external textures.
fn render_nv12(yuv_conversion_matrix: [f32; 16], quality: ImageQuality) -> Option<Vec<[u8; 4]>> {
    let mut ctx = GpuRenderer::new(4, 2, 0, Level::baseline(), RenderMode::OptimizeSpeed);
    if !ctx
        .device()
        .features()
        .contains(wgpu::Features::EXTERNAL_TEXTURE)
    {
        return None;
    }
    let external_texture = create_nv12_external_texture(
        ctx.device(),
        ctx.queue(),
        4,
        2,
        &NV12_Y_PLANE,
        &NV12_UV_PLANE,
        yuv_conversion_matrix,
    );
    let texture_id = ctx.register_wgpu_external_texture(external_texture, 4, 2);
    ctx.set_paint(Image {
        image: ImageSource::external_texture(texture_id, TextureRegion::Full, false),
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality,
            alpha: 1.0,
        },
    });
    ctx.fill_rect(&Rect::new(0., 0., 4., 2.));
    ctx.render();
    let pixmap = ctx.snapshot();
    Some(
        pixmap
            .data_as_u8_slice()
            .chunks_exact(4)
            .map(|pixel| pixel.try_into().unwrap())
            .collect(),
    )
}

/// Creates a two-plane NV12 external texture: a full-resolution `R8Unorm` Y plane and a
/// half-resolution `Rg8Unorm` interleaved Cb/Cr plane, converted with `yuv_conversion_matrix`.
fn create_nv12_external_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    y_plane: &[u8],
    uv_plane: &[u8],
    yuv_conversion_matrix: [f32; 16],
) -> wgpu::ExternalTexture {
    let y_view = create_plane_view(
        device,
        queue,
        wgpu::TextureFormat::R8Unorm,
        width,
        height,
        y_plane,
        width,
    );
    let uv_view = create_plane_view(
        device,
        queue,
        wgpu::TextureFormat::Rg8Unorm,
        width / 2,
        height / 2,
        uv_plane,
        width,
    );
    #[rustfmt::skip]
    let identity_gamut = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    device.create_external_texture(
        &wgpu::ExternalTextureDescriptor {
            label: Some("Test NV12 External Texture"),
            width,
            height,
            format: wgpu::ExternalTextureFormat::Nv12,
            yuv_conversion_matrix,
            gamut_conversion_matrix: identity_gamut,
            src_transfer_function: wgpu::ExternalTextureTransferFunction::default(),
            dst_transfer_function: wgpu::ExternalTextureTransferFunction::default(),
            sample_transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            load_transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        },
        &[&y_view, &uv_view],
    )
}

fn create_plane_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    data: &[u8],
    bytes_per_row: u32,
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Test External Texture Plane"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: None,
        },
        size,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn assert_pixels_near(actual: &[[u8; 4]], expected: impl Fn(usize) -> [u8; 4]) {
    for (i, &pixel) in actual.iter().enumerate() {
        let expected = expected(i);
        assert!(
            pixel.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 1),
            "pixel ({}, {}): got {pixel:?}, expected {expected:?}",
            i % 4,
            i / 4,
        );
    }
}
