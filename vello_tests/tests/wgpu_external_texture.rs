// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests for image paints backed by a native `wgpu::ExternalTexture`.
//!
//! These bypass the CPU/GPU snapshot harness because only wgpu can bind external textures, so
//! they only produce a GPU reference image. Each test returns early when the adapter does not
//! support `wgpu::Features::EXTERNAL_TEXTURE`.

use vello_common::color::palette::css::BLACK;
use vello_common::kurbo::{Affine, Rect};
use vello_common::paint::{Image, ImageSource, TextureRegion};
use vello_common::peniko::{Extend, ImageQuality, ImageSampler};
use vello_cpu::{Level, RenderMode};

use crate::renderer::{GpuRenderer, Renderer};
use crate::util::check_ref;

/// Magnification of the 4×2 test textures, so each texel covers a visible block.
const SCALE: u16 = 16;

/// High-contrast luma checkerboard, so filtering is easy to see.
const CHECKER_Y_PLANE: [u8; 8] = [0, 255, 0, 255, 255, 0, 255, 0];
/// Left 2×2 block is blue/magenta, right block is green/yellow under [`IDENTITY_MATRIX`].
const CHECKER_UV_PLANE: [u8; 4] = [0, 255, 255, 0];

/// Luma ramp, so the out-of-range test covers both clamped and in-range values.
const RAMP_Y_PLANE: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
/// One interleaved Cb/Cr pair per 2×2 luma block.
const RAMP_UV_PLANE: [u8; 4] = [100, 110, 200, 210];

/// Maps each pixel to its raw `[Y, Cb, Cr]` sample.
#[rustfmt::skip]
const IDENTITY_MATRIX: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

/// Each block shows its raw `[Y, Cb, Cr]` sample, so the colors show that both planes are bound
/// and each Cb/Cr pair covers a 2×2 luma block. Nearest sampling keeps the blocks hard-edged.
#[test]
fn wgpu_external_texture_nv12_nearest() {
    check_nv12(
        "wgpu_external_texture_nv12_nearest",
        &CHECKER_Y_PLANE,
        &CHECKER_UV_PLANE,
        IDENTITY_MATRIX,
        ImageQuality::Low,
        1.0,
    );
}

/// Like [`wgpu_external_texture_nv12_nearest`], but bilinear sampling blends neighboring luma
/// and chroma samples into smooth gradients.
#[test]
fn wgpu_external_texture_nv12_bilinear() {
    check_nv12(
        "wgpu_external_texture_nv12_bilinear",
        &CHECKER_Y_PLANE,
        &CHECKER_UV_PLANE,
        IDENTITY_MATRIX,
        ImageQuality::Medium,
        1.0,
    );
}

/// wgpu does not clamp the converted color, so Vello clamps it to the premultiplied range.
///
/// The texture is drawn at half opacity over black, where an unclamped red channel brighter than
/// its alpha shows up as a brighter result. The render target would clamp an opaque out-of-range
/// color anyway, and negative channels are clamped before blending, so only the overflow is
/// observable.
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
    check_nv12(
        "wgpu_external_texture_nv12_out_of_range_is_clamped",
        &RAMP_Y_PLANE,
        &RAMP_UV_PLANE,
        matrix,
        ImageQuality::Low,
        0.5,
    );
}

/// Draws a 4×2 NV12 texture magnified by [`SCALE`] over black and compares the result with the
/// `name` reference snapshot.
fn check_nv12(
    name: &str,
    y_plane: &[u8; 8],
    uv_plane: &[u8; 4],
    yuv_conversion_matrix: [f32; 16],
    quality: ImageQuality,
    alpha: f32,
) {
    let (width, height) = (4 * SCALE, 2 * SCALE);
    let mut ctx = GpuRenderer::new(
        width,
        height,
        0,
        Level::baseline(),
        RenderMode::OptimizeSpeed,
    );
    if !ctx
        .device()
        .features()
        .contains(wgpu::Features::EXTERNAL_TEXTURE)
    {
        return;
    }
    let external_texture = create_nv12_external_texture(
        ctx.device(),
        ctx.queue(),
        4,
        2,
        y_plane,
        uv_plane,
        yuv_conversion_matrix,
    );
    let texture_id = ctx.register_wgpu_external_texture(external_texture, 4, 2);
    let canvas = Rect::new(0., 0., f64::from(width), f64::from(height));
    ctx.set_paint(BLACK);
    ctx.fill_rect(&canvas);
    ctx.set_paint(Image {
        image: ImageSource::external_texture(texture_id, TextureRegion::Full, false),
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality,
            alpha,
        },
    });
    ctx.set_paint_transform(Affine::scale(f64::from(SCALE)));
    ctx.fill_rect(&canvas);
    check_ref(&mut ctx, name, name, 1, 0, true, &[]);
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
