// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! GPU encodings of paints.

#![allow(
    clippy::cast_possible_truncation,
    reason = "GPU paint structures have small, fixed sizes that fit in u32"
)]

use bytemuck::{Pod, Zeroable};
use core::mem::size_of;
use vello_common::encode::{
    EncodedBlurredRoundedRectangle, EncodedGradient, EncodedImage, EncodedKind, FocalKind,
    RadialKind,
};
use vello_common::geometry::RectU16;
use vello_common::image_cache::ImageResource;
use vello_common::paint::{TextureRegion, TintMode};
use vello_common::peniko;

/// Different types of GPU encoded paints.
#[derive(Debug)]
pub(crate) enum GpuEncodedPaint {
    /// An encoded image.
    Image(GpuEncodedImage),
    /// A full external image sampled by the GPU sampler.
    NativeImage(GpuNativeImage),
    /// A full `wgpu::ExternalTexture` sampled by the GPU sampler with pad extend.
    #[cfg(all(feature = "wgpu", feature = "wgpu_external_texture"))]
    WgpuExternalTexture(GpuNativeImage),
    /// An encoded linear gradient.
    LinearGradient(GpuLinearGradient),
    /// An encoded radial gradient.
    RadialGradient(GpuRadialGradient),
    /// An encoded sweep gradient.
    SweepGradient(GpuSweepGradient),
    /// An encoded blurred rounded rectangle.
    BlurredRoundedRect(GpuBlurredRoundedRect),
}

const fn texels<T>() -> u32 {
    let bytes = size_of::<T>();
    assert!(
        bytes.is_multiple_of(16),
        "GPU paints must occupy whole RGBA32Uint texels"
    );
    (bytes / 16) as u32
}

impl GpuEncodedPaint {
    const IMAGE_TEXELS: u32 = texels::<GpuEncodedImage>();
    const NATIVE_IMAGE_TEXELS: u32 = texels::<GpuNativeImage>();
    const LINEAR_TEXELS: u32 = texels::<GpuLinearGradient>();
    const RADIAL_TEXELS: u32 = texels::<GpuRadialGradient>();
    const SWEEP_TEXELS: u32 = texels::<GpuSweepGradient>();
    const BLURRED_RECT_TEXELS: u32 = texels::<GpuBlurredRoundedRect>();

    /// Returns the byte representation of this paint.
    #[inline]
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Image(paint) => bytemuck::bytes_of(paint),
            Self::NativeImage(paint) => bytemuck::bytes_of(paint),
            #[cfg(all(feature = "wgpu", feature = "wgpu_external_texture"))]
            Self::WgpuExternalTexture(paint) => bytemuck::bytes_of(paint),
            Self::LinearGradient(paint) => bytemuck::bytes_of(paint),
            Self::RadialGradient(paint) => bytemuck::bytes_of(paint),
            Self::SweepGradient(paint) => bytemuck::bytes_of(paint),
            Self::BlurredRoundedRect(paint) => bytemuck::bytes_of(paint),
        }
    }

    /// Number of `RGBA32Uint` texels occupied by this paint.
    pub(crate) fn size_texels(&self) -> u32 {
        match self {
            Self::Image(_) => Self::IMAGE_TEXELS,
            Self::NativeImage(_) => Self::NATIVE_IMAGE_TEXELS,
            #[cfg(all(feature = "wgpu", feature = "wgpu_external_texture"))]
            Self::WgpuExternalTexture(_) => Self::NATIVE_IMAGE_TEXELS,
            Self::LinearGradient(_) => Self::LINEAR_TEXELS,
            Self::RadialGradient(_) => Self::RADIAL_TEXELS,
            Self::SweepGradient(_) => Self::SWEEP_TEXELS,
            Self::BlurredRoundedRect(_) => Self::BLURRED_RECT_TEXELS,
        }
    }

    /// Serialize paint enums directly into the provided buffer. Returns the number of bytes written.
    pub(crate) fn serialize_to_buffer(paints: &[Self], buffer: &mut [u8]) {
        let mut offset = 0;
        for paint in paints {
            let paint_bytes = paint.as_bytes();
            let end_offset = offset + paint_bytes.len();
            buffer[offset..end_offset].copy_from_slice(paint_bytes);
            offset = end_offset;
        }
    }
}

/// GPU data for a full external image using native texture sampling.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub(crate) struct GpuNativeImage {
    /// Affine transform [a, b, c, d, tx, ty] from view pixels to normalized UVs.
    pub transform: [f32; 6],
    /// Opacity as an 8-bit unorm value in the low byte.
    pub opacity: u32,
    pub _padding: u32,
}

pub(crate) fn encode_native_image_paint(
    image: &EncodedImage,
    texture_width: u16,
    texture_height: u16,
) -> Option<GpuEncodedPaint> {
    if !crate::paint::supports_native_image(&image.source, image.sampler.quality, image.tint) {
        return None;
    }

    let mut transform = image.transform.as_coeffs();

    for index in [0, 2, 4] {
        transform[index] /= f64::from(texture_width);
    }

    for index in [1, 3, 5] {
        transform[index] /= f64::from(texture_height);
    }

    Some(GpuEncodedPaint::NativeImage(GpuNativeImage {
        transform: transform.map(|x| x as f32),
        opacity: pack_tint(image.tint).0 >> 24,
        _padding: 0,
    }))
}

/// Encode an image paint that samples a `wgpu::ExternalTexture`.
///
/// Returns `None` if the paint can't be drawn from an external texture, which only supports
/// sampling the full texture with pad extend, low or medium quality and opacity as the only tint.
#[cfg(all(feature = "wgpu", feature = "wgpu_external_texture"))]
pub(crate) fn encode_wgpu_external_texture_paint(
    image: &EncodedImage,
    texture_width: u16,
    texture_height: u16,
) -> Option<GpuEncodedPaint> {
    use vello_common::peniko::Extend;

    if image.sampler.x_extend != Extend::Pad || image.sampler.y_extend != Extend::Pad {
        return None;
    }
    let Some(GpuEncodedPaint::NativeImage(paint)) =
        encode_native_image_paint(image, texture_width, texture_height)
    else {
        return None;
    };

    Some(GpuEncodedPaint::WgpuExternalTexture(paint))
}

/// GPU encoded image data.
/// Align to 16 bytes for `RGBA32Uint` alignment.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[allow(dead_code, reason = "Clippy fails when --no-default-features")]
pub(crate) struct GpuEncodedImage {
    /// Packed rendering quality and extend modes.
    /// Bits 4-5: `extend_y` (2 bits)
    /// Bits 2-3: `extend_x` (2 bits)  
    /// Bits 0-1: `quality` (2 bits)
    pub image_params: u32,
    /// Packed image width and height.
    pub image_size: u32,
    /// The offset of the image in the atlas texture in pixels.
    pub image_offset: u32,
    /// Transform matrix [a, b, c, d, tx, ty].
    pub transform: [f32; 6],
    /// Premultiplied tint color packed as RGBA8 unorm (`pack4x8unorm` layout).
    pub tint: u32,
    /// GPU tint mode.
    pub tint_mode: u32,
    /// Number of transparent padding pixels around the image in the atlas.
    pub image_padding: u32,
}

/// GPU encoded blurred rounded rectangle data.
/// Align to 16 bytes for `RGBA32Uint` alignment.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[allow(dead_code, reason = "Clippy fails when --no-default-features")]
pub(crate) struct GpuBlurredRoundedRect {
    /// Transform matrix [a, b, c, d, tx, ty].
    pub transform: [f32; 6],
    /// Premultiplied color packed as RGBA8 unorm (`pack4x8unorm` layout).
    pub color: u32,
    /// Whether to paint the inverse (`1 - alpha`) of the blur coverage
    pub invert: u32,
    /// Blur parameters: exponent, reciprocal exponent, scale, and inverse standard deviation.
    pub params0: [f32; 4],
    /// Blur parameters: minimum edge length, adjusted width, adjusted height, and outer radius.
    pub params1: [f32; 4],
    /// Blur parameters [width, height].
    pub size: [f32; 2],
    /// Padding for 16-byte alignment.
    pub _padding1: [u32; 2],
}

/// GPU encoded linear gradient data.
/// Align to 16 bytes for `RGBA32Uint` alignment.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[allow(dead_code, reason = "Clippy fails when --no-default-features")]
pub(crate) struct GpuLinearGradient {
    /// Packed texture width (bits 0-30) and extend mode (bit 31: 0=Pad, 1=Repeat).
    pub texture_width_and_extend_mode: u32,
    /// Start coordinate in the flat gradient texture.
    pub gradient_start: u32,
    /// Transform matrix [a, b, c, d, tx, ty].
    pub transform: [f32; 6],
}

/// GPU encoded radial gradient data.
/// Align to 16 bytes for `RGBA32Uint` alignment.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[allow(dead_code, reason = "Clippy fails when --no-default-features")]
pub(crate) struct GpuRadialGradient {
    /// Packed texture width (bits 0-30) and extend mode (bit 31: 0=Pad, 1=Repeat).
    pub texture_width_and_extend_mode: u32,
    /// Start coordinate in the flat gradient texture for dense packing.
    pub gradient_start: u32,
    /// Transform matrix [a, b, c, d, tx, ty].
    pub transform: [f32; 6],
    /// Packed kind (bits 0-1, see `RADIAL_KIND_*`) and whether to negate the
    /// square root of `RADIAL_KIND_FOCAL` (bit 2).
    pub kind_and_flags: u32,
    /// `fp0` for focal gradients, `scaled_r0_squared` for strip gradients and 0 otherwise.
    pub param: f32,
    /// Scale applied to the computed gradient position.
    pub scale: f32,
    /// Bias added to the scaled gradient position.
    pub bias: f32,
}

/// GPU encoded sweep gradient data.
/// Align to 16 bytes for `RGBA32Uint` alignment.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[allow(dead_code, reason = "Clippy fails when --no-default-features")]
pub(crate) struct GpuSweepGradient {
    /// Packed texture width (bits 0-30) and extend mode (bit 31: 0=Pad, 1=Repeat).
    pub texture_width_and_extend_mode: u32,
    /// Start coordinate in the flat gradient texture for dense packing.
    pub gradient_start: u32,
    /// Transform matrix [a, b, c, d, tx, ty].
    pub transform: [f32; 6],
    /// Starting angle for sweep gradient.
    pub start_angle: f32,
    /// Inverse of angle delta for sweep gradient.
    pub inv_angle_delta: f32,
    /// Padding for 16-byte alignment.
    pub _padding: [u32; 2],
}

// Constants for packing extend_mode and texture_width.
const EXTEND_MODE_MASK: u32 = 1 << 30;
const TEXTURE_WIDTH_MASK: u32 = !EXTEND_MODE_MASK;

/// Pack `extend_mode` and `texture_width` into a single u32.
/// `extend_mode`: 0=Pad, 1=Repeat, 2=Reflect (stored in bits 30 & 31)
/// `texture_width`: stored in bits 0-29 (max value: 2^30-1)
#[inline(always)]
pub(crate) fn pack_texture_width_and_extend_mode(texture_width: u32, extend_mode: u32) -> u32 {
    debug_assert!(extend_mode <= 2, "extend_mode must be less or equal to 2");
    debug_assert!(
        texture_width <= TEXTURE_WIDTH_MASK,
        "texture_width {texture_width} exceeds maximum value {TEXTURE_WIDTH_MASK}"
    );
    (extend_mode << 30) | (texture_width & TEXTURE_WIDTH_MASK)
}

const RADIAL_KIND_WELL_BEHAVED: u32 = 0;
const RADIAL_KIND_STRIP: u32 = 1;
const RADIAL_KIND_FOCAL_ON_CIRCLE: u32 = 2;
const RADIAL_KIND_FOCAL: u32 = 3;
const RADIAL_NEGATE_ROOT_FLAG: u32 = 1 << 2;

/// Encode a radial gradient.
pub(crate) fn encode_radial_gradient(
    texture_width_and_extend_mode: u32,
    gradient_start: u32,
    transform: [f32; 6],
    radial: &RadialKind,
) -> GpuRadialGradient {
    let (kind_and_flags, param, scale, bias) = match radial {
        RadialKind::Radial { bias, scale } => (RADIAL_KIND_WELL_BEHAVED, 0.0, *scale, *bias),
        RadialKind::Strip { scaled_r0_squared } => {
            (RADIAL_KIND_STRIP, *scaled_r0_squared, 1.0, 0.0)
        }
        RadialKind::Focal {
            kind,
            fp0,
            scale,
            bias,
        } => {
            let kind_and_flags = match kind {
                FocalKind::OnCircle => RADIAL_KIND_FOCAL_ON_CIRCLE,
                FocalKind::WellBehaved => RADIAL_KIND_WELL_BEHAVED,
                FocalKind::General { negate_root: false } => RADIAL_KIND_FOCAL,
                FocalKind::General { negate_root: true } => {
                    RADIAL_KIND_FOCAL | RADIAL_NEGATE_ROOT_FLAG
                }
            };
            (kind_and_flags, *fp0, *scale, *bias)
        }
    };

    GpuRadialGradient {
        texture_width_and_extend_mode,
        gradient_start,
        transform,
        kind_and_flags,
        param,
        scale,
        bias,
    }
}

/// Pack image `width` and `height` into a single u32.
/// `width`: stored in bits 16-31 (upper 16 bits)
/// `height`: stored in bits 0-15 (lower 16 bits)
#[inline(always)]
pub(crate) fn pack_image_size(width: u16, height: u16) -> u32 {
    ((width as u32) << 16) | (height as u32)
}

/// Pack image offset coordinates `x` and `y` into a single u32.
/// `x`: stored in bits 16-31 (upper 16 bits)
/// `y`: stored in bits 0-15 (lower 16 bits)
#[inline(always)]
pub(crate) fn pack_image_offset(x: u16, y: u16) -> u32 {
    ((x as u32) << 16) | (y as u32)
}

/// Pack image `quality` and extend modes into a single u32.
/// `extend_y`: stored in bits 4-5 (2 bits)
/// `extend_x`: stored in bits 2-3 (2 bits)
/// `quality`: stored in bits 0-1 (2 bits)
#[inline(always)]
pub(crate) fn pack_image_params(quality: u32, extend_x: u32, extend_y: u32) -> u32 {
    debug_assert!(extend_x <= 3, "extend_x must be 0-3 (2 bits)");
    debug_assert!(extend_y <= 3, "extend_y must be 0-3 (2 bits)");
    debug_assert!(quality <= 3, "quality must be 0-3 (2 bits)");
    (extend_y << 4) | (extend_x << 2) | quality
}

/// Pack an optional [`Tint`](vello_common::paint::Tint) into a (`tint_color_u32`, `tint_mode_u32`) pair for the GPU.
///
/// The tint color is premultiplied before packing into a u32 in the same layout
/// as WGSL `pack4x8unorm`.
#[inline(always)]
pub(crate) fn pack_tint(tint: Option<vello_common::paint::Tint>) -> (u32, u32) {
    match tint {
        Some(t) => {
            let color = t.color.premultiply().to_rgba8().to_u32();
            (color, t.mode.as_u32())
        }
        // With no tint, use `u32::MAX` (which corresponds to 1.0 on all lanes).
        // Since we use `Multiply` this will essentially just yield the original image
        // sample, having the same effect as no tinting at all.
        None => (u32::MAX, TintMode::Multiply.as_u32()),
    }
}

pub(crate) fn encode_image_paint(
    image: &EncodedImage,
    image_resource: &ImageResource,
) -> GpuEncodedPaint {
    let transform = image.transform.as_coeffs().map(|x| x as f32);
    let image_size = pack_image_size(image_resource.width, image_resource.height);
    let image_offset = pack_image_offset(image_resource.offset[0], image_resource.offset[1]);
    let image_params = pack_image_params(
        image.sampler.quality as u32,
        image.sampler.x_extend as u32,
        image.sampler.y_extend as u32,
    );
    let (tint, tint_mode) = pack_tint(image.tint);

    GpuEncodedPaint::Image(GpuEncodedImage {
        image_params,
        image_size,
        image_offset,
        transform,
        tint,
        tint_mode,
        image_padding: image_resource.padding as u32,
    })
}

pub(crate) fn encode_external_texture_paint(
    image: &EncodedImage,
    source_region: TextureRegion,
    texture_width: u16,
    texture_height: u16,
) -> GpuEncodedPaint {
    if let Some(native_paint) = encode_native_image_paint(image, texture_width, texture_height) {
        return native_paint;
    }

    let region = match source_region {
        TextureRegion::Full => RectU16::new(0, 0, texture_width, texture_height),
        TextureRegion::Rect(rect) => rect,
    };
    let transform = image.transform.as_coeffs().map(|x| x as f32);
    let image_size = pack_image_size(region.width(), region.height());
    let image_offset = pack_image_offset(region.x0, region.y0);
    let image_params = pack_image_params(
        image.sampler.quality as u32,
        image.sampler.x_extend as u32,
        image.sampler.y_extend as u32,
    );
    let (tint, tint_mode) = pack_tint(image.tint);

    GpuEncodedPaint::Image(GpuEncodedImage {
        image_params,
        image_size,
        image_offset,
        transform,
        tint,
        tint_mode,
        image_padding: 0,
    })
}

pub(crate) fn encode_gradient_paint(
    gradient: &EncodedGradient,
    gradient_width: u32,
    gradient_start: u32,
) -> GpuEncodedPaint {
    let transform = gradient.transform.as_coeffs().map(|x| x as f32);
    let extend_mode = match gradient.extend {
        peniko::Extend::Pad => 0,
        peniko::Extend::Repeat => 1,
        peniko::Extend::Reflect => 2,
    };
    let texture_width_and_extend_mode =
        pack_texture_width_and_extend_mode(gradient_width, extend_mode);

    match &gradient.kind {
        EncodedKind::Linear(_) => GpuEncodedPaint::LinearGradient(GpuLinearGradient {
            texture_width_and_extend_mode,
            gradient_start,
            transform,
        }),
        EncodedKind::Radial(radial) => GpuEncodedPaint::RadialGradient(encode_radial_gradient(
            texture_width_and_extend_mode,
            gradient_start,
            transform,
            radial,
        )),
        EncodedKind::Sweep(sweep) => GpuEncodedPaint::SweepGradient(GpuSweepGradient {
            texture_width_and_extend_mode,
            gradient_start,
            transform,
            start_angle: sweep.start_angle,
            inv_angle_delta: sweep.inv_angle_delta,
            _padding: [0, 0],
        }),
    }
}

pub(crate) fn encode_blurred_rounded_rect_paint(
    rect: &EncodedBlurredRoundedRectangle,
) -> GpuEncodedPaint {
    GpuEncodedPaint::BlurredRoundedRect(GpuBlurredRoundedRect {
        transform: rect.transform.as_coeffs().map(|x| x as f32),
        color: rect.color.as_premul_rgba8().to_u32(),
        invert: u32::from(rect.invert),
        params0: [
            rect.exponent,
            rect.recip_exponent,
            rect.scale,
            rect.std_dev_inv,
        ],
        params1: [rect.min_edge, rect.w, rect.h, rect.r1],
        size: [rect.width, rect.height],
        _padding1: [0, 0],
    })
}

#[cfg(test)]
mod tests {
    use super::encode_native_image_paint;
    use vello_common::TextureId;
    use vello_common::encode::EncodedImage;
    use vello_common::kurbo::{Affine, Vec2};
    use vello_common::paint::{ImageSource, TextureRegion, Tint, TintMode};
    use vello_common::peniko::{Color, ImageQuality, ImageSampler};

    #[test]
    fn native_image_encoding_accepts_opacity_but_not_color_tint_or_bicubic() {
        let mut image = EncodedImage {
            source: ImageSource::external_texture(TextureId(1), TextureRegion::Full, true),
            sampler: ImageSampler::default(),
            may_have_transparency: true,
            transform: Affine::IDENTITY,
            x_advance: Vec2::new(1.0, 0.0),
            y_advance: Vec2::new(0.0, 1.0),
            tint: Some(Tint {
                color: Color::new([1.0, 1.0, 1.0, 0.5]),
                mode: TintMode::Multiply,
            }),
        };
        assert!(encode_native_image_paint(&image, 2, 2).is_some());

        image.tint = Some(Tint {
            color: Color::new([0.5, 1.0, 1.0, 0.5]),
            mode: TintMode::Multiply,
        });
        assert!(encode_native_image_paint(&image, 2, 2).is_none());

        image.tint = Some(Tint {
            color: Color::new([1.0, 1.0, 1.0, 0.5]),
            mode: TintMode::AlphaMask,
        });
        assert!(encode_native_image_paint(&image, 2, 2).is_none());

        image.tint = None;
        image.sampler.quality = ImageQuality::High;
        assert!(encode_native_image_paint(&image, 2, 2).is_none());
    }

    #[cfg(all(feature = "wgpu", feature = "wgpu_external_texture"))]
    #[test]
    fn wgpu_external_texture_encoding_requires_full_pad_sampling() {
        use super::{GpuEncodedPaint, encode_wgpu_external_texture_paint};
        use vello_common::geometry::RectU16;
        use vello_common::peniko::Extend;

        let encode = |modify: fn(&mut EncodedImage)| {
            let mut image = EncodedImage {
                source: ImageSource::external_texture(TextureId(1), TextureRegion::Full, false),
                sampler: ImageSampler {
                    quality: ImageQuality::Medium,
                    ..ImageSampler::default()
                },
                may_have_transparency: false,
                transform: Affine::IDENTITY,
                x_advance: Vec2::new(1.0, 0.0),
                y_advance: Vec2::new(0.0, 1.0),
                tint: None,
            };
            modify(&mut image);
            encode_wgpu_external_texture_paint(&image, 2, 2)
        };

        assert!(matches!(
            encode(|_| {}),
            Some(GpuEncodedPaint::WgpuExternalTexture(_))
        ));
        assert!(encode(|image| image.sampler.x_extend = Extend::Repeat).is_none());
        assert!(encode(|image| image.sampler.y_extend = Extend::Reflect).is_none());
        assert!(encode(|image| image.sampler.quality = ImageQuality::High).is_none());
        assert!(
            encode(|image| {
                image.source = ImageSource::external_texture(
                    TextureId(1),
                    TextureRegion::Rect(RectU16::new(0, 0, 1, 1)),
                    false,
                );
            })
            .is_none()
        );
    }
}
