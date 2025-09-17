// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types for paints.

use crate::pixmap::Pixmap;
use alloc::sync::Arc;
use peniko::{
    Gradient, ImageQuality, ImageSampler,
    color::{AlphaColor, PremulRgba8, Srgb},
};

/// A paint that needs to be resolved via its index.
// In the future, we might add additional flags, that's why we have
// this thin wrapper around u32, so we can change the underlying
// representation without breaking the API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedPaint(u32);

impl IndexedPaint {
    /// Create a new indexed paint from an index.
    pub fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("exceeded the maximum number of paints"))
    }

    /// Return the index of the paint.
    pub fn index(&self) -> usize {
        usize::try_from(self.0).unwrap()
    }
}

/// A paint that is used internally by a rendering frontend to store how a wide tile command
/// should be painted. There are only two types of paint:
///
/// 1) Simple solid colors, which are stored in premultiplied representation so that
///    each wide tile doesn't have to recompute it.
/// 2) Indexed paints, which can represent any arbitrary, more complex paint that is
///    determined by the frontend. The intended way of using this is to store a vector
///    of paints and store its index inside `IndexedPaint`.
#[derive(Debug, Clone, PartialEq)]
pub enum Paint {
    /// A premultiplied RGBA8 color.
    Solid(PremulColor),
    /// A paint that needs to be resolved via an index.
    Indexed(IndexedPaint),
}

impl From<AlphaColor<Srgb>> for Paint {
    fn from(value: AlphaColor<Srgb>) -> Self {
        Self::Solid(PremulColor::from_alpha_color(value))
    }
}

/// Opaque image handle
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct ImageId(u32);

impl ImageId {
    // TODO: make this private in future
    /// Create a new image id from a u32.
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the image id as a u32.
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

/// Bitmap source used by `Image`.
#[derive(Debug, Clone)]
pub enum ImageSource {
    /// Pixmap pixels travel with the scene packet.
    Pixmap(Arc<Pixmap>),
    /// Pixmap pixels were registered earlier; this is just a handle.
    OpaqueId(ImageId),
}

/// An image.
#[derive(Debug, Clone)]
pub struct Image {
    /// The underlying pixmap of the image.
    pub source: ImageSource,
    /// Extend mode in the horizontal direction.
    pub x_extend: peniko::Extend,
    /// Extend mode in the vertical direction.
    pub y_extend: peniko::Extend,
    /// Hint for desired rendering quality.
    pub quality: ImageQuality,
}

impl Image {
    /// Convert a [`peniko::ImageBrush`] to an [`Image`].
    ///
    /// This is a somewhat lossy conversion, as the image data data is transformed to
    /// [premultiplied RGBA8](`PremulRgba8`).
    ///
    /// # Panics
    ///
    /// This panics if `image` has a `width` or `height` greater than `u16::MAX`.
    pub fn from_peniko_image(brush: &peniko::ImageBrush) -> Self {
        // TODO: how do we deal with `peniko::ImageFormat` growing? See also
        // <https://github.com/linebender/vello/pull/996#discussion_r2080510863>.
        if brush.image.format != peniko::ImageFormat::Rgba8 {
            unimplemented!("Unsupported image format: {:?}", brush.image.format);
        }
        if brush.image.alpha_type != peniko::ImageAlphaType::Alpha {
            unimplemented!("Unsupported image alpha type: {:?}", brush.image.alpha_type);
        }

        assert!(
            brush.image.width <= u16::MAX as u32 && brush.image.height <= u16::MAX as u32,
            "The image is too big. Its width and height can be no larger than {} pixels.",
            u16::MAX,
        );
        let width = brush.image.width.try_into().unwrap();
        let height = brush.image.height.try_into().unwrap();
        let ImageSampler {
            x_extend,
            y_extend,
            quality,
            alpha: global_alpha,
        } = brush.sampler;

        #[expect(clippy::cast_possible_truncation, reason = "deliberate quantization")]
        let global_alpha = u16::from((global_alpha * 255. + 0.5) as u8);

        // TODO: SIMD
        #[expect(clippy::cast_possible_truncation, reason = "This cannot overflow.")]
        let pixels = brush
            .image
            .data
            .data()
            .chunks_exact(4)
            .map(|rgba| {
                let alpha = ((u16::from(rgba[3]) * global_alpha) / 255) as u8;
                let multiply = |component| ((u16::from(alpha) * u16::from(component)) / 255) as u8;
                PremulRgba8 {
                    r: multiply(rgba[0]),
                    g: multiply(rgba[1]),
                    b: multiply(rgba[2]),
                    a: alpha,
                }
            })
            .collect();
        let pixmap = Pixmap::from_parts(pixels, width, height);

        Self {
            source: ImageSource::Pixmap(Arc::new(pixmap)),
            x_extend,
            y_extend,
            quality,
        }
    }
}

/// A premultiplied color.
#[derive(Debug, Clone, PartialEq, Copy)]
pub struct PremulColor {
    premul_u8: PremulRgba8,
    premul_f32: peniko::color::PremulColor<Srgb>,
}

impl PremulColor {
    /// Create a new premultiplied color.
    pub fn from_alpha_color(color: AlphaColor<Srgb>) -> Self {
        Self::from_premul_color(color.premultiply())
    }

    /// Create a new premultiplied color from `peniko::PremulColor`.
    pub fn from_premul_color(color: peniko::color::PremulColor<Srgb>) -> Self {
        Self {
            premul_u8: color.to_rgba8(),
            premul_f32: color,
        }
    }

    /// Return the color as a premultiplied RGBA8 color.
    pub fn as_premul_rgba8(&self) -> PremulRgba8 {
        self.premul_u8
    }

    /// Return the color as a premultiplied RGBAF32 color.
    pub fn as_premul_f32(&self) -> peniko::color::PremulColor<Srgb> {
        self.premul_f32
    }

    /// Return whether the color is opaque (i.e. doesn't have transparency).
    pub fn is_opaque(&self) -> bool {
        self.premul_f32.components[3] == 1.0
    }
}

/// A kind of paint that can be used for filling and stroking shapes.
#[derive(Debug, Clone)]
pub enum PaintType {
    /// A solid color.
    Solid(AlphaColor<Srgb>),
    /// A gradient.
    Gradient(Gradient),
    /// An image.
    Image(Image),
}

impl From<AlphaColor<Srgb>> for PaintType {
    fn from(value: AlphaColor<Srgb>) -> Self {
        Self::Solid(value)
    }
}

impl From<Gradient> for PaintType {
    fn from(value: Gradient) -> Self {
        Self::Gradient(value)
    }
}

impl From<Image> for PaintType {
    fn from(value: Image) -> Self {
        Self::Image(value)
    }
}
