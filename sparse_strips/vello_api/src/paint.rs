// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types for paints.

use crate::kurbo::{Affine, Point};
use crate::pixmap::Pixmap;
use alloc::sync::Arc;
use peniko::color::{AlphaColor, ColorSpaceTag, HueDirection, PremulRgba8, Srgb};
use peniko::{ColorStops, GradientKind, ImageQuality};

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
        // TODO: This might be slow on x86, see https://github.com/linebender/color/issues/142.
        // Since we only do that conversion once per path it might not be critical, but should
        // still be measured. This also applies to all other usages of `to_rgba8` in the current
        // code.
        Self::Solid(PremulColor::new(value))
    }
}

// TODO: Replace this with the peniko type, once it supports transforms.
/// A gradient.
#[derive(Debug, Clone)]
pub struct Gradient {
    /// The underlying kind of gradient.
    pub kind: GradientKind,
    /// The stops that makes up the gradient.
    ///
    /// Note that the first stop must have an offset of 0.0 and the last stop
    /// must have an offset of 1.0. In addition to that, the stops must be sorted
    /// with offsets in ascending order.
    pub stops: ColorStops,
    /// A transformation to apply to the gradient.
    pub transform: Affine,
    /// The extend of the gradient.
    pub extend: peniko::Extend,
    /// The color space to be used for interpolation.
    ///
    /// The colors in the color stops will be converted to this color space.
    ///
    /// This defaults to [sRGB](ColorSpaceTag::Srgb).
    pub interpolation_cs: ColorSpaceTag,
    /// When interpolating within a cylindrical color space, the direction for the hue.
    ///
    /// This is interpreted as described in [CSS Color Module Level 4 § 12.4].
    ///
    /// [CSS Color Module Level 4 § 12.4]: https://drafts.csswg.org/css-color/#hue-interpolation
    pub hue_direction: HueDirection,
}

impl Default for Gradient {
    fn default() -> Self {
        Self {
            kind: GradientKind::Linear {
                start: Point::default(),
                end: Point::default(),
            },
            transform: Affine::IDENTITY,
            interpolation_cs: ColorSpaceTag::Srgb,
            extend: Default::default(),
            hue_direction: Default::default(),
            stops: Default::default(),
        }
    }
}

impl Gradient {
    /// Returns the gradient with the alpha component for all color stops
    /// multiplied by `alpha`.
    #[must_use]
    pub fn multiply_alpha(mut self, alpha: f32) -> Self {
        self.stops
            .iter_mut()
            .for_each(|stop| *stop = stop.multiply_alpha(alpha));
        self
    }
}

/// An image.
#[derive(Debug, Clone)]
pub struct Image {
    /// The underlying pixmap of the image.
    pub pixmap: Arc<Pixmap>,
    /// Extend mode in the horizontal direction.
    pub x_extend: peniko::Extend,
    /// Extend mode in the vertical direction.
    pub y_extend: peniko::Extend,
    /// Hint for desired rendering quality.
    pub quality: ImageQuality,
    /// A transform to apply to the image.
    pub transform: Affine,
}

/// A premultiplied color.
#[derive(Debug, Clone, PartialEq, Copy)]
pub struct PremulColor {
    premul_u8: PremulRgba8,
    premul_f32: peniko::color::PremulColor<Srgb>,
}

impl PremulColor {
    /// Create a new premultiplied color.
    pub fn new(color: AlphaColor<Srgb>) -> Self {
        let premul = color.premultiply();

        Self {
            premul_u8: premul.to_rgba8(),
            premul_f32: premul,
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
