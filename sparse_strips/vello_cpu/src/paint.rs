// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use alloc::boxed::Box;
use alloc::sync::Arc;
use core::fmt;
use vello_common::encode::EncodedPaint;
use vello_common::kurbo::{Affine, Point, Vec2};
use vello_common::paint::PaintType as PenikoPaint;

/// The coordinates and dimensions of a span passed to a custom paint shader.
#[derive(Clone, Copy, Debug)]
pub struct PaintSpan {
    /// The paint-space position of the center of the first pixel.
    pub start: Point,
    /// The paint-space advance for moving one pixel in the x direction.
    pub x_advance: Vec2,
    /// The paint-space advance for moving one pixel in the y direction.
    pub y_advance: Vec2,
    /// The number of pixel columns in the output buffer.
    pub width: u16,
    /// The number of pixel rows in the output buffer. This is currently always four.
    pub height: u16,
}

/// A custom CPU paint shader.
///
/// Output pixels use premultiplied RGBA and are arranged in column-major order. The index of a
/// component is `((x * span.height + y) * 4 + component)`. Implementations must fill the entire
/// buffer and may be called concurrently and in an unspecified order.
pub trait PaintShader: Send + Sync + 'static {
    /// Fill a buffer with premultiplied RGBA8 pixels.
    fn paint_u8(&self, buffer: &mut [u8], span: PaintSpan);

    /// Fill a buffer with premultiplied RGBAF32 pixels whose components are in the `0.0..=1.0`
    /// range.
    fn paint_f32(&self, buffer: &mut [f32], span: PaintSpan);
}

/// An owned custom paint shader.
///
/// TODO: Link to the custom paint test for an example.
#[derive(Clone)]
pub struct CustomPaint {
    pub(crate) shader: Arc<dyn PaintShader>,
    may_have_transparency: bool,
}

impl CustomPaint {
    /// Create a custom paint that may produce transparent pixels.
    pub fn new(shader: impl PaintShader) -> Self {
        Self::from_arc(Arc::new(shader))
    }

    /// Create a custom paint from a shared shader that may produce transparent pixels.
    pub fn from_arc(shader: Arc<dyn PaintShader>) -> Self {
        Self {
            shader,
            may_have_transparency: true,
        }
    }

    /// Create a custom paint from an owned shader that may produce transparent pixels.
    pub fn from_box(shader: Box<dyn PaintShader>) -> Self {
        Self::from_arc(shader.into())
    }

    /// Declare that this paint always produces fully opaque pixels.
    ///
    /// Incorrectly marking a transparent shader as opaque can produce incorrect rendering due to
    /// depth culling.
    #[must_use]
    pub fn opaque(mut self) -> Self {
        self.may_have_transparency = false;
        self
    }

    /// Return whether this paint may produce transparent pixels.
    pub fn may_have_transparency(&self) -> bool {
        self.may_have_transparency
    }
}

impl fmt::Debug for CustomPaint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomPaint")
            .field("may_have_transparency", &self.may_have_transparency)
            .finish_non_exhaustive()
    }
}

impl PartialEq for CustomPaint {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.shader, &other.shader)
            && self.may_have_transparency == other.may_have_transparency
    }
}

/// A paint supported by Vello CPU.
#[derive(Clone, Debug)]
pub enum PaintType {
    /// A built-in Peniko brush.
    Brush(PenikoPaint),
    /// A custom CPU paint shader.
    Custom(CustomPaint),
}

impl<T> From<T> for PaintType
where
    T: Into<PenikoPaint>,
{
    fn from(value: T) -> Self {
        Self::Brush(value.into())
    }
}

impl From<CustomPaint> for PaintType {
    fn from(value: CustomPaint) -> Self {
        Self::Custom(value)
    }
}

#[doc(hidden)]
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "boxing built-in paints would add an allocation for every non-solid paint"
)]
pub enum PaintResource {
    BuiltIn(EncodedPaint),
    Custom(EncodedCustomPaint),
}

impl From<EncodedPaint> for PaintResource {
    fn from(value: EncodedPaint) -> Self {
        Self::BuiltIn(value)
    }
}

impl PaintResource {
    pub(crate) fn may_have_transparency(&self) -> bool {
        match self {
            Self::BuiltIn(paint) => paint.may_have_transparency(),
            Self::Custom(paint) => paint.paint.may_have_transparency,
        }
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct EncodedCustomPaint {
    pub(crate) paint: CustomPaint,
    pub(crate) transform: Affine,
    pub(crate) x_advance: Vec2,
    pub(crate) y_advance: Vec2,
}

impl EncodedCustomPaint {
    pub(crate) fn new(paint: CustomPaint, transform: Affine) -> Self {
        let transform = transform.inverse();
        let c = transform.as_coeffs();

        Self {
            paint,
            transform,
            x_advance: Vec2::new(c[0], c[1]),
            y_advance: Vec2::new(c[2], c[3]),
        }
    }
}
