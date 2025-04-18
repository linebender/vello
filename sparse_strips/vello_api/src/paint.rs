// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types for paints.

use crate::kurbo::Affine;
use peniko::color::{AlphaColor, PremulRgba8, Srgb};
use peniko::{ColorStops, GradientKind};

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Paint {
    /// A premultiplied RGBA8 color.
    Solid(PremulRgba8),
    /// A paint that needs to be resolved via an index.
    Indexed(IndexedPaint),
}

impl From<AlphaColor<Srgb>> for Paint {
    fn from(value: AlphaColor<Srgb>) -> Self {
        // TODO: This might be slow on x86, see https://github.com/linebender/color/issues/142.
        // Since we only do that conversion once per path it might not be critical, but should
        // still be measured. This also applies to all other usages of `to_rgba8` in the current
        // code.
        Self::Solid(value.premultiply().to_rgba8())
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

/// A kind of paint that can be used for filling and stroking shapes.
#[derive(Debug, Clone)]
pub enum PaintType {
    /// A solid color.
    Solid(AlphaColor<Srgb>),
    /// A gradient.
    Gradient(Gradient),
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
