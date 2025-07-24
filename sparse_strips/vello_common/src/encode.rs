// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Paints for drawing shapes.

use crate::blurred_rounded_rect::BlurredRoundedRectangle;
use crate::color::palette::css::BLACK;
use crate::color::{ColorSpaceTag, HueDirection, Srgb, gradient};
use crate::kurbo::{Affine, Point, Vec2};
use crate::math::{FloatExt, compute_erf7};
use crate::paint::{Image, ImageSource, IndexedPaint, Paint, PremulColor};
use crate::peniko::{ColorStop, Extend, Gradient, GradientKind, ImageQuality};
use alloc::borrow::Cow;
use alloc::fmt::Debug;
use alloc::vec;
use alloc::vec::Vec;
#[cfg(not(feature = "multithreading"))]
use core::cell::OnceCell;
use fearless_simd::{Simd, SimdBase, SimdFloat, f32x4, f32x16};
use smallvec::SmallVec;
// So we can just use `OnceCell` regardless of which feature is activated.
#[cfg(feature = "multithreading")]
use std::sync::OnceLock as OnceCell;

use crate::simd::{Splat4thExt, element_wise_splat};
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

const DEGENERATE_THRESHOLD: f32 = 1.0e-6;
const NUDGE_VAL: f32 = 1.0e-7;
const PIXEL_CENTER_OFFSET: f64 = 0.5;

#[cfg(feature = "std")]
fn exp(val: f32) -> f32 {
    val.exp()
}

#[cfg(not(feature = "std"))]
fn exp(val: f32) -> f32 {
    #[cfg(feature = "libm")]
    return libm::expf(val);
    #[cfg(not(feature = "libm"))]
    compile_error!("vello_common requires either the `std` or `libm` feature");
}

/// A trait for encoding gradients.
pub trait EncodeExt: private::Sealed {
    /// Encode the gradient and push it into a vector of encoded paints, returning
    /// the corresponding paint in the process. This will also validate the gradient.
    fn encode_into(&self, paints: &mut Vec<EncodedPaint>, transform: Affine) -> Paint;
}

impl EncodeExt for Gradient {
    /// Encode the gradient into a paint.
    fn encode_into(&self, paints: &mut Vec<EncodedPaint>, transform: Affine) -> Paint {
        // First make sure that the gradient is valid and not degenerate.
        if let Err(paint) = validate(self) {
            return paint;
        }

        let mut has_opacities = self.stops.iter().any(|s| s.color.components[3] != 1.0);
        let pad = self.extend == Extend::Pad;

        let mut base_transform;

        let mut stops = Cow::Borrowed(&self.stops.0);

        let kind = match self.kind {
            GradientKind::Linear {
                start: p0,
                end: mut p1,
            } => {
                // Double the length of the iterator, and append stops in reverse order in case
                // we have the extend `Reflect`.
                // Then we can treat it the same as a repeated gradient.
                if self.extend == Extend::Reflect {
                    p1.x += p1.x - p0.x;
                    p1.y += p1.y - p0.y;
                    stops = Cow::Owned(apply_reflect(&stops));
                }

                // We update the transform currently in-place, such that the gradient line always
                // starts at the point (0, 0) and ends at the point (1, 0). This simplifies the
                // calculation for the current position along the gradient line a lot.
                base_transform = ts_from_line_to_line(p0, p1, Point::ZERO, Point::new(1.0, 0.0));

                EncodedKind::Linear(LinearKind)
            }
            GradientKind::Radial {
                start_center: c0,
                start_radius: r0,
                end_center: mut c1,
                end_radius: mut r1,
            } => {
                // The implementation of radial gradients is translated from Skia.
                // See:
                // - <https://skia.org/docs/dev/design/conical/>
                // - <https://github.com/google/skia/blob/main/src/shaders/gradients/SkConicalGradient.h>
                // - <https://github.com/google/skia/blob/main/src/shaders/gradients/SkConicalGradient.cpp>

                // Same story as for linear gradients, mutate stops so that reflect and repeat
                // can be treated the same.
                if self.extend == Extend::Reflect {
                    c1 += c1 - c0;
                    r1 += r1 - r0;
                    stops = Cow::Owned(apply_reflect(&stops));
                }

                let d_radius = r1 - r0;

                // <https://github.com/google/skia/blob/1e07a4b16973cf716cb40b72dd969e961f4dd950/src/shaders/gradients/SkConicalGradient.cpp#L83-L112>
                let radial_kind = if ((c1 - c0).length() as f32).is_nearly_zero() {
                    base_transform = Affine::translate((-c1.x, -c1.y));
                    base_transform = base_transform.then_scale(1.0 / r0.max(r1) as f64);

                    let scale = r1.max(r0) / d_radius;
                    let bias = -r0 / d_radius;

                    RadialKind::Radial { bias, scale }
                } else {
                    base_transform =
                        ts_from_line_to_line(c0, c1, Point::ZERO, Point::new(1.0, 0.0));

                    if (r1 - r0).is_nearly_zero() {
                        let scaled_r0 = r1 / (c1 - c0).length() as f32;
                        RadialKind::Strip {
                            scaled_r0_squared: scaled_r0 * scaled_r0,
                        }
                    } else {
                        let d_center = (c0 - c1).length() as f32;

                        let focal_data =
                            FocalData::create(r0 / d_center, r1 / d_center, &mut base_transform);

                        let fp0 = 1.0 / focal_data.fr1;
                        let fp1 = focal_data.f_focal_x;

                        RadialKind::Focal {
                            focal_data,
                            fp0,
                            fp1,
                        }
                    }
                };

                // Even if the gradient has no stops with transparency, we might have to force
                // alpha-compositing in case the radial gradient is undefined in certain positions,
                // in which case the resulting color will be transparent and thus the gradient overall
                // must be treated as non-opaque.
                has_opacities |= radial_kind.has_undefined();

                EncodedKind::Radial(radial_kind)
            }
            GradientKind::Sweep {
                center,
                start_angle,
                end_angle,
            } => {
                // For sweep gradients, the position on the "color line" is defined by the
                // angle towards the gradient center.
                let start_angle = start_angle.to_radians();
                let mut end_angle = end_angle.to_radians();

                // Same as before, reduce `Reflect` to `Repeat`.
                if self.extend == Extend::Reflect {
                    end_angle += end_angle - start_angle;
                    stops = Cow::Owned(apply_reflect(&stops));
                }

                // Make sure the center of the gradient falls on the origin (0, 0), to make
                // angle calculation easier.
                let x_offset = -center.x as f32;
                let y_offset = -center.y as f32;
                base_transform = Affine::translate((x_offset as f64, y_offset as f64));

                EncodedKind::Sweep(SweepKind {
                    start_angle,
                    // Save the inverse so that we can use a multiplication in the shader instead.
                    inv_angle_delta: 1.0 / (end_angle - start_angle),
                })
            }
        };

        let ranges = encode_stops(&stops, self.interpolation_cs, self.hue_direction);

        // This represents the transform that needs to be applied to the starting point of a
        // command before starting with the rendering.
        // First we need to account for the base transform of the shader, then
        // we account for the fact that we sample in the center of a pixel and not in the corner by
        // adding 0.5.
        // Finally, we need to apply the _inverse_ paint transform to the point so that we can account
        // for the paint transform of the render context.
        let transform = base_transform
            * transform.inverse()
            * Affine::translate((PIXEL_CENTER_OFFSET, PIXEL_CENTER_OFFSET));

        // One possible approach of calculating the positions would be to apply the above
        // transform to _each_ pixel that we render in the wide tile. However, a much better
        // approach is to apply the transform once for the first pixel in each wide tile,
        // and from then on only apply incremental updates to the current x/y position
        // that we calculate based on the transform.
        //
        // Remember that we render wide tiles in column major order (i.e. we first calculate the
        // values for a specific x for all Tile::HEIGHT y by incrementing y by 1, and then finally
        // we increment the x position by 1 and start from the beginning). If we want to implement
        // the above approach of incrementally updating the position, we need to calculate
        // how the x/y unit vectors are affected by the transform, and then use this as the
        // step delta for a step in the x/y direction.
        let (x_advance, y_advance) = x_y_advances(&transform);

        let encoded = EncodedGradient {
            kind,
            transform,
            x_advance,
            y_advance,
            ranges,
            pad,
            has_opacities,
            u8_lut: OnceCell::new(),
            f32_lut: OnceCell::new(),
        };

        let idx = paints.len();
        paints.push(encoded.into());

        Paint::Indexed(IndexedPaint::new(idx))
    }
}

/// Returns a fallback paint in case the gradient is invalid.
///
/// The paint will be either black or contain the color of the first stop of the gradient.
fn validate(gradient: &Gradient) -> Result<(), Paint> {
    let black = Err(BLACK.into());

    // Gradients need at least two stops.
    if gradient.stops.is_empty() {
        return black;
    }

    let first = Err(gradient.stops[0].color.to_alpha_color::<Srgb>().into());

    if gradient.stops.len() == 1 {
        return first;
    }

    // First stop must be at offset 0.0 and last offset must be at 1.0.
    if gradient.stops[0].offset != 0.0 || gradient.stops[gradient.stops.len() - 1].offset != 1.0 {
        return first;
    }

    for stops in gradient.stops.windows(2) {
        let f = stops[0];
        let n = stops[1];

        // Offsets must be between 0 and 1.
        if f.offset > 1.0 || f.offset < 0.0 {
            return first;
        }

        // Stops must be sorted by ascending offset.
        if f.offset > n.offset {
            return first;
        }
    }

    let degenerate_point = |p1: &Point, p2: &Point| {
        (p1.x - p2.x).abs() as f32 <= DEGENERATE_THRESHOLD
            && (p1.y - p2.y).abs() as f32 <= DEGENERATE_THRESHOLD
    };

    let degenerate_val = |v1: f32, v2: f32| (v2 - v1).abs() <= DEGENERATE_THRESHOLD;

    match &gradient.kind {
        GradientKind::Linear { start, end } => {
            // Start and end points must not be too close together.
            if degenerate_point(start, end) {
                return first;
            }
        }
        GradientKind::Radial {
            start_center,
            start_radius,
            end_center,
            end_radius,
        } => {
            // Radii must not be negative.
            if *start_radius < 0.0 || *end_radius < 0.0 {
                return first;
            }

            // Radii and center points must not be close to the same.
            if degenerate_point(start_center, end_center)
                && degenerate_val(*start_radius, *end_radius)
            {
                return first;
            }
        }
        GradientKind::Sweep {
            start_angle,
            end_angle,
            ..
        } => {
            // The end angle must be larger than the start angle.
            if degenerate_val(*start_angle, *end_angle) {
                return first;
            }

            if end_angle <= start_angle {
                return first;
            }
        }
    }

    Ok(())
}

/// Extend the stops so that we can treat a repeated gradient like a reflected gradient.
fn apply_reflect(stops: &[ColorStop]) -> SmallVec<[ColorStop; 4]> {
    let first_half = stops.iter().map(|s| ColorStop {
        offset: s.offset / 2.0,
        color: s.color,
    });

    let second_half = stops.iter().rev().map(|s| ColorStop {
        offset: 0.5 + (1.0 - s.offset) / 2.0,
        color: s.color,
    });

    first_half.chain(second_half).collect::<SmallVec<_>>()
}

/// Encode all stops into a sequence of ranges.
fn encode_stops(
    stops: &[ColorStop],
    cs: ColorSpaceTag,
    hue_dir: HueDirection,
) -> Vec<GradientRange> {
    #[derive(Debug)]
    struct EncodedColorStop {
        offset: f32,
        color: crate::color::PremulColor<Srgb>,
    }

    let create_range = |left_stop: &EncodedColorStop, right_stop: &EncodedColorStop| {
        let clamp = |mut color: [f32; 4]| {
            // The linear approximation of the gradient can produce values slightly outside of
            // [0.0, 1.0], so clamp them.
            for c in &mut color {
                *c = c.clamp(0.0, 1.0);
            }

            color
        };

        let x0 = left_stop.offset;
        let x1 = right_stop.offset;
        let c0 = clamp(left_stop.color.components);
        let c1 = clamp(right_stop.color.components);

        // We calculate a bias and scale factor, such that we can simply calculate
        // bias + x * scale to get the interpolated color, where x is between x0 and x1,
        // to calculate the resulting color.
        // Apply a nudge value because we sometimes call `create_range` with the same offset
        // to create the padded stops.
        let x1_minus_x0 = (x1 - x0).max(NUDGE_VAL);
        let mut scale = [0.0; 4];
        let mut bias = c0;

        for i in 0..4 {
            scale[i] = (c1[i] - c0[i]) / x1_minus_x0;
            bias[i] = c0[i] - x0 * scale[i];
        }

        GradientRange { x1, bias, scale }
    };

    // Create additional (SRGB-encoded) stops in-between to approximate the color space we want to
    // interpolate in.
    if cs != ColorSpaceTag::Srgb {
        let interpolated_stops = stops
            .windows(2)
            .flat_map(|s| {
                let left_stop = &s[0];
                let right_stop = &s[1];

                let interpolated =
                    gradient::<Srgb>(left_stop.color, right_stop.color, cs, hue_dir, 0.01);

                interpolated.map(|st| EncodedColorStop {
                    offset: left_stop.offset + (right_stop.offset - left_stop.offset) * st.0,
                    color: st.1,
                })
            })
            .collect::<Vec<_>>();

        interpolated_stops
            .windows(2)
            .map(|s| {
                let left_stop = &s[0];
                let right_stop = &s[1];

                create_range(left_stop, right_stop)
            })
            .collect()
    } else {
        stops
            .windows(2)
            .map(|c| {
                let c0 = EncodedColorStop {
                    offset: c[0].offset,
                    color: c[0].color.to_alpha_color::<Srgb>().premultiply(),
                };

                let c1 = EncodedColorStop {
                    offset: c[1].offset,
                    color: c[1].color.to_alpha_color::<Srgb>().premultiply(),
                };

                create_range(&c0, &c1)
            })
            .collect()
    }
}

pub(crate) fn x_y_advances(transform: &Affine) -> (Vec2, Vec2) {
    let scale_skew_transform = {
        let c = transform.as_coeffs();
        Affine::new([c[0], c[1], c[2], c[3], 0.0, 0.0])
    };

    let x_advance = scale_skew_transform * Point::new(1.0, 0.0);
    let y_advance = scale_skew_transform * Point::new(0.0, 1.0);

    (
        Vec2::new(x_advance.x, x_advance.y),
        Vec2::new(y_advance.x, y_advance.y),
    )
}

impl private::Sealed for Image {}

impl EncodeExt for Image {
    fn encode_into(&self, paints: &mut Vec<EncodedPaint>, transform: Affine) -> Paint {
        let idx = paints.len();

        let mut quality = self.quality;

        let c = transform.as_coeffs();

        // Optimize image quality for integer-only translations.
        if (c[0] as f32 - 1.0).is_nearly_zero()
            && (c[1] as f32).is_nearly_zero()
            && (c[2] as f32).is_nearly_zero()
            && (c[3] as f32 - 1.0).is_nearly_zero()
            && ((c[4] - c[4].floor()) as f32).is_nearly_zero()
            && ((c[5] - c[5].floor()) as f32).is_nearly_zero()
            && quality == ImageQuality::Medium
        {
            quality = ImageQuality::Low;
        }

        // Similarly to gradients, apply a 0.5 offset so we sample at the center of
        // a pixel.
        let transform = transform.inverse() * Affine::translate((0.5, 0.5));

        let (x_advance, y_advance) = x_y_advances(&transform);

        let encoded = match &self.source {
            ImageSource::Pixmap(pixmap) => {
                EncodedImage {
                    source: ImageSource::Pixmap(pixmap.clone()),
                    extends: (self.x_extend, self.y_extend),
                    quality,
                    // While we could optimize RGB8 images, it's probably not worth the trouble.
                    has_opacities: true,
                    transform,
                    x_advance,
                    y_advance,
                }
            }
            ImageSource::OpaqueId(image) => EncodedImage {
                source: ImageSource::OpaqueId(*image),
                extends: (self.x_extend, self.y_extend),
                quality: self.quality,
                has_opacities: true,
                transform,
                x_advance,
                y_advance,
            },
        };

        paints.push(EncodedPaint::Image(encoded));

        Paint::Indexed(IndexedPaint::new(idx))
    }
}

/// An encoded paint.
#[derive(Debug)]
pub enum EncodedPaint {
    /// An encoded gradient.
    Gradient(EncodedGradient),
    /// An encoded image.
    Image(EncodedImage),
    /// A blurred, rounded rectangle.
    BlurredRoundedRect(EncodedBlurredRoundedRectangle),
}

impl From<EncodedGradient> for EncodedPaint {
    fn from(value: EncodedGradient) -> Self {
        Self::Gradient(value)
    }
}

impl From<EncodedBlurredRoundedRectangle> for EncodedPaint {
    fn from(value: EncodedBlurredRoundedRectangle) -> Self {
        Self::BlurredRoundedRect(value)
    }
}

/// An encoded image.
#[derive(Debug)]
pub struct EncodedImage {
    /// The underlying pixmap of the image.
    pub source: ImageSource,
    /// The extends in the horizontal and vertical direction.
    pub extends: (Extend, Extend),
    /// The rendering quality of the image.
    pub quality: ImageQuality,
    /// Whether the image has opacities.
    pub has_opacities: bool,
    /// A transform to apply to the image.
    pub transform: Affine,
    /// The advance in image coordinates for one step in the x direction.
    pub x_advance: Vec2,
    /// The advance in image coordinates for one step in the y direction.
    pub y_advance: Vec2,
}

/// Computed properties of a linear gradient.
#[derive(Debug, Copy, Clone)]
pub struct LinearKind;

/// Focal data for a radial gradient.
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct FocalData {
    /// The normalized radius of the outer circle in focal space.
    pub fr1: f32,
    /// The x-coordinate of the focal point in normalized space \[0,1\].
    pub f_focal_x: f32,
    /// Whether the focal points have been swapped.
    pub f_is_swapped: bool,
}

impl FocalData {
    /// Create a new `FocalData` with the given radii and update the matrix.
    pub fn create(mut r0: f32, mut r1: f32, matrix: &mut Affine) -> Self {
        let mut swapped = false;
        let mut f_focal_x = r0 / (r0 - r1);

        if (f_focal_x - 1.0).is_nearly_zero() {
            *matrix = matrix.then_translate(Vec2::new(-1.0, 0.0));
            *matrix = matrix.then_scale_non_uniform(-1.0, 1.0);
            core::mem::swap(&mut r0, &mut r1);
            f_focal_x = 0.0;
            swapped = true;
        }

        let focal_matrix = ts_from_line_to_line(
            Point::new(f_focal_x as f64, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
        );
        *matrix = focal_matrix * *matrix;

        let fr1 = r1 / (1.0 - f_focal_x).abs();

        let data = Self {
            fr1,
            f_focal_x,
            f_is_swapped: swapped,
        };

        if data.is_focal_on_circle() {
            *matrix = matrix.then_scale(0.5);
        } else {
            *matrix = matrix.then_scale_non_uniform(
                (fr1 / (fr1 * fr1 - 1.0)) as f64,
                1.0 / (fr1 * fr1 - 1.0).abs().sqrt() as f64,
            );
        }

        *matrix = matrix.then_scale((1.0 - f_focal_x).abs() as f64);

        data
    }

    /// Whether the focal is on the circle.
    pub fn is_focal_on_circle(&self) -> bool {
        (1.0 - self.fr1).is_nearly_zero()
    }

    /// Whether the focal points have been swapped.
    pub fn is_swapped(&self) -> bool {
        self.f_is_swapped
    }

    /// Whether the gradient is well-behaved.
    pub fn is_well_behaved(&self) -> bool {
        !self.is_focal_on_circle() && self.fr1 > 1.0
    }

    /// Whether the gradient is natively focal.
    pub fn is_natively_focal(&self) -> bool {
        self.f_focal_x.is_nearly_zero()
    }
}

/// A radial gradient.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum RadialKind {
    /// A radial gradient, i.e. the start and end center points are the same.
    Radial {
        /// The `bias` value (from the Skia implementation).
        ///
        /// It is a correction factor that accounts for the fact that the focal center might not
        /// lie on the inner circle (if r0 > 0).
        bias: f32,
        /// The `scale` value (from the Skia implementation).
        ///
        /// It is a scaling factor that maps from r0 to r1.
        scale: f32,
    },
    /// A strip gradient, i.e. the start and end radius are the same.
    Strip {
        /// The squared value of `scaled_r0` (from the Skia implementation).
        scaled_r0_squared: f32,
    },
    /// A general, two-point conical gradient.
    Focal {
        /// The focal data  (from the Skia implementation).
        focal_data: FocalData,
        /// The `fp0` value (from the Skia implementation).
        fp0: f32,
        /// The `fp1` value (from the Skia implementation).
        fp1: f32,
    },
}

impl RadialKind {
    /// Whether the gradient is undefined at any location.
    pub fn has_undefined(&self) -> bool {
        match self {
            Self::Radial { .. } => false,
            Self::Strip { .. } => true,
            Self::Focal { focal_data, .. } => !focal_data.is_well_behaved(),
        }
    }
}

/// Computed properties of a sweep gradient.
#[derive(Debug)]
pub struct SweepKind {
    /// The start angle of the sweep gradient.
    pub start_angle: f32,
    /// The inverse delta between start and end angle.
    pub inv_angle_delta: f32,
}

/// A kind of encoded gradient.
#[derive(Debug)]
pub enum EncodedKind {
    /// An encoded linear gradient.
    Linear(LinearKind),
    /// An encoded radial gradient.
    Radial(RadialKind),
    /// An encoded sweep gradient.
    Sweep(SweepKind),
}

/// An encoded gradient.
#[derive(Debug)]
pub struct EncodedGradient {
    /// The underlying kind of gradient.
    pub kind: EncodedKind,
    /// A transform that needs to be applied to the position of the first processed pixel.
    pub transform: Affine,
    /// How much to advance into the x/y direction for one step in the x direction.
    pub x_advance: Vec2,
    /// How much to advance into the x/y direction for one step in the y direction.
    pub y_advance: Vec2,
    /// The color ranges of the gradient.
    pub ranges: Vec<GradientRange>,
    /// Whether the gradient should be padded.
    pub pad: bool,
    /// Whether the gradient requires `source_over` compositing.
    pub has_opacities: bool,
    u8_lut: OnceCell<GradientLut<u8>>,
    f32_lut: OnceCell<GradientLut<f32>>,
}

impl EncodedGradient {
    /// Get the lookup table for sampling u8-based gradient values.
    pub fn u8_lut<S: Simd>(&self, simd: S) -> &GradientLut<u8> {
        self.u8_lut
            .get_or_init(|| GradientLut::new(simd, &self.ranges))
    }

    /// Get the lookup table for sampling f32-based gradient values.
    pub fn f32_lut<S: Simd>(&self, simd: S) -> &GradientLut<f32> {
        self.f32_lut
            .get_or_init(|| GradientLut::new(simd, &self.ranges))
    }
}

/// An encoded ange between two color stops.
#[derive(Debug, Clone)]
pub struct GradientRange {
    /// The end value of the range.
    pub x1: f32,
    /// A bias to apply when interpolating the color (in this case just the values of the start
    /// color of the gradient).
    pub bias: [f32; 4],
    /// The scale factors of the range. By calculating bias + x * factors (where x is
    /// between 0.0 and 1.0), we can interpolate between start and end color of the gradient range.
    pub scale: [f32; 4],
}

/// An encoded blurred, rounded rectangle.
#[derive(Debug)]
pub struct EncodedBlurredRoundedRectangle {
    /// An component for computing the blur effect.
    pub exponent: f32,
    /// An component for computing the blur effect.
    pub recip_exponent: f32,
    /// An component for computing the blur effect.
    pub scale: f32,
    /// An component for computing the blur effect.
    pub std_dev_inv: f32,
    /// An component for computing the blur effect.
    pub min_edge: f32,
    /// An component for computing the blur effect.
    pub w: f32,
    /// An component for computing the blur effect.
    pub h: f32,
    /// An component for computing the blur effect.
    pub width: f32,
    /// An component for computing the blur effect.
    pub height: f32,
    /// An component for computing the blur effect.
    pub r1: f32,
    /// The base color for the blurred rectangle.
    pub color: PremulColor,
    /// A transform that needs to be applied to the position of the first processed pixel.
    pub transform: Affine,
    /// How much to advance into the x/y direction for one step in the x direction.
    pub x_advance: Vec2,
    /// How much to advance into the x/y direction for one step in the y direction.
    pub y_advance: Vec2,
}

impl private::Sealed for BlurredRoundedRectangle {}

impl EncodeExt for BlurredRoundedRectangle {
    fn encode_into(&self, paints: &mut Vec<EncodedPaint>, transform: Affine) -> Paint {
        let rect = {
            // Ensure rectangle has positive width/height.
            let mut rect = self.rect;

            if self.rect.x0 > self.rect.x1 {
                core::mem::swap(&mut rect.x0, &mut rect.x1);
            }

            if self.rect.y0 > self.rect.y1 {
                core::mem::swap(&mut rect.y0, &mut rect.y1);
            }

            rect
        };

        let transform = Affine::translate((-rect.x0, -rect.y0)) * transform.inverse();

        let (x_advance, y_advance) = x_y_advances(&transform);

        let width = rect.width() as f32;
        let height = rect.height() as f32;
        let radius = self.radius.min(0.5 * width.min(height));

        // To avoid divide by 0; potentially should be a bigger number for antialiasing.
        let std_dev = self.std_dev.max(1e-6);

        let min_edge = width.min(height);
        let rmax = 0.5 * min_edge;
        let r0 = radius.hypot(std_dev * 1.15).min(rmax);
        let r1 = radius.hypot(std_dev * 2.0).min(rmax);

        let exponent = 2.0 * r1 / r0;

        let std_dev_inv = std_dev.recip();

        // Pull in long end (make less eccentric).
        let delta = 1.25
            * std_dev
            * (exp(-(0.5 * std_dev_inv * width).powi(2))
                - exp(-(0.5 * std_dev_inv * height).powi(2)));
        let w = width + delta.min(0.0);
        let h = height - delta.max(0.0);

        let recip_exponent = exponent.recip();
        let scale = 0.5 * compute_erf7(std_dev_inv * 0.5 * (w.max(h) - 0.5 * radius));

        let encoded = EncodedBlurredRoundedRectangle {
            exponent,
            recip_exponent,
            width,
            height,
            scale,
            r1,
            std_dev_inv,
            min_edge,
            color: PremulColor::from_alpha_color(self.color),
            w,
            h,
            transform,
            x_advance,
            y_advance,
        };

        let idx = paints.len();
        paints.push(encoded.into());

        Paint::Indexed(IndexedPaint::new(idx))
    }
}

/// Calculates the transform necessary to map the line spanned by points src1, src2 to
/// the line spanned by dst1, dst2.
///
/// This creates a transformation that maps any line segment to any other line segment.
/// For gradients, we use this to transform the gradient line to a standard form (0,0) → (1,0).
///
/// Copied from <https://github.com/linebender/tiny-skia/blob/68b198a7210a6bbf752b43d6bc4db62445730313/src/shaders/radial_gradient.rs#L182>
fn ts_from_line_to_line(src1: Point, src2: Point, dst1: Point, dst2: Point) -> Affine {
    let unit_to_line1 = unit_to_line(src1, src2);
    // Calculate the transform necessary to map line1 to the unit vector.
    let line1_to_unit = unit_to_line1.inverse();
    // Then map the unit vector to line2.
    let unit_to_line2 = unit_to_line(dst1, dst2);

    unit_to_line2 * line1_to_unit
}

/// Calculate the transform necessary to map the unit vector to the line spanned by the points
/// `p1` and `p2`.
fn unit_to_line(p0: Point, p1: Point) -> Affine {
    Affine::new([
        p1.y - p0.y,
        p0.x - p1.x,
        p1.x - p0.x,
        p1.y - p0.y,
        p0.x,
        p0.y,
    ])
}

/// A helper trait for converting a premultiplied f32 color to `Self`.
pub trait FromF32Color: Sized + Debug + Copy + Clone {
    /// The zero value.
    const ZERO: Self;
    /// Convert from a premultiplied f32 color to `Self`.
    fn from_f32<S: Simd>(color: f32x4<S>) -> [Self; 4];
}

impl FromF32Color for f32 {
    const ZERO: Self = 0.0;

    fn from_f32<S: Simd>(color: f32x4<S>) -> [Self; 4] {
        color.val
    }
}

impl FromF32Color for u8 {
    const ZERO: Self = 0;

    fn from_f32<S: Simd>(mut color: f32x4<S>) -> [Self; 4] {
        let simd = color.simd;
        color = f32x4::splat(simd, 0.5).madd(color, f32x4::splat(simd, 255.0));

        [
            color[0] as Self,
            color[1] as Self,
            color[2] as Self,
            color[3] as Self,
        ]
    }
}

/// A lookup table for sampled gradient values.
#[derive(Debug)]
pub struct GradientLut<T: FromF32Color> {
    lut: Vec<[T; 4]>,
    scale: f32,
}

impl<T: FromF32Color> GradientLut<T> {
    /// Create a new lookup table.
    fn new<S: Simd>(simd: S, ranges: &[GradientRange]) -> Self {
        // Inspired by Blend2D.
        let lut_size = match ranges.len() {
            1 => 256,
            2 => 512,
            _ => 1024,
        };

        // Add a bit of padding since we always process in blocks of 4, even though less might be
        // needed.
        let mut lut = vec![[T::ZERO, T::ZERO, T::ZERO, T::ZERO]; lut_size + 3];

        // Calculate how many indices are covered by each range.
        let ramps = {
            let mut ramps = Vec::with_capacity(ranges.len());
            let mut prev_idx = 0;

            for range in ranges {
                let max_idx = (range.x1 * lut_size as f32) as usize;

                ramps.push((prev_idx..max_idx, range));
                prev_idx = max_idx;
            }

            ramps
        };

        let scale = lut_size as f32 - 1.0;

        let inv_lut_scale = f32x4::splat(simd, 1.0 / scale);
        let add_factor = f32x4::from_slice(simd, &[0.0, 1.0, 2.0, 3.0]) * inv_lut_scale;

        for (ramp_range, range) in ramps {
            let biases = f32x16::block_splat(f32x4::from_slice(simd, &range.bias));
            let scales = f32x16::block_splat(f32x4::from_slice(simd, &range.scale));

            ramp_range.step_by(4).for_each(|idx| {
                let t_vals = add_factor.madd(f32x4::splat(simd, idx as f32), inv_lut_scale);

                let t_vals = element_wise_splat(simd, t_vals);

                let mut result = biases.madd(scales, t_vals);
                let alphas = result.splat_4th();
                // Due to floating-point impreciseness, it can happen that
                // values either become greater than 1 or the RGB channels
                // become greater than the alpha channel. To prevent overflows
                // in later parts of the pipeline, we need to take the minimum here.
                result = result.min(1.0).min(alphas);
                let (im1, im2) = simd.split_f32x16(result);
                let (r1, r2) = simd.split_f32x8(im1);
                let (r3, r4) = simd.split_f32x8(im2);

                let lut = &mut lut[idx..][..4];
                lut[0] = T::from_f32(r1);
                lut[1] = T::from_f32(r2);
                lut[2] = T::from_f32(r3);
                lut[3] = T::from_f32(r4);
            });
        }

        // Due to SIMD we worked in blocks of 4, so we need to truncate to the actual length.
        lut.truncate(lut_size);

        Self { lut, scale }
    }

    /// Get the sample value at a specific index.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> [T; 4] {
        self.lut[idx]
    }

    /// Return the raw array of gradient sample values.
    #[inline(always)]
    pub fn lut(&self) -> &[[T; 4]] {
        &self.lut
    }

    /// Get the scale factor by which to scale the parametric value to
    /// compute the correct lookup index.
    #[inline(always)]
    pub fn scale_factor(&self) -> f32 {
        self.scale
    }
}

mod private {
    #[expect(unnameable_types, reason = "Sealed trait pattern.")]
    pub trait Sealed {}

    impl Sealed for super::Gradient {}
}

#[cfg(test)]
mod tests {
    use super::{EncodeExt, Gradient};
    use crate::color::DynamicColor;
    use crate::color::palette::css::{BLACK, BLUE, GREEN};
    use crate::kurbo::{Affine, Point};
    use crate::peniko::{ColorStop, ColorStops, GradientKind};
    use alloc::vec;
    use smallvec::smallvec;

    #[test]
    fn gradient_missing_stops() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(20.0, 0.0),
            },
            ..Default::default()
        };

        assert_eq!(
            gradient.encode_into(&mut buf, Affine::IDENTITY),
            BLACK.into()
        );
    }

    #[test]
    fn gradient_one_stop() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(20.0, 0.0),
            },
            stops: ColorStops(smallvec![ColorStop {
                offset: 0.0,
                color: DynamicColor::from_alpha_color(GREEN),
            }]),
            ..Default::default()
        };

        // Should return the color of the first stop.
        assert_eq!(
            gradient.encode_into(&mut buf, Affine::IDENTITY),
            GREEN.into()
        );
    }

    #[test]
    fn gradient_not_padded_stops() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(20.0, 0.0),
            },
            stops: ColorStops(smallvec![
                ColorStop {
                    offset: 0.0,
                    color: DynamicColor::from_alpha_color(GREEN),
                },
                ColorStop {
                    offset: 0.5,
                    color: DynamicColor::from_alpha_color(BLUE),
                },
            ]),
            ..Default::default()
        };

        assert_eq!(
            gradient.encode_into(&mut buf, Affine::IDENTITY),
            GREEN.into()
        );
    }

    #[test]
    fn gradient_not_sorted_stops() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(20.0, 0.0),
            },
            stops: ColorStops(smallvec![
                ColorStop {
                    offset: 1.0,
                    color: DynamicColor::from_alpha_color(GREEN),
                },
                ColorStop {
                    offset: 0.0,
                    color: DynamicColor::from_alpha_color(BLUE),
                },
            ]),
            ..Default::default()
        };

        assert_eq!(
            gradient.encode_into(&mut buf, Affine::IDENTITY),
            GREEN.into()
        );
    }

    #[test]
    fn gradient_linear_degenerate() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(0.0, 0.0),
            },
            stops: ColorStops(smallvec![
                ColorStop {
                    offset: 0.0,
                    color: DynamicColor::from_alpha_color(GREEN),
                },
                ColorStop {
                    offset: 1.0,
                    color: DynamicColor::from_alpha_color(BLUE),
                },
            ]),
            ..Default::default()
        };

        assert_eq!(
            gradient.encode_into(&mut buf, Affine::IDENTITY),
            GREEN.into()
        );
    }

    #[test]
    fn gradient_radial_degenerate() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Radial {
                start_center: Point::new(0.0, 0.0),
                start_radius: 20.0,
                end_center: Point::new(0.0, 0.0),
                end_radius: 20.0,
            },
            stops: ColorStops(smallvec![
                ColorStop {
                    offset: 0.0,
                    color: DynamicColor::from_alpha_color(GREEN),
                },
                ColorStop {
                    offset: 1.0,
                    color: DynamicColor::from_alpha_color(BLUE),
                },
            ]),
            ..Default::default()
        };

        assert_eq!(
            gradient.encode_into(&mut buf, Affine::IDENTITY),
            GREEN.into()
        );
    }
}
