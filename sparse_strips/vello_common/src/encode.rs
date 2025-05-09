// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Paints for drawing shapes.

use crate::blurred_rounded_rect::BlurredRoundedRectangle;
use crate::color::palette::css::BLACK;
use crate::color::{ColorSpaceTag, HueDirection, Srgb, gradient};
use crate::kurbo::{Affine, Point, Vec2};
use crate::math::compute_erf7;
use crate::paint::{Image, IndexedPaint, Paint, PremulColor};
use crate::peniko::{ColorStop, Extend, Gradient, GradientKind, ImageQuality};
use crate::pixmap::Pixmap;
use alloc::borrow::Cow;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::f32::consts::PI;
use core::iter;
use smallvec::SmallVec;

#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

const DEGENERATE_THRESHOLD: f32 = 1.0e-6;
const NUDGE_VAL: f32 = 1.0e-7;

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

        let mut stops = Cow::Borrowed(&self.stops.0);
        // For each gradient type, before doing anything we first translate it such that
        // one of the points of the gradient lands on the origin (0, 0). We do this because
        // it makes things simpler and allows for some optimizations for certain calculations.
        let (x_offset, y_offset);
        // The start/end range of the color line. We use this to resolve the extend of the gradient.
        // Currently radial gradients uses normalized values between 0.0 and 1.0, for sweep and
        // linear gradients different values are used (TODO: Would be nice to make this more consistent).
        let mut clamp_range = (0.0, 1.0);

        let kind = match self.kind {
            GradientKind::Linear { start, end } => {
                // For linear gradients, we want to interpolate the color along the line that is
                // formed by `start` and `end`.
                let mut p0 = start;
                let mut p1 = end;

                // For simplicity, ensure that the gradient line always goes from left to right.
                if p0.x >= p1.x {
                    core::mem::swap(&mut p0, &mut p1);

                    stops = Cow::Owned(
                        stops
                            .iter()
                            .rev()
                            .map(|s| ColorStop {
                                offset: 1.0 - s.offset,
                                color: s.color,
                            })
                            .collect::<SmallVec<[ColorStop; 4]>>(),
                    );
                }

                // Double the length of the iterator, and append stops in reverse order in case
                // we have the extend `Reflect`.
                // Then we can treat it the same as a repeated gradient.
                if self.extend == Extend::Reflect {
                    p1.x += p1.x - p0.x;
                    p1.y += p1.y - p0.y;
                    stops = Cow::Owned(apply_reflect(&stops));
                }

                // To translate p0 to the origin of the coordinate system, we need to apply
                // the negative.
                x_offset = -p0.x as f32;
                y_offset = -p0.y as f32;

                let dx = p1.x as f32 + x_offset;
                let dy = p1.y as f32 + y_offset;
                // In order to calculate where a pixel lies along the gradient line (the line made up
                // by the two points of the linear gradient), we need to calculate its position
                // on the gradient line. Remember that our gradient line always start at the origin
                // (0, 0). Therefore, we can simply calculate the normal vector of the line,
                // and then, for each pixel that we render, we calculate the distance to the line.
                // That distance then corresponds to our position on the gradient line, and allows
                // us to resolve which color stops we need to load and how to interpolate them.
                let norm = (-dy, dx);

                // We precalculate some values so that we can more easily calculate the distance
                // from the position of the pixel to the line of the normal vector. See
                // here for the formula: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points

                // The denominator, i.e. sqrt((y_2 - y_1)^2 + (x_2 - x_1)^2). Since x_1 and y_1
                // are always 0, this shortens to sqrt(y_2^2 + x_2^2).
                let distance = (norm.1 * norm.1 + norm.0 * norm.0).sqrt();
                // This corresponds to (y_2 - y_1) in the formula, but because of the above reasons
                // shortens to y_2.
                let y2_minus_y1 = norm.1;
                // This corresponds to (x_2 - x_1) in the formula, but because of the above reasons
                // shortens to x_2.
                let x2_minus_x1 = norm.0;
                // Note that we can completely disregard the x_2 * y_1 - y_2 * x_1 factor, since
                // y_1 and x_1 are both 0.

                let end_val = (dx * dx + dy * dy).sqrt();
                clamp_range = (0.0, end_val);

                EncodedKind::Linear(LinearKind {
                    distance,
                    y2_minus_y1,
                    x2_minus_x1,
                })
            }
            GradientKind::Radial {
                start_center,
                start_radius,
                end_center,
                end_radius,
            } => {
                // For radial gradients, we conceptually interpolate a circle from c0 with radius
                // r0 to the circle at c1 with radius r1.
                let c0 = start_center;
                let mut c1 = end_center;
                let r0 = start_radius;
                let mut r1 = end_radius;

                // Same story as for linear gradients, mutate stops so that reflect and repeat
                // can be treated the same.
                if self.extend == Extend::Reflect {
                    c1 += c1 - c0;
                    r1 += r1 - r0;
                    stops = Cow::Owned(apply_reflect(&stops));
                }

                // Similarly to linear gradients, ensure that c0 lands on the origin (0, 0).
                x_offset = -c0.x as f32;
                y_offset = -c0.y as f32;

                let end_point = c1 - c0;

                let dist = (end_point.x * end_point.x + end_point.y * end_point.y).sqrt() as f32;
                let c0_in_c1 = r1 >= r0 + dist;
                let c1_in_c0 = r0 >= r1 + dist;
                let cone_like = !(c0_in_c1 || c1_in_c0);
                // If the inner circle is not completely contained within the outer circle, the gradient
                // can deform into a cone-like structure where some areas of the shape are not defined.
                // Because of this, we might need opacities and source-over compositing in that case.
                has_opacities |= cone_like;

                EncodedKind::Radial(RadialKind {
                    c1: (end_point.x as f32, end_point.y as f32),
                    r0,
                    r1,
                    cone_like,
                })
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

                // Make sure the center of the gradient falls on the origin (0, 0).
                x_offset = -center.x as f32;
                y_offset = -center.y as f32;
                clamp_range = (start_angle, end_angle);

                EncodedKind::Sweep(SweepKind)
            }
        };

        let ranges = encode_stops(
            &stops,
            clamp_range.0,
            clamp_range.1,
            pad,
            self.interpolation_cs,
            self.hue_direction,
        );

        // This represents the transform that needs to be applied to the starting point of a
        // command before starting with the rendering.
        // First we need to account for a potential offset of the gradient (x_offset/y_offset), then
        // we account for the fact that we sample in the center of a pixel and not in the corner by
        // adding 0.5.
        // Finally, we need to apply the _inverse_ transform to the point so that we can account
        // for the transform on the gradient.
        let transform = Affine::translate((f64::from(x_offset) + 0.5, f64::from(y_offset) + 0.5))
            * transform.inverse();

        // One possible approach of calculating the positions would be to apply the above
        // transform to _each_ pixel that we render in the wide tile. However, a much better
        // approach is to apply the transform once for the first pixel,
        // and from then on only apply incremental updates to the current x/y position
        // that we calculated in the beginning.
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
            clamp_range,
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
    start: f32,
    end: f32,
    pad: bool,
    cs: ColorSpaceTag,
    hue_dir: HueDirection,
) -> Vec<GradientRange> {
    struct EncodedColorStop {
        offset: f32,
        color: crate::color::PremulColor<Srgb>,
    }

    // Create additional (SRGB-encoded) stops in-between to approximate the color space we want to
    // interpolate in.
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

    let create_range = |left_stop: &EncodedColorStop, right_stop: &EncodedColorStop| {
        let clamp = |mut color: [f32; 4]| {
            // The linear approximation of the gradient can produce values slightly outside of
            // [0.0, 1.0], so clamp them.
            for c in &mut color {
                *c = c.clamp(0.0, 1.0);
            }

            color
        };

        let x0 = start + (end - start) * left_stop.offset;
        let x1 = start + (end - start) * right_stop.offset;
        let c0 = clamp(left_stop.color.components);
        let c1 = clamp(right_stop.color.components);

        // Given two positions x0 and x1 as well as two corresponding colors c0 and c1,
        // the delta that needs to be applied to c0 to calculate the color of x between x0 and x1
        // is calculated by c0 + ((x - x0) / (x1 - x0)) * (c1 - c0).
        // We can precompute the (c1 - c0)/(x1 - x0) part for each color component.

        // We call this method with two same stops for `left_range` and `right_range`, so make
        // sure we don't actually end up with a 0 here.
        let x1_minus_x0 = (x1 - x0).max(NUDGE_VAL);
        let mut factors_f32 = [0.0; 4];

        for i in 0..4 {
            let c1_minus_c0 = c1[i] - c0[i];
            factors_f32[i] = c1_minus_c0 / x1_minus_x0;
        }

        GradientRange {
            x0,
            x1,
            c0: PremulColor::from_premul_color(left_stop.color),
            factors_f32,
        }
    };

    // Note: this could use `Iterator::map_windows` once stabilized, meaning `interpolated_stops`
    // no longer needs to be collected.
    let stop_ranges = interpolated_stops.windows(2).map(|s| {
        let left_stop = &s[0];
        let right_stop = &s[1];

        create_range(left_stop, right_stop)
    });

    if pad {
        // We handle padding by inserting dummy stops in the beginning and end with a very big
        // range.
        let left_range = iter::once({
            let first_stop = interpolated_stops.first().unwrap();
            let mut encoded_range = create_range(first_stop, first_stop);
            encoded_range.x0 = f32::MIN;
            encoded_range
        });

        let right_range = iter::once({
            let last_stop = interpolated_stops.last().unwrap();
            let mut encoded_range = create_range(last_stop, last_stop);
            encoded_range.x1 = f32::MAX;
            encoded_range
        });

        left_range.chain(stop_ranges.chain(right_range)).collect()
    } else {
        stop_ranges.collect()
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

        let transform = transform.inverse();
        // TODO: This is somewhat expensive for large images, maybe it's not worth optimizing
        // non-opaque images in the first place..
        let has_opacities = self.pixmap.data().iter().any(|pixel| pixel.a != 255);

        let (x_advance, y_advance) = x_y_advances(&transform);

        let encoded = EncodedImage {
            pixmap: self.pixmap.clone(),
            extends: (self.x_extend, self.y_extend),
            quality: self.quality,
            has_opacities,
            transform,
            x_advance,
            y_advance,
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
    pub pixmap: Arc<Pixmap>,
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
#[derive(Debug)]
pub struct LinearKind {
    distance: f32,
    y2_minus_y1: f32,
    x2_minus_x1: f32,
}

/// Computed properties of a radial gradient.
#[derive(Debug)]
pub struct RadialKind {
    c1: (f32, f32),
    r0: f32,
    r1: f32,
    cone_like: bool,
}

impl RadialKind {
    fn pos_inner(&self, pos: Point) -> Option<f32> {
        // The values for a radial gradient can be calculated for any t as follow:
        // Let x(t) = (x_1 - x_0)*t + x_0 (since x_0 is always 0, this shortens to x_1 * t)
        // Let y(t) = (y_1 - y_0)*t + y_0 (since y_0 is always 0, this shortens to y_1 * t)
        // Let r(t) = (r_1 - r_0)*t + r_0
        // Given a pixel at a position (x_2, y_2), we need to find the largest t such that
        // (x_2 - x(t))^2 + (y - y_(t))^2 = r_t()^2, i.e. the circle with the interpolated
        // radius and center position needs to intersect the pixel we are processing.
        //
        // You can reformulate this problem to a quadratic equation (TODO: add derivation. Since
        // I'm not sure if that code will stay the same after performance optimizations I haven't
        // written this down yet), to which we then simply need to find the solutions.

        let r0 = self.r0;
        let dx = self.c1.0;
        let dy = self.c1.1;
        let dr = self.r1 - self.r0;

        let px = pos.x as f32;
        let py = pos.y as f32;

        let a = dx * dx + dy * dy - dr * dr;
        let b = -2.0 * (px * dx + py * dy + r0 * dr);
        let c = px * px + py * py - r0 * r0;

        let discriminant = b * b - 4.0 * a * c;

        // No solution available.
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_d = discriminant.sqrt();
        let t1 = (-b - sqrt_d) / (2.0 * a);
        let t2 = (-b + sqrt_d) / (2.0 * a);

        let max = t1.max(t2);
        let min = t1.min(t2);

        // We only want values for `t` where the interpolated radius is actually positive.
        if self.r0 + dr * max < 0.0 {
            if self.r0 + dr * min < 0.0 {
                None
            } else {
                Some(min)
            }
        } else {
            Some(max)
        }
    }
}

/// Computed properties of a sweep gradient.
#[derive(Debug)]
pub struct SweepKind;

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
    /// The values that should be used for clamping when applying the extend.
    pub clamp_range: (f32, f32),
}

/// An encoded ange between two color stops.
#[derive(Debug, Clone)]
pub struct GradientRange {
    /// The start value of the range.
    pub x0: f32,
    /// The end value of the range.
    pub x1: f32,
    /// The start color of the range.
    pub c0: PremulColor,
    /// The interpolation factors of the range.
    pub factors_f32: [f32; 4],
}

/// Sampling positions in a gradient.
pub trait GradientLike {
    /// Given a position, return the position on the gradient range.
    fn cur_pos(&self, pos: Point) -> f32;
    /// Whether the gradient is possibly not defined over the whole domain of points.
    fn has_undefined(&self) -> bool;
    /// Whether the current position is defined in the gradient. If `has_undefined` returns `false`,
    /// this will return false for all possible points.
    fn is_defined(&self, pos: Point) -> bool;
}

impl GradientLike for SweepKind {
    fn cur_pos(&self, pos: Point) -> f32 {
        // The position in a sweep gradient is simply determined by its angle from the origin.
        let angle = (-pos.y as f32).atan2(pos.x as f32);

        if angle >= 0.0 {
            angle
        } else {
            angle + 2.0 * PI
        }
    }

    fn has_undefined(&self) -> bool {
        false
    }

    fn is_defined(&self, _: Point) -> bool {
        true
    }
}

impl GradientLike for LinearKind {
    fn cur_pos(&self, pos: Point) -> f32 {
        // The position of a point relative to a linear gradient is determined by its distance
        // to the normal vector. See `encode_into` for more information.
        (pos.x as f32 * self.y2_minus_y1 - pos.y as f32 * self.x2_minus_x1) / self.distance
    }

    fn has_undefined(&self) -> bool {
        false
    }

    fn is_defined(&self, _: Point) -> bool {
        true
    }
}

impl GradientLike for RadialKind {
    fn cur_pos(&self, pos: Point) -> f32 {
        self.pos_inner(pos).unwrap_or(0.0)
    }

    fn has_undefined(&self) -> bool {
        self.cone_like
    }

    fn is_defined(&self, pos: Point) -> bool {
        self.pos_inner(pos).is_some()
    }
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
                core::mem::swap(&mut rect.x0, &mut rect.x1);
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
