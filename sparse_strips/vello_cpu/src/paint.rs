// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Paints for drawing shapes.

use crate::fine::COLOR_COMPONENTS;
use std::borrow::Cow;
use std::f32::consts::PI;
use std::iter;
use vello_common::color::palette::css::BLACK;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Affine, Point, Vec2};
use vello_common::paint::{IndexedPaint, Paint};
use vello_common::peniko::{Extend, GradientKind};

const DEGENERATE_THRESHOLD: f32 = 1.0e-6;

/// A kind of paint used for drawing shapes.
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
    pub stops: Vec<Stop>,
    /// A transformation to apply to the gradient.
    pub transform: Affine,
    /// The extend of the gradient.
    pub extend: Extend,
    /// An additional opacity to apply to all gradient stops.
    pub opacity: f32,
}

impl Gradient {
    /// Encode the gradient into a paint.
    pub fn encode_into(&self, paints: &mut Vec<EncodedPaint>) -> Paint {
        // First make sure that the gradient is valid and not degenerate.
        if let Some(paint) = self.validate() {
            return paint;
        }

        let opacity = self.opacity.clamp(0.0, 1.0);

        let mut has_opacities =
            self.stops.iter().any(|s| s.color.components[3] != 1.0) || opacity != 1.0;
        let pad = self.extend == Extend::Pad;

        let mut stops = Cow::Borrowed(&self.stops);
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
                    std::mem::swap(&mut p0, &mut p1);

                    stops = Cow::Owned(
                        stops
                            .iter()
                            .rev()
                            .map(|s| Stop {
                                offset: 1.0 - s.offset,
                                color: s.color,
                            })
                            .collect::<Vec<_>>(),
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

        let ranges = encode_stops(&stops, clamp_range.0, clamp_range.1, pad, opacity);

        // This represents the transform that needs to be applied to the starting point of a
        // command before starting with the rendering.
        // First we need to account for a potential offset of the gradient (x_offset/y_offset), then
        // we account for the fact that we sample in the center of a pixel and not in the corner by
        // adding 0.5.
        // Finally, we need to apply the _inverse_ transform to the point so that we can account
        // for the transform on the gradient.
        let transform = Affine::translate((x_offset as f64 + 0.5, y_offset as f64 + 0.5))
            * self.transform.inverse();

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
        let (x_advance, y_advance) = {
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
        };

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

    fn validate(&self) -> Option<Paint> {
        let black = Some(BLACK.into());

        // Gradients need at least two stops.
        if self.stops.is_empty() {
            return black;
        }

        let first = Some(self.stops[0].color.into());

        if self.stops.len() == 1 {
            return first;
        }

        // First stop must be at offset 0.0 and last offset must be at 1.0.
        if self.stops[0].offset != 0.0 || self.stops[self.stops.len() - 1].offset != 1.0 {
            return black;
        }

        for stops in self.stops.windows(2) {
            let f = stops[0];
            let n = stops[1];

            // Offsets must be between 0 and 1.
            if f.offset > 1.0 || f.offset < 0.0 {
                return black;
            }

            // Stops must be sorted by ascending offset.
            if f.offset >= n.offset {
                return black;
            }
        }

        let degenerate_point = |p1: &Point, p2: &Point| {
            (p1.x - p2.x).abs() as f32 <= DEGENERATE_THRESHOLD
                && (p1.y - p2.y).abs() as f32 <= DEGENERATE_THRESHOLD
        };

        let degenerate_val = |v1: f32, v2: f32| (v2 - v1).abs() <= DEGENERATE_THRESHOLD;

        match &self.kind {
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
                // Angles must be between 0 and 360.
                if *start_angle < 0.0
                    || *start_angle > 360.0
                    || *end_angle < 0.0
                    || *end_angle > 360.0
                {
                    return first;
                }

                // The end angle must be larger than the start angle.
                if degenerate_val(*start_angle, *end_angle) {
                    return first;
                }

                if end_angle <= start_angle {
                    return first;
                }
            }
        }

        None
    }
}

/// A color stop.
#[derive(Debug, Clone, Copy)]
pub struct Stop {
    /// The normalized offset of the stop. Must be between 0.0 and 1.0.
    pub offset: f32,
    /// The color of the stop.
    pub color: AlphaColor<Srgb>,
}

/// Extend the stops so that we can treat a repeated gradient like a reflected gradient.
fn apply_reflect(stops: &[Stop]) -> Vec<Stop> {
    let first_half = stops.iter().map(|s| Stop {
        offset: s.offset / 2.0,
        color: s.color,
    });

    let second_half = stops.iter().rev().map(|s| Stop {
        offset: 0.5 + (1.0 - s.offset) / 2.0,
        color: s.color,
    });

    first_half.chain(second_half).collect::<Vec<_>>()
}

/// Encode all stops into a sequence of ranges.
fn encode_stops(
    stops: &[Stop],
    start: f32,
    end: f32,
    pad: bool,
    opacity: f32,
) -> Vec<GradientRange> {
    let create_range = |left_stop: &Stop, right_stop: &Stop| {
        let x0 = start + (end - start) * left_stop.offset;
        let x1 = start + (end - start) * right_stop.offset;
        let c0 = left_stop
            .color
            .multiply_alpha(opacity)
            .premultiply()
            .to_rgba8()
            .to_u8_array();
        let c1 = right_stop
            .color
            .multiply_alpha(opacity)
            .premultiply()
            .to_rgba8()
            .to_u8_array();

        // Given two positions x0 and x1 as well as two corresponding colors c0 and c1,
        // the delta that needs to be applied to c0 to calculate the color of x between x0 and x1
        // is calculated by c0 + ((x - x0) / (x1 - x0)) * (c1 - c0).
        // We can precompute the (c1 - c0)/(x1 - x0) part for each color component.

        // We call this method with two same stops for `left_range` and `right_range`, so make
        // sure we don't actually end up with a 0 here.
        let x1_minus_x0 = (x1 - x0).max(0.0000001);
        let mut factors = [0.0; 4];

        for i in 0..COLOR_COMPONENTS {
            let c1_minus_c0 = c1[i] as f32 - c0[i] as f32;
            factors[i] = c1_minus_c0 / x1_minus_x0;
        }

        GradientRange {
            x0,
            x1,
            c0,
            factors,
        }
    };

    let stop_ranges = stops.windows(2).map(|s| {
        let left_stop = &s[0];
        let right_stop = &s[1];

        create_range(left_stop, right_stop)
    });

    if pad {
        // We handle padding by inserting dummy stops in the beginning and end with a very big
        // range.
        let left_range = iter::once({
            let first_stop = stops.first().unwrap();
            let mut encoded_range = create_range(first_stop, first_stop);
            encoded_range.x0 = f32::MIN;

            encoded_range
        });

        let right_range = iter::once({
            let last_stop = stops.last().unwrap();

            let mut encoded_range = create_range(last_stop, last_stop);
            encoded_range.x1 = f32::MAX;

            encoded_range
        });

        left_range.chain(stop_ranges.chain(right_range)).collect()
    } else {
        stop_ranges.collect()
    }
}

/// An encoded paint.
#[derive(Debug)]
pub enum EncodedPaint {
    /// An encoded gradient.
    Gradient(EncodedGradient),
}

impl From<EncodedGradient> for EncodedPaint {
    fn from(value: EncodedGradient) -> Self {
        Self::Gradient(value)
    }
}

#[derive(Debug)]
pub(crate) struct LinearKind {
    pub(crate) distance: f32,
    pub(crate) y2_minus_y1: f32,
    pub(crate) x2_minus_x1: f32,
}

#[derive(Debug)]
pub(crate) struct RadialKind {
    pub(crate) c1: (f32, f32),
    pub(crate) r0: f32,
    pub(crate) r1: f32,
    pub(crate) cone_like: bool,
}

impl RadialKind {
    fn pos_inner(&self, pos: &Point) -> Option<f32> {
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

#[derive(Debug)]
pub(crate) struct SweepKind;

#[derive(Debug)]
pub(crate) enum EncodedKind {
    Linear(LinearKind),
    Radial(RadialKind),
    Sweep(SweepKind),
}

/// An encoded gradient.
#[derive(Debug)]
pub struct EncodedGradient {
    /// The underlying kind of gradient.
    pub(crate) kind: EncodedKind,
    /// A transform that needs to be applied to the position of the first processed pixel.
    pub(crate) transform: Affine,
    /// How much to advance into the x/y direction for one step in the x direction.
    pub(crate) x_advance: Vec2,
    /// How much to advance into the x/y direction for one step in the y direction.
    pub(crate) y_advance: Vec2,
    /// The color ranges of the gradient.
    pub(crate) ranges: Vec<GradientRange>,
    /// Whether the gradient should be padded.
    pub(crate) pad: bool,
    /// Whether the gradient requires `source_over` compositing.
    pub(crate) has_opacities: bool,
    /// The values that should be used for clamping when applying the extend.
    pub(crate) clamp_range: (f32, f32),
}

#[derive(Debug, Clone)]
pub(crate) struct GradientRange {
    /// The start value of the range.
    pub(crate) x0: f32,
    /// The end value of the range.
    pub(crate) x1: f32,
    /// The start color of the range.
    pub(crate) c0: [u8; 4],
    /// The interpolation factors of the range.
    pub(crate) factors: [f32; 4],
}

pub(crate) trait GradientLike {
    fn cur_pos(&self, pos: &Point) -> f32;
    fn has_undefined(&self) -> bool;
    fn is_defined(&self, pos: &Point) -> bool;
}

impl GradientLike for SweepKind {
    fn cur_pos(&self, pos: &Point) -> f32 {
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

    fn is_defined(&self, _: &Point) -> bool {
        true
    }
}

impl GradientLike for LinearKind {
    fn cur_pos(&self, pos: &Point) -> f32 {
        // The position of a point relative to a linear gradient is determined by its distance
        // to the normal vector. See `encode_into` for more information.
        (pos.x as f32 * self.y2_minus_y1 - pos.y as f32 * self.x2_minus_x1) / self.distance
    }

    fn has_undefined(&self) -> bool {
        false
    }

    fn is_defined(&self, _: &Point) -> bool {
        true
    }
}

impl GradientLike for RadialKind {
    fn cur_pos(&self, pos: &Point) -> f32 {
        self.pos_inner(pos).unwrap_or(0.0)
    }

    fn has_undefined(&self) -> bool {
        self.cone_like
    }

    fn is_defined(&self, pos: &Point) -> bool {
        self.pos_inner(pos).is_some()
    }
}

#[cfg(test)]
mod tests {
    use crate::paint::{Gradient, Stop};
    use vello_common::color::palette::css::{BLACK, BLUE, GREEN};
    use vello_common::kurbo::{Affine, Point};
    use vello_common::peniko::Extend;
    use vello_common::peniko::GradientKind;

    #[test]
    fn gradient_missing_stops() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(20.0, 0.0),
            },
            stops: vec![],
            transform: Affine::IDENTITY,
            extend: Extend::Pad,
            opacity: 1.0,
        };

        assert_eq!(gradient.encode_into(&mut buf), BLACK.into());
    }

    #[test]
    fn gradient_one_stop() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(20.0, 0.0),
            },
            stops: vec![Stop {
                offset: 0.0,
                color: GREEN,
            }],
            transform: Affine::IDENTITY,
            extend: Extend::Pad,
            opacity: 1.0,
        };

        // Should return the color of the first stop.
        assert_eq!(gradient.encode_into(&mut buf), GREEN.into());
    }

    #[test]
    fn gradient_not_padded_stops() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(20.0, 0.0),
            },
            stops: vec![
                Stop {
                    offset: 0.0,
                    color: GREEN,
                },
                Stop {
                    offset: 0.5,
                    color: BLUE,
                },
            ],
            transform: Affine::IDENTITY,
            extend: Extend::Pad,
            opacity: 1.0,
        };

        assert_eq!(gradient.encode_into(&mut buf), BLACK.into());
    }

    #[test]
    fn gradient_not_sorted_stops() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(20.0, 0.0),
            },
            stops: vec![
                Stop {
                    offset: 1.0,
                    color: GREEN,
                },
                Stop {
                    offset: 0.0,
                    color: BLUE,
                },
            ],
            transform: Affine::IDENTITY,
            extend: Extend::Pad,
            opacity: 1.0,
        };

        assert_eq!(gradient.encode_into(&mut buf), BLACK.into());
    }

    #[test]
    fn gradient_linear_degenerate() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Linear {
                start: Point::new(0.0, 0.0),
                end: Point::new(0.0, 0.0),
            },
            stops: vec![
                Stop {
                    offset: 0.0,
                    color: GREEN,
                },
                Stop {
                    offset: 1.0,
                    color: BLUE,
                },
            ],
            transform: Affine::IDENTITY,
            extend: Extend::Pad,
            opacity: 1.0,
        };

        assert_eq!(gradient.encode_into(&mut buf), GREEN.into());
    }

    #[test]
    fn gradient_sweep_degenerate() {
        let mut buf = vec![];

        let gradient = Gradient {
            kind: GradientKind::Sweep {
                center: Point::new(0.0, 0.0),
                start_angle: 0.0,
                end_angle: 380.0,
            },
            stops: vec![
                Stop {
                    offset: 0.0,
                    color: GREEN,
                },
                Stop {
                    offset: 1.0,
                    color: BLUE,
                },
            ],
            transform: Affine::IDENTITY,
            extend: Extend::Pad,
            opacity: 1.0,
        };

        assert_eq!(gradient.encode_into(&mut buf), GREEN.into());
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
            stops: vec![
                Stop {
                    offset: 0.0,
                    color: GREEN,
                },
                Stop {
                    offset: 1.0,
                    color: BLUE,
                },
            ],
            transform: Affine::IDENTITY,
            extend: Extend::Pad,
            opacity: 1.0,
        };

        assert_eq!(gradient.encode_into(&mut buf), GREEN.into());
    }
}
