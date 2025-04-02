use crate::fine::COLOR_COMPONENTS;
use crate::util::ColorExt;
use std::iter;
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Affine, Point};
use vello_common::peniko::Extend;

// TODO: Validate gradient before encoding it.

#[derive(Debug, Clone)]
pub enum PaintType {
    Solid(AlphaColor<Srgb>),
    LinearGradient(LinearGradient),
    SweepGradient(SweepGradient),
    RadialGradient(RadialGradient),
}

impl From<AlphaColor<Srgb>> for PaintType {
    fn from(value: AlphaColor<Srgb>) -> Self {
        Self::Solid(value)
    }
}

impl From<LinearGradient> for PaintType {
    fn from(value: LinearGradient) -> Self {
        Self::LinearGradient(value)
    }
}

impl From<SweepGradient> for PaintType {
    fn from(value: SweepGradient) -> Self {
        Self::SweepGradient(value)
    }
}

impl From<RadialGradient> for PaintType {
    fn from(value: RadialGradient) -> Self {
        Self::RadialGradient(value)
    }
}

/// A color stop.
#[derive(Debug, Clone, Copy)]
pub struct Stop {
    /// The normalized offset of the stop.
    pub offset: f32,
    /// The color of the stop.
    pub color: AlphaColor<Srgb>,
}

#[derive(Debug, Clone)]
pub struct RadialGradient {
    pub c1: Point,
    pub r1: f32,
    pub c2: Point,
    pub r2: f32,
    pub stops: Vec<Stop>,
    pub transform: Affine,
    pub extend: Extend,
}

impl RadialGradient {
    pub fn encode(mut self) -> EncodedRadialGradient {
        let mut c0 = self.c1;
        let mut c1 = self.c2;
        let mut r0 = self.r1;
        let mut r1 = self.r2;

        let mut stops = self.stops;

        if self.extend == Extend::Reflect {
            c1 += c1 - c0;
            r1 += r1 - r0;

            let first_half = stops.iter().map(|s| Stop {
                offset: s.offset / 2.0,
                color: s.color,
            });

            let second_half = stops.iter().rev().map(|s| Stop {
                offset: 0.5 + (1.0 - s.offset) / 2.0,
                color: s.color,
            });

            let combined = first_half.chain(second_half).collect::<Vec<_>>();
            stops = combined;
        }

        let x_offset = -c0.x as f32;
        let y_offset = -c0.y as f32;

        let c = self.transform.as_coeffs();
        let transform = Affine::translate((x_offset as f64, y_offset as f64))
            * Affine::new([c[0], c[1], c[2], c[3], c[4] - 0.5, c[5] - 0.5]).inverse();

        let pad = self.extend == Extend::Pad;
        let has_opacities = stops.iter().any(|s| s.color.components[3] != 1.0);
        let ranges = encode_stops(&stops, 0.0, 1.0, pad);

        let end_point = c1 - c0;

        EncodedRadialGradient {
            transform,
            c1: (end_point.x as f32, end_point.y as f32),
            r0,
            r1,
            ranges,
            pad,
            has_opacities,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SweepGradient {
    /// The x coordinate of the first point.
    pub center: Point,
    pub start_angle: f32,
    pub end_angle: f32,
    /// The color stops of the linear gradient.
    pub stops: Vec<Stop>,
    pub extend: Extend,
    pub transform: Affine,
}

impl SweepGradient {
    pub fn encode(self) -> EncodedSweepGradient {
        let mut stops = self.stops;
        let start_angle = self.start_angle.to_radians();
        let mut end_angle = self.end_angle.to_radians();

        if self.extend == Extend::Reflect {
            end_angle += end_angle - start_angle;

            let first_half = stops.iter().map(|s| Stop {
                offset: s.offset / 2.0,
                color: s.color,
            });

            let second_half = stops.iter().rev().map(|s| Stop {
                offset: 0.5 + (1.0 - s.offset) / 2.0,
                color: s.color,
            });

            let combined = first_half.chain(second_half).collect::<Vec<_>>();
            stops = combined;
        }

        let pad = self.extend == Extend::Pad;
        let has_opacities = stops.iter().any(|s| s.color.components[3] != 1.0);
        let stops = encode_stops(&stops, start_angle, end_angle, pad);
        let center = self.transform * self.center;
        let c = self.transform.as_coeffs();
        let trans = Affine::new([c[0], c[1], c[2], c[3], center.x - 0.5, center.y - 0.5]).inverse();

        EncodedSweepGradient {
            trans,
            start_angle,
            end_angle,
            ranges: stops,
            pad,
            has_opacities,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearGradient {
    /// The x coordinate of the first point.
    pub p0: Point,
    /// The x coordinate of the second point.
    pub p1: Point,
    /// The color stops of the linear gradient.
    pub stops: Vec<Stop>,
    pub transform: Affine,
    pub extend: Extend,
}

impl LinearGradient {
    pub fn encode(self) -> EncodedLinearGradient {
        // Note that this will not work for transforms with coordinate skewing.
        let mut p0 = self.p0;
        let mut p1 = self.p1;

        let has_opacities = self.stops.iter().any(|s| s.color.components[3] != 1.0);

        let mut stops = if self.p0.x <= self.p1.x {
            self.stops
        } else {
            std::mem::swap(&mut p0, &mut p1);

            self.stops
                .iter()
                .rev()
                .map(|s| Stop {
                    offset: 1.0 - s.offset,
                    color: s.color,
                })
                .collect::<Vec<_>>()
        };

        // Double the length of the iterator, and append stops in reverse order.
        // Then we can treat it the same as repeated gradients.
        if self.extend == Extend::Reflect {
            p1.x += p1.x - p0.x;
            p1.y += p1.y - p0.y;

            let first_half = stops.iter().map(|s| Stop {
                offset: s.offset / 2.0,
                color: s.color,
            });

            let second_half = stops.iter().rev().map(|s| Stop {
                offset: 0.5 + (1.0 - s.offset) / 2.0,
                color: s.color,
            });

            let combined = first_half.chain(second_half).collect::<Vec<_>>();
            stops = combined;
        }

        let x_offset = -p0.x as f32;
        let y_offset = -p0.y as f32;

        let dx = p1.x as f32 + x_offset;
        let dy = p1.y as f32 + y_offset;
        let norm = (-dy, dx);

        let denom = (norm.1 * norm.1 + norm.0 * norm.0).sqrt();
        let fact1 = norm.1;
        let fact2 = norm.0;

        let end = (dx * dx + dy * dy).sqrt();

        let ranges = encode_stops(&stops, 0.0, end, self.extend == Extend::Pad);

        let c = self.transform.as_coeffs();
        let transform = Affine::translate((x_offset as f64, y_offset as f64))
            * Affine::new([c[0], c[1], c[2], c[3], c[4] - 0.5, c[5] - 0.5]).inverse();

        EncodedLinearGradient {
            denom,
            fact1,
            fact2,
            end: (dx * dx + dy * dy).sqrt(),
            ranges,
            pad: self.extend == Extend::Pad,
            has_opacities,
            transform,
        }
    }
}

fn encode_stops(stops: &[Stop], start: f32, end: f32, pad: bool) -> Vec<GradientRange> {
    let create_range = |left_stop: &Stop, right_stop: &Stop| {
        let x0 = start + (end - start) * left_stop.offset;
        let x1 = start + (end - start) * right_stop.offset;
        let c0 = left_stop.color.premultiply().to_rgba8_fast();
        let c1 = right_stop.color.premultiply().to_rgba8_fast();

        let mut im1 = [0.0; 4];
        // Make sure this doesn't end up being 0 for our pad stops.
        let im2 = (x1 - x0).max(0.0000001);
        let mut im3 = [0.0; 4];

        for i in 0..COLOR_COMPONENTS {
            im1[i] = c1[i] as f32 - c0[i] as f32;
            im3[i] = im1[i] / im2;
        }

        GradientRange { x0, x1, c0, im3 }
    };

    let stop_ranges = stops.windows(2).map(|s| {
        let left_stop = &s[0];
        let right_stop = &s[1];

        create_range(left_stop, right_stop)
    });

    if pad {
        let left_range = iter::once({
            let first_stop = &stops[0];
            let mut encoded_range = create_range(first_stop, first_stop);
            encoded_range.x0 = f32::MIN;

            encoded_range
        });

        let right_range = iter::once({
            let last_stop = stops.last().unwrap();

            let mut encoded_range = create_range(&last_stop, &last_stop);
            encoded_range.x1 = f32::MAX;

            encoded_range
        });

        left_range.chain(stop_ranges.chain(right_range)).collect()
    } else {
        stop_ranges.collect()
    }
}

#[derive(Debug)]
pub enum EncodedPaint {
    LinearGradient(EncodedLinearGradient),
    SweepGradient(EncodedSweepGradient),
    RadialGradient(EncodedRadialGradient),
}

impl From<EncodedLinearGradient> for EncodedPaint {
    fn from(value: EncodedLinearGradient) -> Self {
        EncodedPaint::LinearGradient(value)
    }
}

impl From<EncodedSweepGradient> for EncodedPaint {
    fn from(value: EncodedSweepGradient) -> Self {
        EncodedPaint::SweepGradient(value)
    }
}

impl From<EncodedRadialGradient> for EncodedPaint {
    fn from(value: EncodedRadialGradient) -> Self {
        EncodedPaint::RadialGradient(value)
    }
}

#[derive(Debug)]
pub struct EncodedRadialGradient {
    pub transform: Affine,
    pub c1: (f32, f32),
    pub r0: f32,
    pub r1: f32,
    pub ranges: Vec<GradientRange>,
    pub pad: bool,
    pub has_opacities: bool,
}

#[derive(Debug)]
pub struct EncodedSweepGradient {
    pub trans: Affine,
    pub start_angle: f32,
    pub end_angle: f32,
    pub ranges: Vec<GradientRange>,
    pub pad: bool,
    pub has_opacities: bool,
}

#[derive(Debug, Clone)]
pub struct EncodedLinearGradient {
    pub end: f32,
    // Below are the factors that will be used to later on calculate
    // the distance of a strip to the line making up the gradient. Basis of the formula
    // is https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
    // sqrt((y2 - y1)ˆ2 + (x2 - x1)ˆ2)
    pub denom: f32,
    // (y2 - y1)
    pub fact1: f32,
    // (x2 - x1)
    pub fact2: f32,
    pub ranges: Vec<GradientRange>,
    pub pad: bool,
    pub has_opacities: bool,
    pub transform: Affine,
}

/// A color stop.
#[derive(Debug, Clone)]
pub struct GradientRange {
    pub(crate) x0: f32,
    pub(crate) x1: f32,
    pub(crate) c0: [u8; 4],
    pub(crate) im3: [f32; 4],
}
