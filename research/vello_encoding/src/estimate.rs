// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This utility provides conservative size estimation for buffer allocations backing
//! GPU bump memory. This estimate relies on heuristics and naturally overestimates.

use super::{BumpAllocatorMemory, BumpAllocators, Transform};
use peniko::kurbo::{Cap, Join, PathEl, Point, Stroke, Vec2};

const RSQRT_OF_TOL: f64 = 2.2360679775; // tol = 0.2

#[derive(Clone, Default)]
pub struct BumpEstimator {
    // TODO: support binning
    // TODO: support ptcl
    // TODO: support tile

    // NOTE: The segment count estimation could use further refinement, particularly to handle
    // viewport clipping and rotation applied to fragments during append. We can produce a more
    // optimal result under scale and rotation if we track more data for each shape during insertion
    // and defer the final tally to resolve-time (in which we could evaluate the estimates using
    // precisely transformed coordinates). For now we apply a fudge factor of sqrt(2) and inflate
    // the number of tile crossing (a near~diagonal line orientation would result in worst case for
    // the number of intersected tiles) to account for this.
    //
    // Accounting for viewport clipping (for the right and bottom edges of the viewport) is simply
    // impossible at insertion time as the render target dimensions are unknown. We could
    // potentially account for clipping (including clip shapes/layers) by tracking bounding boxes
    // during insertion and resolving all clips at tally time (e.g. one could come up with a
    // heuristic for scaling the counts based on the proportions of a clipped bbox area).
    //
    // Since we currently don't account for clipping, this will always overshoot when clips are
    // present and when the bounding box of a shape is partially or wholly outside the viewport.
    segments: u32,
    lines: LineSoup,
}

impl BumpEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Combine the counts of this estimator with `other` after applying an optional `transform`.
    pub fn append(&mut self, other: &Self, transform: Option<&Transform>) {
        let scale = transform_scale(transform);
        self.segments += (other.segments as f64 * scale).ceil() as u32;
        self.lines.add(&other.lines, scale);
    }

    pub fn count_path(
        &mut self,
        path: impl Iterator<Item = PathEl>,
        t: &Transform,
        stroke: Option<&Stroke>,
    ) {
        let mut caps = 1;
        let mut joins: u32 = 0;
        let mut lineto_lines = 0;
        let mut fill_close_lines = 1;
        let mut curve_lines = 0;
        let mut curve_count = 0;
        let mut segments = 0;

        // Track the path state to correctly count empty paths and close joins.
        let mut first_pt = None;
        let mut last_pt = None;
        let scale = transform_scale(Some(t));
        let scaled_width = stroke.map(|s| s.width * scale).unwrap_or(0.);
        let offset_fudge = scaled_width.sqrt().max(1.);
        for el in path {
            match el {
                PathEl::MoveTo(p0) => {
                    first_pt = Some(p0);
                    let Some(point) = last_pt else {
                        continue;
                    };
                    caps += 1;
                    joins = joins.saturating_sub(1);
                    fill_close_lines += 1;
                    segments += count_segments_for_line(first_pt.unwrap(), point, t);
                    last_pt = None;
                }
                PathEl::ClosePath => {
                    if let Some(last_pt) = last_pt {
                        joins += 1;
                        lineto_lines += 1;
                        segments += count_segments_for_line(first_pt.unwrap(), last_pt, t);
                    }
                    last_pt = first_pt;
                }
                PathEl::LineTo(p0) => {
                    last_pt = Some(p0);
                    joins += 1;
                    lineto_lines += 1;
                    segments += count_segments_for_line(first_pt.unwrap(), last_pt.unwrap(), t);
                }
                PathEl::QuadTo(p1, p2) => {
                    let Some(p0) = last_pt.or(first_pt) else {
                        continue;
                    };
                    last_pt = Some(p2);

                    let p0 = p0.to_vec2();
                    let p1 = p1.to_vec2();
                    let p2 = p2.to_vec2();
                    let lines = offset_fudge * wang::quadratic(RSQRT_OF_TOL, p0, p1, p2, t);

                    curve_lines += lines.ceil() as u32;
                    curve_count += 1;
                    joins += 1;

                    let segs = offset_fudge * count_segments_for_quadratic(p0, p1, p2, t);
                    segments += segs.ceil().max(lines.ceil()) as u32;
                }
                PathEl::CurveTo(p1, p2, p3) => {
                    let Some(p0) = last_pt.or(first_pt) else {
                        continue;
                    };
                    last_pt = Some(p3);

                    let p0 = p0.to_vec2();
                    let p1 = p1.to_vec2();
                    let p2 = p2.to_vec2();
                    let p3 = p3.to_vec2();
                    let lines = offset_fudge * wang::cubic(RSQRT_OF_TOL, p0, p1, p2, p3, t);

                    curve_lines += lines.ceil() as u32;
                    curve_count += 1;
                    joins += 1;
                    let segs = count_segments_for_cubic(p0, p1, p2, p3, t);
                    segments += segs.ceil().max(lines.ceil()) as u32;
                }
            }
        }

        let Some(style) = stroke else {
            self.lines.linetos += lineto_lines + fill_close_lines;
            self.lines.curves += curve_lines;
            self.lines.curve_count += curve_count;
            self.segments += segments;

            // Account for the implicit close
            if let (Some(first_pt), Some(last_pt)) = (first_pt, last_pt) {
                self.segments += count_segments_for_line(first_pt, last_pt, t);
            }
            return;
        };

        // For strokes, double-count the lines to estimate offset curves.
        self.lines.linetos += 2 * lineto_lines;
        self.lines.curves += 2 * curve_lines;
        self.lines.curve_count += 2 * curve_count;
        self.segments += 2 * segments;

        self.count_stroke_caps(style.start_cap, scaled_width, caps);
        self.count_stroke_caps(style.end_cap, scaled_width, caps);
        self.count_stroke_joins(style.join, scaled_width, style.miter_limit, joins);
    }

    /// Produce the final total, applying an optional transform to all content.
    pub fn tally(&self, transform: Option<&Transform>) -> BumpAllocatorMemory {
        let scale = transform_scale(transform);

        // The post-flatten line estimate.
        let lines = self.lines.tally(scale);

        // The estimate for tile crossings for lines. Here we ensure that there are at least as many
        // segments as there are lines, in case `segments` was underestimated at small scales.
        let n_segments = ((self.segments as f64 * scale).ceil() as u32).max(lines);

        let bump = BumpAllocators {
            failed: 0,
            // TODO: we can provide a tighter bound here but for now we
            // assume that binning must be bounded by the segment count.
            binning: n_segments,
            ptcl: 0,
            tile: 0,
            blend: 0,
            seg_counts: n_segments,
            segments: n_segments,
            lines,
        };
        bump.memory()
    }

    fn count_stroke_caps(&mut self, style: Cap, scaled_width: f64, count: u32) {
        match style {
            Cap::Butt => {
                self.lines.linetos += count;
                self.segments += count_segments_for_line_length(scaled_width) * count;
            }
            Cap::Square => {
                self.lines.linetos += 3 * count;
                self.segments += count_segments_for_line_length(scaled_width) * count;
                self.segments += 2 * count_segments_for_line_length(0.5 * scaled_width) * count;
            }
            Cap::Round => {
                let (arc_lines, line_len) = estimate_arc_lines(scaled_width);
                self.lines.curves += count * arc_lines;
                self.lines.curve_count += 1;
                self.segments += count * arc_lines * count_segments_for_line_length(line_len);
            }
        }
    }

    fn count_stroke_joins(&mut self, style: Join, scaled_width: f64, miter_limit: f64, count: u32) {
        match style {
            Join::Bevel => {
                self.lines.linetos += count;
                self.segments += count_segments_for_line_length(scaled_width) * count;
            }
            Join::Miter => {
                let max_miter_len = scaled_width * miter_limit;
                self.lines.linetos += 2 * count;
                self.segments += 2 * count * count_segments_for_line_length(max_miter_len);
            }
            Join::Round => {
                let (arc_lines, line_len) = estimate_arc_lines(scaled_width);
                self.lines.curves += count * arc_lines;
                self.lines.curve_count += 1;
                self.segments += count * arc_lines * count_segments_for_line_length(line_len);
            }
        }

        // Count inner join lines
        self.lines.linetos += count;
        self.segments += count_segments_for_line_length(scaled_width) * count;
    }
}

fn estimate_arc_lines(scaled_stroke_width: f64) -> (u32, f64) {
    // These constants need to be kept consistent with the definitions in `flatten_arc` in
    // flatten.wgsl.
    // TODO: It would be better if these definitions were shared/configurable. For example an
    // option is for all tolerances to be parameters to the estimator as well as the GPU pipelines
    // (the latter could be in the form of a config uniform) which would help to keep them in
    // sync.
    const MIN_THETA: f64 = 1e-6;
    const TOL: f64 = 0.25;
    let radius = TOL.max(scaled_stroke_width * 0.5);
    let theta = (2. * (1. - TOL / radius).acos()).max(MIN_THETA);
    let arc_lines = ((std::f64::consts::FRAC_PI_2 / theta).ceil() as u32).max(2);
    (arc_lines, 2. * theta.sin() * radius)
}

#[derive(Clone, Default)]
struct LineSoup {
    // Explicit lines (such as linetos and non-round stroke caps/joins) and Bezier curves
    // get tracked separately to ensure that explicit lines remain scale invariant.
    linetos: u32,
    curves: u32,

    // Curve count is simply used to ensure a minimum number of lines get counted for each curve
    // at very small scales to reduce the chances of an under-estimate.
    curve_count: u32,
}

impl LineSoup {
    fn tally(&self, scale: f64) -> u32 {
        let curves = self
            .scaled_curve_line_count(scale)
            .max(5 * self.curve_count);

        self.linetos + curves
    }

    fn scaled_curve_line_count(&self, scale: f64) -> u32 {
        (self.curves as f64 * scale.sqrt()).ceil() as u32
    }

    fn add(&mut self, other: &Self, scale: f64) {
        self.linetos += other.linetos;
        self.curves += other.scaled_curve_line_count(scale);
        self.curve_count += other.curve_count;
    }
}

// TODO: The 32-bit Vec2 definition from cpu_shaders/util.rs could come in handy here.
fn transform(t: &Transform, v: Vec2) -> Vec2 {
    Vec2::new(
        t.matrix[0] as f64 * v.x + t.matrix[2] as f64 * v.y,
        t.matrix[1] as f64 * v.x + t.matrix[3] as f64 * v.y,
    )
}

fn transform_scale(t: Option<&Transform>) -> f64 {
    match t {
        Some(t) => {
            let m = t.matrix;
            let v1x = m[0] as f64 + m[3] as f64;
            let v2x = m[0] as f64 - m[3] as f64;
            let v1y = m[1] as f64 - m[2] as f64;
            let v2y = m[1] as f64 + m[2] as f64;
            (v1x * v1x + v1y * v1y).sqrt() + (v2x * v2x + v2y * v2y).sqrt()
        }
        None => 1.,
    }
}

fn approx_arc_length_cubic(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2) -> f64 {
    let chord_len = (p3 - p0).length();
    // Length of the control polygon
    let poly_len = (p1 - p0).length() + (p2 - p1).length() + (p3 - p2).length();
    0.5 * (chord_len + poly_len)
}

fn count_segments_for_cubic(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: &Transform) -> f64 {
    let p0 = transform(t, p0);
    let p1 = transform(t, p1);
    let p2 = transform(t, p2);
    let p3 = transform(t, p3);
    (approx_arc_length_cubic(p0, p1, p2, p3) * 0.0625 * std::f64::consts::SQRT_2).ceil()
}

fn count_segments_for_quadratic(p0: Vec2, p1: Vec2, p2: Vec2, t: &Transform) -> f64 {
    count_segments_for_cubic(p0, p1.lerp(p0, 0.333333), p1.lerp(p2, 0.333333), p2, t)
}

// Estimate tile crossings for a line with known endpoints.
fn count_segments_for_line(p0: Point, p1: Point, t: &Transform) -> u32 {
    let dxdy = p0 - p1;
    let dxdy = transform(t, dxdy);
    let segments = (dxdy.x.abs().ceil() * 0.0625).ceil() + (dxdy.y.abs().ceil() * 0.0625).ceil();
    (segments as u32).max(1)
}

// Estimate tile crossings for a line with a known length.
fn count_segments_for_line_length(scaled_width: f64) -> u32 {
    // scale the tile count by sqrt(2) to allow some slack for diagonal lines.
    // TODO: Would "2" be a better factor?
    ((scaled_width * 0.0625 * std::f64::consts::SQRT_2).ceil() as u32).max(1)
}

/// Wang's Formula (as described in Pyramid Algorithms by Ron Goldman, 2003, Chapter 5, Section
/// 5.6.3 on Bezier Approximation) is a fast method for computing a lower bound on the number of
/// recursive subdivisions required to approximate a Bezier curve within a certain tolerance. The
/// formula for a Bezier curve of degree `n`, control points `p[0]...p[n]`, and number of levels of
/// subdivision `l`, and flattening tolerance `tol` is defined as follows:
///
/// ```ignore
///     m = max([length(p[k+2] - 2 * p[k+1] + p[k]) for (0 <= k <= n-2)])
///     l >= log_4((n * (n - 1) * m) / (8 * tol))
/// ```
///
/// For recursive subdivisions that split a curve into 2 segments at each level, the minimum number
/// of segments is given by 2^l. From the formula above it follows that:
///
/// ```ignore
///       segments >= 2^l >= 2^log_4(x)                      (1)
///     segments^2 >= 2^(2*log_4(x)) >= 4^log_4(x)           (2)
///     segments^2 >= x
///       segments >= sqrt((n * (n - 1) * m) / (8 * tol))    (3)
/// ```
///
/// Wang's formula computes an error bound on recursive subdivision based on the second derivative
/// which tends to result in a suboptimal estimate when the curvature within the curve has a lot of
/// variation. This is expected to frequently overshoot the flattening formula used in vello, which
/// is closer to optimal (vello uses a method based on a numerical approximation of the integral
/// over the continuous change in the number of flattened segments, with an error expressed in terms
/// of curvature and infinitesimal arclength).
mod wang {
    use super::{Transform, Vec2, transform};

    // The curve degree term sqrt(n * (n - 1) / 8) specialized for cubics:
    //
    //    sqrt(3 * (3 - 1) / 8)
    //
    const SQRT_OF_DEGREE_TERM_CUBIC: f64 = 0.86602540378;

    // The curve degree term sqrt(n * (n - 1) / 8) specialized for quadratics:
    //
    //    sqrt(2 * (2 - 1) / 8)
    //
    const SQRT_OF_DEGREE_TERM_QUAD: f64 = 0.5;

    pub(crate) fn quadratic(rsqrt_of_tol: f64, p0: Vec2, p1: Vec2, p2: Vec2, t: &Transform) -> f64 {
        let v = -2. * p1 + p0 + p2;
        let v = transform(t, v); // transform is distributive
        let m = v.length();
        (SQRT_OF_DEGREE_TERM_QUAD * m.sqrt() * rsqrt_of_tol).ceil()
    }

    pub(crate) fn cubic(
        rsqrt_of_tol: f64,
        p0: Vec2,
        p1: Vec2,
        p2: Vec2,
        p3: Vec2,
        t: &Transform,
    ) -> f64 {
        let v1 = -2. * p1 + p0 + p2;
        let v2 = -2. * p2 + p1 + p3;
        let v1 = transform(t, v1);
        let v2 = transform(t, v2);
        let m = v1.length().max(v2.length());
        (SQRT_OF_DEGREE_TERM_CUBIC * m.sqrt() * rsqrt_of_tol).ceil()
    }
}
