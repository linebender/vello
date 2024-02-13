// Copyright 2024 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This utility provides conservative size estimation for buffer allocations backing
//! GPU bump memory. This estimate relies on heuristics and naturally overestimates.

use super::{BufferSize, BumpAllocatorMemory, Transform};
use peniko::kurbo::{Cap, Join, PathEl, Stroke, Vec2};

const RSQRT_OF_TOL: f64 = 2.2360679775; // tol = 0.2

#[derive(Clone, Default)]
pub struct BumpEstimator {
    // TODO: support binning
    // TODO: support ptcl
    // TODO: support tile
    // TODO: support segment counts
    // TODO: support segments
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
        self.lines.add(&other.lines, transform_scale(transform));
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

        // Track the path state to correctly count empty paths and close joins.
        let mut first_pt = None;
        let mut last_pt = None;
        for el in path {
            match el {
                PathEl::MoveTo(p0) => {
                    first_pt = Some(p0);
                    if last_pt.is_none() {
                        continue;
                    }
                    caps += 1;
                    joins = joins.saturating_sub(1);
                    last_pt = None;
                    fill_close_lines += 1;
                }
                PathEl::ClosePath => {
                    if last_pt.is_some() {
                        joins += 1;
                        lineto_lines += 1;
                    }
                    last_pt = first_pt;
                }
                PathEl::LineTo(p0) => {
                    last_pt = Some(p0);
                    joins += 1;
                    lineto_lines += 1;
                }
                PathEl::QuadTo(p1, p2) => {
                    let Some(p0) = last_pt.or(first_pt) else {
                        continue;
                    };
                    curve_count += 1;
                    curve_lines +=
                        wang::quadratic(RSQRT_OF_TOL, p0.to_vec2(), p1.to_vec2(), p2.to_vec2(), t);
                    last_pt = Some(p2);
                    joins += 1;
                }
                PathEl::CurveTo(p1, p2, p3) => {
                    let Some(p0) = last_pt.or(first_pt) else {
                        continue;
                    };
                    curve_count += 1;
                    curve_lines += wang::cubic(
                        RSQRT_OF_TOL,
                        p0.to_vec2(),
                        p1.to_vec2(),
                        p2.to_vec2(),
                        p3.to_vec2(),
                        t,
                    );
                    last_pt = Some(p3);
                    joins += 1;
                }
            }
        }
        let Some(style) = stroke else {
            self.lines.linetos += lineto_lines + fill_close_lines;
            self.lines.curves += curve_lines;
            self.lines.curve_count += curve_count;
            return;
        };

        // For strokes, double-count the lines to estimate offset curves.
        self.lines.linetos += 2 * lineto_lines;
        self.lines.curves += 2 * curve_lines;
        self.lines.curve_count += 2 * curve_count;

        let round_scale = transform_scale(Some(t));
        let width = style.width as f32;
        self.count_stroke_caps(style.start_cap, width, caps, round_scale);
        self.count_stroke_caps(style.end_cap, width, caps, round_scale);
        self.count_stroke_joins(style.join, width, joins, round_scale);
    }

    /// Produce the final total, applying an optional transform to all content.
    pub fn tally(&self, transform: Option<&Transform>) -> BumpAllocatorMemory {
        let scale = transform_scale(transform);
        let binning = BufferSize::new(0);
        let ptcl = BufferSize::new(0);
        let tile = BufferSize::new(0);
        let seg_counts = BufferSize::new(0);
        let segments = BufferSize::new(0);
        let lines = BufferSize::new(self.lines.tally(scale));
        BumpAllocatorMemory {
            total: binning.size_in_bytes()
                + ptcl.size_in_bytes()
                + tile.size_in_bytes()
                + seg_counts.size_in_bytes()
                + lines.size_in_bytes(),
            binning,
            ptcl,
            tile,
            seg_counts,
            segments,
            lines,
        }
    }

    fn count_stroke_caps(&mut self, style: Cap, width: f32, count: u32, scale: f32) {
        match style {
            Cap::Butt => self.lines.linetos += count,
            Cap::Square => self.lines.linetos += 3 * count,
            Cap::Round => {
                self.lines.curves += count * estimate_arc_lines(width as f32, scale);
                self.lines.curve_count += 1;
            }
        }
    }

    fn count_stroke_joins(&mut self, style: Join, width: f32, count: u32, scale: f32) {
        match style {
            Join::Bevel => self.lines.linetos += count,
            Join::Miter => self.lines.linetos += 2 * count,
            Join::Round => {
                self.lines.curves += count * estimate_arc_lines(width as f32, scale);
                self.lines.curve_count += 1;
            }
        }
    }
}

fn estimate_arc_lines(stroke_width: f32, scale: f32) -> u32 {
    // These constants need to be kept consistent with the definitions in `flatten_arc` in
    // flatten.wgsl.
    const MIN_THETA: f32 = 1e-4;
    const TOL: f32 = 0.1;
    let radius = TOL.max(scale * stroke_width * 0.5);
    let theta = (2. * (1. - TOL / radius).acos()).max(MIN_THETA);
    let arc_lines = ((std::f32::consts::FRAC_PI_2 / theta).ceil() as u32).max(1);
    arc_lines
}

#[derive(Clone, Default)]
struct LineSoup {
    // Explicit lines (such as linetos and non-round stroke caps/joins) and Bezier curves
    // get tracked separately to ensure that explicit lines remain scale invariant.
    linetos: u32,
    curves: u32,

    // Curve count is simply used to ensure a minimum number of lines get counted for each curve
    // at very small scales to reduce the chance of under-allocating.
    curve_count: u32,
}

impl LineSoup {
    fn tally(&self, scale: f32) -> u32 {
        let curves = self
            .scaled_curve_line_count(scale)
            .max(5 * self.curve_count);

        self.linetos + curves
    }

    fn scaled_curve_line_count(&self, scale: f32) -> u32 {
        (self.curves as f32 * scale.sqrt()).ceil() as u32
    }

    fn add(&mut self, other: &LineSoup, scale: f32) {
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

fn transform_scale(t: Option<&Transform>) -> f32 {
    match t {
        Some(t) => {
            let m = t.matrix;
            let v1x = m[0] + m[3];
            let v2x = m[0] - m[3];
            let v1y = m[1] - m[2];
            let v2y = m[1] + m[2];
            (v1x * v1x + v1y * v1y).sqrt() + (v2x * v2x + v2y * v2y).sqrt()
        }
        None => 1.,
    }
}

/// Wang's Formula (as described in Pyramid Algorithms by Ron Goldman, 2003, Chapter 5, Section
/// 5.6.3 on Bezier Approximation) is a fast method for computing a lower bound on the number of
/// recursive subdivisions required to approximate a Bezier curve within a certain tolerance. The
/// formula for a Bezier curve of degree `n`, control points p[0]...p[n], and number of levels of
/// subdivision `l`, and flattening tolerance `tol` is defined as follows:
///
///     m = max([length(p[k+2] - 2 * p[k+1] + p[k]) for (0 <= k <= n-2)])
///     l >= log_4((n * (n - 1) * m) / (8 * tol))
///
/// For recursive subdivisions that split a curve into 2 segments at each level, the minimum number
/// of segments is given by 2^l. From the formula above it follows that:
///
///                 segments >= 2^l >= 2^log_4(x)                      (1)
///               segments^2 >= 2^(2*log_4(x)) >= 4^log_4(x)           (2)
///               segments^2 >= x
///                 segments >= sqrt((n * (n - 1) * m) / (8 * tol))    (3)
///
/// Wang's formula computes an error bound on recursive subdivion based on the second derivative of
/// the curve and results in a suboptimal estimate when the curvature within the curve has a lot of
/// variation. This is expected to frequently overshoot the flattening formula used in vello (based
/// on a numerical approximation of an integral over the continuous change in the number of flattened
/// segments, with an error expressed in terms of curvature and infinitesimal arclength), which is
/// closer to optimal.
mod wang {
    use super::*;

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

    pub fn quadratic(rsqrt_of_tol: f64, p0: Vec2, p1: Vec2, p2: Vec2, t: &Transform) -> u32 {
        let v = -2. * p1 + p0 + p2;
        let v = transform(t, v); // transform is distributive
        let m = v.length();
        (SQRT_OF_DEGREE_TERM_QUAD * m.sqrt() * rsqrt_of_tol).ceil() as u32
    }

    pub fn cubic(rsqrt_of_tol: f64, p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: &Transform) -> u32 {
        let v1 = -2. * p1 + p0 + p2;
        let v2 = -2. * p2 + p1 + p3;
        let v1 = transform(t, v1);
        let v2 = transform(t, v2);
        let m = v1.length().max(v2.length()) as f64;
        (SQRT_OF_DEGREE_TERM_CUBIC * m.sqrt() * rsqrt_of_tol).ceil() as u32
    }
}
