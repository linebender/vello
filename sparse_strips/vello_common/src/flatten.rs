// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flattening filled and stroked paths.

use crate::flatten_simd::Callback;
use crate::kurbo::{self, Affine, BezPath, PathEl, Stroke, StrokeOpts};
use alloc::vec::Vec;
use fearless_simd::{Level, Simd, simd_dispatch};
use log::warn;

pub use crate::flatten_simd::FlattenCtx;

/// The flattening tolerance.
pub(crate) const TOL: f64 = 0.25;
pub(crate) const TOL_2: f64 = TOL * TOL;

/// A point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// The x coordinate of the point.
    pub x: f32,
    /// The y coordinate of the point.
    pub y: f32,
}

impl Point {
    /// The point `(0, 0)`.
    pub const ZERO: Self = Self::new(0., 0.);

    /// Create a new point.
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl core::ops::Add for Point {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl core::ops::Sub for Point {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl core::ops::Mul<f32> for Point {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

/// A line.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    /// The start point of the line.
    pub p0: Point,
    /// The end point of the line.
    pub p1: Point,
}

impl Line {
    /// Create a new line.
    pub fn new(p0: Point, p1: Point) -> Self {
        Self { p0, p1 }
    }
}

/// Flatten a filled bezier path into line segments.
pub fn fill(
    level: Level,
    path: impl IntoIterator<Item = PathEl>,
    affine: Affine,
    line_buf: &mut Vec<Line>,
    ctx: &mut FlattenCtx,
) {
    fill_dispatch(level, path, affine, line_buf, ctx);
}

simd_dispatch!(fn fill_dispatch(
    level,
    path: impl IntoIterator<Item = PathEl>, 
    affine: Affine,
    line_buf: &mut Vec<Line>, 
    ctx: &mut FlattenCtx
) = fill_impl);

/// Flatten a filled bezier path into line segments.
pub fn fill_impl<S: Simd>(
    simd: S,
    path: impl IntoIterator<Item = PathEl>,
    affine: Affine,
    line_buf: &mut Vec<Line>,
    flatten_ctx: &mut FlattenCtx,
) {
    line_buf.clear();
    let iter = path.into_iter().map(|el| affine * el);

    let mut lb = FlattenerCallback {
        line_buf,
        start: kurbo::Point::default(),
        p0: kurbo::Point::default(),
        is_nan: false,
        closed: false,
    };

    crate::flatten_simd::flatten(simd, iter, TOL, &mut lb, flatten_ctx);

    if !lb.closed {
        close_path(lb.start, lb.p0, lb.line_buf);
    }

    // A path that contains NaN is ill-defined, so ignore it.
    if lb.is_nan {
        warn!("A path contains NaN, ignoring it.");

        line_buf.clear();
    }
}
/// Flatten a stroked bezier path into line segments.
pub fn stroke(
    level: Level,
    path: impl IntoIterator<Item = PathEl>,
    style: &Stroke,
    affine: Affine,
    line_buf: &mut Vec<Line>,
    flatten_ctx: &mut FlattenCtx,
) {
    // TODO: Temporary hack to ensure that strokes are scaled properly by the transform.
    let tolerance = TOL
        / affine.as_coeffs()[0]
            .abs()
            .max(affine.as_coeffs()[3].abs())
            .max(1.);

    let expanded = expand_stroke(path, style, tolerance);
    fill(level, &expanded, affine, line_buf, flatten_ctx);
}

/// Expand a stroked path to a filled path.
pub fn expand_stroke(
    path: impl IntoIterator<Item = PathEl>,
    style: &Stroke,
    tolerance: f64,
) -> BezPath {
    kurbo::stroke(path, style, &StrokeOpts::default(), tolerance)
}

struct FlattenerCallback<'a> {
    line_buf: &'a mut Vec<Line>,
    start: kurbo::Point,
    p0: kurbo::Point,
    is_nan: bool,
    closed: bool,
}

impl Callback for FlattenerCallback<'_> {
    #[inline(always)]
    fn callback(&mut self, el: PathEl) {
        self.is_nan |= el.is_nan();

        match el {
            kurbo::PathEl::MoveTo(p) => {
                if !self.closed && self.p0 != self.start {
                    close_path(self.start, self.p0, self.line_buf);
                }

                self.closed = false;
                self.start = p;
                self.p0 = p;
            }
            kurbo::PathEl::LineTo(p) => {
                let pt0 = Point::new(self.p0.x as f32, self.p0.y as f32);
                let pt1 = Point::new(p.x as f32, p.y as f32);
                self.line_buf.push(Line::new(pt0, pt1));
                self.p0 = p;
            }
            el @ (kurbo::PathEl::QuadTo(_, _) | kurbo::PathEl::CurveTo(_, _, _)) => {
                unreachable!("Path has been flattened, so shouldn't contain {el:?}.")
            }
            kurbo::PathEl::ClosePath => {
                self.closed = true;

                close_path(self.start, self.p0, self.line_buf);
            }
        }
    }
}

fn close_path(start: kurbo::Point, p0: kurbo::Point, line_buf: &mut Vec<Line>) {
    let pt0 = Point::new(p0.x as f32, p0.y as f32);
    let pt1 = Point::new(start.x as f32, start.y as f32);

    if pt0 != pt1 {
        line_buf.push(Line::new(pt0, pt1));
    }
}
