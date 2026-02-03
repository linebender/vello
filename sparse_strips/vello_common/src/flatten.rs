// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flattening filled and stroked paths.

use crate::flatten_simd::{Callback, LinePathEl};
use crate::kurbo::{self, Affine, PathEl, Stroke, StrokeCtx, StrokeOpts};
use alloc::vec::Vec;
use fearless_simd::{Level, Simd, dispatch};
use log::warn;

pub use crate::flatten_simd::FlattenCtx;

// The current tolerance is set to 0.25. Since `sqrt` doesn't work in const contexts, we instead
// hardcode the squared tolerance and derive the others from that.
pub(crate) const SQRT_TOL: f64 = 0.5;
pub(crate) const TOL: f64 = SQRT_TOL * SQRT_TOL;
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
    dispatch!(level, simd => fill_impl(simd, path, affine, line_buf, ctx));
}

/// Flatten a filled bezier path into line segments.
#[inline(always)]
pub fn fill_impl<S: Simd>(
    simd: S,
    path: impl IntoIterator<Item = PathEl>,
    affine: Affine,
    line_buf: &mut Vec<Line>,
    flatten_ctx: &mut FlattenCtx,
) {
    line_buf.clear();
    let iter = path.into_iter().map(
        #[inline(always)]
        |el| affine * el,
    );

    let mut lb = FlattenerCallback {
        line_buf,
        start: Point::ZERO,
        p0: Point::ZERO,
        is_nan: false,
    };

    crate::flatten_simd::flatten(simd, iter, &mut lb, flatten_ctx);

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
    stroke_ctx: &mut StrokeCtx,
) {
    // TODO: Temporary hack to ensure that strokes are scaled properly by the transform.
    let tolerance = TOL
        / affine.as_coeffs()[0]
            .abs()
            .max(affine.as_coeffs()[3].abs())
            .max(1.);

    expand_stroke(path, style, tolerance, stroke_ctx);
    fill(level, stroke_ctx.output(), affine, line_buf, flatten_ctx);
}

/// Expand a stroked path to a filled path.
pub fn expand_stroke(
    path: impl IntoIterator<Item = PathEl>,
    style: &Stroke,
    tolerance: f64,
    stroke_ctx: &mut StrokeCtx,
) {
    kurbo::stroke_with(path, style, &StrokeOpts::default(), tolerance, stroke_ctx);
}

struct FlattenerCallback<'a> {
    line_buf: &'a mut Vec<Line>,
    start: Point,
    p0: Point,
    is_nan: bool,
}

impl Callback for FlattenerCallback<'_> {
    #[inline(always)]
    fn callback(&mut self, el: LinePathEl) {
        match el {
            LinePathEl::MoveTo(p) => {
                self.is_nan |= p.is_nan();

                self.start = Point::new(p.x as f32, p.y as f32);
                self.p0 = self.start;
            }
            LinePathEl::LineTo(p) => {
                self.is_nan |= p.is_nan();

                let p = Point::new(p.x as f32, p.y as f32);
                self.line_buf.push(Line::new(self.p0, p));
                self.p0 = p;
            }
        }
    }
}
