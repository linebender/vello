// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flattening filled and stroked paths.

use flatten::stroke::LoweredPath;
use vello_api::kurbo;
use vello_api::kurbo::{Affine, BezPath, Line, Stroke};

/// The flattening tolerance
const TOL: f64 = 0.25;

/// A point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// The x coordinate of the point.
    pub x: f32,
    /// The y coordinate of the point.
    pub y: f32,
}

impl Point {
    /// Create a new point.
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl std::ops::Add for Point {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::Sub for Point {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Mul<f32> for Point {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

/// A flat line.
#[derive(Clone, Copy, Debug)]
pub struct FlatLine {
    /// The start point of the line.
    pub p0: Point,
    /// The end point of the line.
    pub p1: Point,
}

impl FlatLine {
    /// Create a new flat line.
    pub fn new(p0: Point, p1: Point) -> Self {
        Self { p0, p1 }
    }
}

/// Flatten a filled bezier path into line segments.
pub fn fill(path: &BezPath, affine: Affine, line_buf: &mut Vec<FlatLine>) {
    line_buf.clear();
    let mut start = kurbo::Point::default();
    let mut p0 = kurbo::Point::default();
    let iter = path.iter().map(|el| affine * el);

    let mut closed = false;

    kurbo::flatten(iter, TOL, |el| match el {
        kurbo::PathEl::MoveTo(p) => {
            if !closed && p0 != start {
                close_path(start, p0, line_buf);
            }

            closed = false;
            start = p;
            p0 = p;
        }
        kurbo::PathEl::LineTo(p) => {
            let pt0 = Point::new(p0.x as f32, p0.y as f32);
            let pt1 = Point::new(p.x as f32, p.y as f32);
            line_buf.push(FlatLine::new(pt0, pt1));
            p0 = p;
        }
        kurbo::PathEl::QuadTo(_, _) => unreachable!(),
        kurbo::PathEl::CurveTo(_, _, _) => unreachable!(),
        kurbo::PathEl::ClosePath => {
            closed = true;

            close_path(start, p0, line_buf);
        }
    });

    if !closed {
        close_path(start, p0, line_buf);
    }
}

/// Flatten a stroked bezier path into line segments.
pub fn stroke(path: &BezPath, style: &Stroke, affine: Affine, line_buf: &mut Vec<FlatLine>) {
    line_buf.clear();

    // TODO: Temporary hack to ensure that strokes are scaled properly by the transform.
    let tolerance = TOL / affine.as_coeffs()[0].abs().max(affine.as_coeffs()[3].abs());

    let lines: LoweredPath<Line> = flatten::stroke::stroke_undashed(path.iter(), style, tolerance);
    for line in &lines.path {
        let scaled_p0 = affine * line.p0;
        let scaled_p1 = affine * line.p1;
        let p0 = Point::new(scaled_p0.x as f32, scaled_p0.y as f32);
        let p1 = Point::new(scaled_p1.x as f32, scaled_p1.y as f32);
        line_buf.push(FlatLine::new(p0, p1));
    }
}

fn close_path(start: kurbo::Point, p0: kurbo::Point, line_buf: &mut Vec<FlatLine>) {
    let pt0 = Point::new(p0.x as f32, p0.y as f32);
    let pt1 = Point::new(start.x as f32, start.y as f32);

    if pt0 != pt1 {
        line_buf.push(FlatLine::new(pt0, pt1));
    }
}
