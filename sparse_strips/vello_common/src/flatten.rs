// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Flattening filled and stroked paths.

use crate::kurbo::{self, Affine, BezPath, Stroke, StrokeOpts};
use alloc::vec::Vec;

/// The flattening tolerance.
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
pub fn fill(path: &BezPath, affine: Affine, line_buf: &mut Vec<Line>) {
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
            line_buf.push(Line::new(pt0, pt1));
            p0 = p;
        }
        el @ (kurbo::PathEl::QuadTo(_, _) | kurbo::PathEl::CurveTo(_, _, _)) => {
            unreachable!("Path has been flattened, so shouldn't contain {el:?}.")
        }
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
pub fn stroke(path: &BezPath, style: &Stroke, affine: Affine, line_buf: &mut Vec<Line>) {
    // TODO: Temporary hack to ensure that strokes are scaled properly by the transform.
    let tolerance = TOL / affine.as_coeffs()[0].abs().max(affine.as_coeffs()[3].abs());

    let expanded = kurbo::stroke(path.iter(), style, &StrokeOpts::default(), tolerance);
    fill(&expanded, affine, line_buf);
}

fn close_path(start: kurbo::Point, p0: kurbo::Point, line_buf: &mut Vec<Line>) {
    let pt0 = Point::new(p0.x as f32, p0.y as f32);
    let pt1 = Point::new(start.x as f32, start.y as f32);

    if pt0 != pt1 {
        line_buf.push(Line::new(pt0, pt1));
    }
}
