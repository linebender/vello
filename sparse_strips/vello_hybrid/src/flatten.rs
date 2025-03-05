// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utilities for flattening

use flatten::stroke::LoweredPath;
use vello_api::peniko::kurbo::{self, Affine, BezPath, Line, Point, Stroke};

use crate::tiling::FlatLine;

/// The flattening tolerance
const TOL: f64 = 0.25;

pub(crate) fn fill(path: &BezPath, affine: Affine, line_buf: &mut Vec<FlatLine>) {
    line_buf.clear();
    let mut start = Point::default();
    let mut p0 = Point::default();
    let iter = path.iter().map(|el| affine * el);
    kurbo::flatten(iter, TOL, |el| match el {
        kurbo::PathEl::MoveTo(p) => {
            start = p;
            p0 = p;
        }
        kurbo::PathEl::LineTo(p) => {
            let pt0 = [p0.x as f32, p0.y as f32];
            let pt1 = [p.x as f32, p.y as f32];
            line_buf.push(FlatLine::new(pt0, pt1));
            p0 = p;
        }
        kurbo::PathEl::QuadTo(_, _) => unreachable!(),
        kurbo::PathEl::CurveTo(_, _, _) => unreachable!(),
        kurbo::PathEl::ClosePath => {
            let pt0 = [p0.x as f32, p0.y as f32];
            let pt1 = [start.x as f32, start.y as f32];
            if pt0 != pt1 {
                line_buf.push(FlatLine::new(pt0, pt1));
            }
        }
    });
}

pub(crate) fn stroke(path: &BezPath, style: &Stroke, affine: Affine, line_buf: &mut Vec<FlatLine>) {
    line_buf.clear();
    let iter = path.iter().map(|el| affine * el);
    let lines: LoweredPath<Line> = flatten::stroke::stroke_undashed(iter, style, TOL);
    for line in &lines.path {
        let p0 = [line.p0.x as f32, line.p0.y as f32];
        let p1 = [line.p1.x as f32, line.p1.y as f32];
        line_buf.push(FlatLine::new(p0, p1));
    }
}
