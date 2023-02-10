// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello::kurbo::{PathEl, Point};

/// Helper trait for converting cubic splines to paths.
pub trait SplineToPath {
    fn get(&self, index: usize) -> Point;
    fn len(&self) -> usize;

    fn to_path(&self, is_closed: bool, path: &mut Vec<PathEl>) -> Option<()> {
        use PathEl::*;
        path.push(MoveTo(self.get(0)));
        let n_vertices = self.len() / 3;
        let mut add_element = |from_vertex, to_vertex| {
            let from_index = 3 * from_vertex;
            let to_index = 3 * to_vertex;
            let p0: Point = self.get(from_index);
            let p1: Point = self.get(to_index);
            let mut c0: Point = self.get(from_index + 2);
            c0.x += p0.x;
            c0.y += p0.y;
            let mut c1: Point = self.get(to_index + 1);
            c1.x += p1.x;
            c1.y += p1.y;
            if c0 == p0 && c1 == p1 {
                path.push(LineTo(p1));
            } else {
                path.push(CurveTo(c0, c1, p1));
            }
        };
        for i in 1..n_vertices {
            add_element(i - 1, i);
        }
        if is_closed && n_vertices != 0 {
            add_element(n_vertices - 1, 0);
            path.push(ClosePath);
        }
        Some(())
    }
}

/// Converts a static spline to a path.
impl SplineToPath for &'_ [Point] {
    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn get(&self, index: usize) -> Point {
        self[index]
    }
}

/// Produces a path by lerping between two sets of points.
impl SplineToPath for (&'_ [Point], &'_ [Point], f64) {
    fn len(&self) -> usize {
        self.0.len().min(self.1.len())
    }

    fn get(&self, index: usize) -> Point {
        self.0[index].lerp(self.1[index], self.2)
    }
}
