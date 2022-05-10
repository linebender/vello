// Copyright 2022 The piet-gpu authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Also licensed under MIT license, at your choice.

use bytemuck::{Pod, Zeroable};
use core::borrow::Borrow;
use core::hash::{Hash, Hasher};

/// Two dimensional point.
#[derive(Copy, Clone, PartialEq, PartialOrd, Default, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}

impl Point {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn transform(&self, affine: &Affine) -> Self {
        Self {
            x: self.x * affine.xx + self.y * affine.yx + affine.dx,
            y: self.y * affine.yy + self.y * affine.xy + affine.dy,
        }
    }
}

impl From<[f32; 2]> for Point {
    fn from(value: [f32; 2]) -> Self {
        Self::new(value[0], value[1])
    }
}

impl From<(f32, f32)> for Point {
    fn from(value: (f32, f32)) -> Self {
        Self::new(value.0, value.1)
    }
}

/// Affine transformation matrix.
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Affine {
    pub xx: f32,
    pub yx: f32,
    pub xy: f32,
    pub yy: f32,
    pub dx: f32,
    pub dy: f32,
}

impl Affine {
    pub const IDENTITY: Self = Self {
        xx: 1.0,
        yx: 0.0,
        xy: 0.0,
        yy: 1.0,
        dx: 0.0,
        dy: 0.0,
    };

    pub const fn new(elements: &[f32; 6]) -> Self {
        Self {
            xx: elements[0],
            yx: elements[1],
            xy: elements[2],
            yy: elements[3],
            dx: elements[4],
            dy: elements[5],
        }
    }

    /// Creates a new affine transform representing the specified scale along the
    /// x and y axes.
    pub fn scale(x: f32, y: f32) -> Self {
        Self::new(&[x, 0., 0., y, 0., 0.])
    }

    /// Creates a new affine transform representing the specified translation.
    pub fn translate(x: f32, y: f32) -> Self {
        Self::new(&[1., 0., 0., 1., x, y])
    }

    /// Creates a new affine transform representing a counter-clockwise
    /// rotation for the specified angle in radians.
    pub fn rotate(th: f32) -> Self {
        let (s, c) = th.sin_cos();
        Self::new(&[c, s, -s, c, 0., 0.])
    }

    /// Creates a new skew transform
    pub fn skew(x: f32, y: f32) -> Self {
        Self::new(&[1., x.tan(), y.tan(), 1., 0., 0.])
    }

    pub fn around_center(&self, x: f32, y: f32) -> Self {
        Self::translate(x, y) * *self * Self::translate(-x, -y)
    }

    /// Transforms the specified point.
    pub fn transform_point(&self, point: Point) -> Point {
        Point {
            x: point.x * self.xx + point.y * self.yx + self.dx,
            y: point.y * self.yy + point.y * self.xy + self.dy,
        }
    }

    /// Compute the determinant of this transform.
    pub fn determinant(self) -> f32 {
        self.xx * self.yy - self.yx * self.xy
    }

    /// Compute the inverse transform.
    ///
    /// Produces NaN values when the determinant is zero.
    pub fn inverse(self) -> Self {
        let inv_det = self.determinant().recip();
        Self::new(&[
            inv_det * self.yy,
            -inv_det * self.yx,
            -inv_det * self.xy,
            inv_det * self.xx,
            inv_det * (self.xy * self.dy - self.yy * self.dx),
            inv_det * (self.yx * self.dx - self.xx * self.dy),
        ])
    }
}

impl Default for Affine {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl std::ops::Mul for Affine {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(&[
            self.xx * other.xx + self.xy * other.yx,
            self.yx * other.xx + self.yy * other.yx,
            self.xx * other.xy + self.xy * other.yy,
            self.yx * other.xy + self.yy * other.yy,
            self.xx * other.dx + self.xy * other.dy + self.dx,
            self.yx * other.dx + self.yy * other.dy + self.dy,
        ])
    }
}

/// Axis-aligned rectangle represented as minimum and maximum points.
#[derive(Copy, Clone, Default, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Rect {
    pub min: Point,
    pub max: Point,
}

impl Rect {
    /// Creates a new rectangle that encloses the specified collection of
    /// points.
    pub fn from_points<I>(points: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<Point>,
    {
        let mut rect = Self {
            min: Point::new(f32::MAX, f32::MAX),
            max: Point::new(f32::MIN, f32::MIN),
        };
        let mut count = 0;
        for point in points {
            rect.add(*point.borrow());
            count += 1;
        }
        if count != 0 {
            rect
        } else {
            Self::default()
        }
    }

    /// Returns the width of the rectangle.
    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    /// Returns the height of the rectangle.
    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }

    /// Extends the rectangle to include the specified point.
    pub fn add(&mut self, point: Point) {
        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);
        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
    }

    /// Returns a new rectangle that encloses the minimum and maximum points
    /// of this rectangle after applying the specified transform to each.
    pub fn transform(&self, affine: &Affine) -> Self {
        Self::from_points([self.min.transform(affine), self.max.transform(affine)])
    }
}
