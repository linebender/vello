// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Types for flattening curves.

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
