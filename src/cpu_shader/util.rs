// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility types

use vello_encoding::ConfigUniform;

#[derive(Clone, Copy, Default, Debug, PartialEq)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: Self) -> Self {
        Vec2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: Self) -> Self {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: f32) -> Self {
        Vec2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: f32) -> Self {
        Vec2 {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl std::ops::Mul<Vec2> for f32 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Self::Output {
        rhs * self
    }
}

impl std::ops::Neg for Vec2 {
    type Output = Vec2;

    fn neg(self) -> Self::Output {
        Vec2::new(-self.x, -self.y)
    }
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }

    pub fn dot(self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(self, other: Vec2) -> f32 {
        (self.x * other.y) - (self.y * other.x)
    }

    pub fn length(self) -> f32 {
        self.x.hypot(self.y)
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }

    pub fn from_array(a: [f32; 2]) -> Self {
        Vec2 { x: a[0], y: a[1] }
    }

    pub fn mix(self, other: Vec2, t: f32) -> Self {
        let x = self.x + (other.x - self.x) * t;
        let y = self.y + (other.y - self.y) * t;
        Vec2 { x, y }
    }

    pub fn normalize(self) -> Vec2 {
        self / self.length()
    }

    pub fn atan2(self) -> f32 {
        self.y.atan2(self.x)
    }

    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }

    pub fn min(&self, other: Vec2) -> Vec2 {
        Vec2::new(self.x.min(other.x), self.y.min(other.y))
    }

    pub fn max(&self, other: Vec2) -> Vec2 {
        Vec2::new(self.x.max(other.x), self.y.max(other.y))
    }
}

#[derive(Clone)]
pub struct Transform(pub [f32; 6]);

impl Transform {
    pub fn identity() -> Self {
        Self([1., 0., 0., 1., 0., 0.])
    }

    pub fn apply(&self, p: Vec2) -> Vec2 {
        let z = self.0;
        let x = z[0] * p.x + z[2] * p.y + z[4];
        let y = z[1] * p.x + z[3] * p.y + z[5];
        Vec2 { x, y }
    }

    pub fn read(transform_base: u32, ix: u32, data: &[u32]) -> Transform {
        let mut z = [0.0; 6];
        let base = (transform_base + ix * 6) as usize;
        for i in 0..6 {
            z[i] = f32::from_bits(data[base + i]);
        }
        Transform(z)
    }
}

pub fn span(a: f32, b: f32) -> u32 {
    (a.max(b).ceil() - a.min(b).floor()).max(1.0) as u32
}

const DRAWTAG_NOP: u32 = 0;

/// Read draw tag, guarded by number of draw objects.
///
/// The `ix` argument is allowed to exceed the number of draw objects,
/// in which case a NOP is returned.
pub fn read_draw_tag_from_scene(config: &ConfigUniform, scene: &[u32], ix: u32) -> u32 {
    if ix < config.layout.n_draw_objects {
        let tag_ix = config.layout.draw_tag_base + ix;
        scene[tag_ix as usize]
    } else {
        DRAWTAG_NOP
    }
}

/// The largest floating point value strictly less than 1.
///
/// This value is used to limit the value of b so that its floor is strictly less
/// than 1. That guarantees that floor(a * i + b) == 0 for i == 0, which lands on
/// the correct first tile.
pub const ONE_MINUS_ULP: f32 = 0.99999994;

/// An epsilon to be applied in path numerical robustness.
///
/// When floor(a * (n - 1) + b) does not match the expected value (the width in
/// grid cells minus one), this delta is applied to a to push it in the correct
/// direction. The theory is that a is not off by more than a few ulp, and it's
/// always in the range of 0..1.
pub const ROBUST_EPSILON: f32 = 2e-7;
