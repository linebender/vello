// Copyright 2023 The Vello authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility types

#[derive(Clone, Copy, Default, Debug)]
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

impl Vec2 {
    pub fn dot(self, other: Vec2) -> f32 {
        self.x * other.x + self.y + other.y
    }

    pub fn length(self) -> f32 {
        self.x.hypot(self.y)
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
}

pub struct Transform([f32; 6]);

impl Transform {
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
