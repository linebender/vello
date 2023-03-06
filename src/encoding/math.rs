// Copyright 2022 Google LLC
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

use std::ops::Mul;

use bytemuck::{Pod, Zeroable};
use peniko::kurbo;

/// Affine transformation matrix.
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct Transform {
    /// 2x2 matrix.
    pub matrix: [f32; 4],
    /// Translation.
    pub translation: [f32; 2],
}

impl Transform {
    /// Identity transform.
    pub const IDENTITY: Self = Self {
        matrix: [1.0, 0.0, 0.0, 1.0],
        translation: [0.0; 2],
    };

    /// Creates a transform from a kurbo affine matrix.
    pub fn from_kurbo(transform: &kurbo::Affine) -> Self {
        let c = transform.as_coeffs().map(|x| x as f32);
        Self {
            matrix: [c[0], c[1], c[2], c[3]],
            translation: [c[4], c[5]],
        }
    }

    /// Converts the transform to a kurbo affine matrix.
    pub fn to_kurbo(&self) -> kurbo::Affine {
        kurbo::Affine::new(
            [
                self.matrix[0],
                self.matrix[1],
                self.matrix[2],
                self.matrix[3],
                self.translation[0],
                self.translation[1],
            ]
            .map(|x| x as f64),
        )
    }
}

impl Mul for Transform {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            matrix: [
                self.matrix[0] * other.matrix[0] + self.matrix[2] * other.matrix[1],
                self.matrix[1] * other.matrix[0] + self.matrix[3] * other.matrix[1],
                self.matrix[0] * other.matrix[2] + self.matrix[2] * other.matrix[3],
                self.matrix[1] * other.matrix[2] + self.matrix[3] * other.matrix[3],
            ],
            translation: [
                self.matrix[0] * other.translation[0]
                    + self.matrix[2] * other.translation[1]
                    + self.translation[0],
                self.matrix[1] * other.translation[0]
                    + self.matrix[3] * other.translation[1]
                    + self.translation[1],
            ],
        }
    }
}

pub fn point_to_f32(point: kurbo::Point) -> [f32; 2] {
    [point.x as f32, point.y as f32]
}
