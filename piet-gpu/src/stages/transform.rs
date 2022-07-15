// Copyright 2021 The piet-gpu authors.
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

//! The transform stage of the element processing pipeline.

use bytemuck::{Pod, Zeroable};

use piet::kurbo::Affine;

/// An affine transform.
// This is equivalent to the version in piet-gpu-types, but the bytemuck
// representation will likely be faster.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Transform {
    pub mat: [f32; 4],
    pub translate: [f32; 2],
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        mat: [1.0, 0.0, 0.0, 1.0],
        translate: [0.0, 0.0],
    };

    pub fn from_kurbo(a: Affine) -> Transform {
        let c = a.as_coeffs();
        Transform {
            mat: [c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32],
            translate: [c[4] as f32, c[5] as f32],
        }
    }

    pub fn to_kurbo(self) -> Affine {
        Affine::new([
            self.mat[0] as f64,
            self.mat[1] as f64,
            self.mat[2] as f64,
            self.mat[3] as f64,
            self.translate[0] as f64,
            self.translate[1] as f64,
        ])
    }
}
