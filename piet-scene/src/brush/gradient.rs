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

use super::color::Color;
use crate::geometry::Point;
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

#[derive(Copy, Clone, PartialOrd, Default, Debug)]
pub struct Stop {
    pub offset: f32,
    pub color: Color,
}

impl Hash for Stop {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.offset.to_bits().hash(state);
        self.color.hash(state);
    }
}

// Override PartialEq to use to_bits for the offset to match with the Hash impl
impl std::cmp::PartialEq for Stop {
    fn eq(&self, other: &Self) -> bool {
        self.offset.to_bits() == other.offset.to_bits() && self.color == other.color
    }
}

impl std::cmp::Eq for Stop {}

pub type StopVec = SmallVec<[Stop; 4]>;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Extend {
    Pad,
    Repeat,
    Reflect,
}

#[derive(Clone, Debug)]
pub struct LinearGradient {
    pub start: Point,
    pub end: Point,
    pub stops: StopVec,
    pub extend: Extend,
}

#[derive(Clone, Debug)]
pub struct RadialGradient {
    pub center0: Point,
    pub radius0: f32,
    pub center1: Point,
    pub radius1: f32,
    pub stops: StopVec,
    pub extend: Extend,
}

#[derive(Clone, Debug)]
pub struct SweepGradient {
    pub center: Point,
    pub start_angle: f32,
    pub end_angle: f32,
    pub stops: StopVec,
    pub extend: Extend,
}
