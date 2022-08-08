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
use super::ExtendMode;
use crate::geometry::Point;
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// Offset and color of a transition point in a gradient.
#[derive(Copy, Clone, PartialOrd, Default, Debug)]
pub struct GradientStop {
    pub offset: f32,
    pub color: Color,
}

impl Hash for GradientStop {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.offset.to_bits().hash(state);
        self.color.hash(state);
    }
}

// Override PartialEq to use to_bits for the offset to match with the Hash impl
impl std::cmp::PartialEq for GradientStop {
    fn eq(&self, other: &Self) -> bool {
        self.offset.to_bits() == other.offset.to_bits() && self.color == other.color
    }
}

impl std::cmp::Eq for GradientStop {}

/// Collection of gradient stops.
pub type GradientStops = SmallVec<[GradientStop; 4]>;

/// Definition of a gradient that transitions between two or more colors along
/// a line.
#[derive(Clone, Debug)]
pub struct LinearGradient {
    pub start: Point,
    pub end: Point,
    pub stops: GradientStops,
    pub extend: ExtendMode,
}

/// Definition of a gradient that transitions between two or more colors that
/// radiate from an origin.
#[derive(Clone, Debug)]
pub struct RadialGradient {
    pub center0: Point,
    pub radius0: f32,
    pub center1: Point,
    pub radius1: f32,
    pub stops: GradientStops,
    pub extend: ExtendMode,
}

/// Definition gradient that transitions between two or more colors that rotate
/// around a center point.
#[derive(Clone, Debug)]
pub struct SweepGradient {
    pub center: Point,
    pub start_angle: f32,
    pub end_angle: f32,
    pub stops: GradientStops,
    pub extend: ExtendMode,
}
