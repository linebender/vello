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

use super::geometry::{Point, Rect};

/// Action of a path element.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PathVerb {
    MoveTo,
    LineTo,
    QuadTo,
    CurveTo,
    Close,
}

/// Element of a path represented by a verb and its associated points.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PathElement {
    MoveTo(Point),
    LineTo(Point),
    QuadTo(Point, Point),
    CurveTo(Point, Point, Point),
    Close,
}

impl PathElement {
    /// Returns the verb that describes the action of the path element.
    pub fn verb(&self) -> PathVerb {
        match self {
            Self::MoveTo(..) => PathVerb::MoveTo,
            Self::LineTo(..) => PathVerb::LineTo,
            Self::QuadTo(..) => PathVerb::QuadTo,
            Self::CurveTo(..) => PathVerb::CurveTo,
            Self::Close => PathVerb::Close,
        }
    }
}

impl Rect {
    pub fn elements(&self) -> impl Iterator<Item = PathElement> + Clone {
        let elements = [
            PathElement::MoveTo((self.min.x, self.min.y).into()),
            PathElement::LineTo((self.max.x, self.min.y).into()),
            PathElement::LineTo((self.max.x, self.max.y).into()),
            PathElement::LineTo((self.min.x, self.max.y).into()),
            PathElement::Close,
        ];
        (0..5).map(move |i| elements[i])
    }
}
