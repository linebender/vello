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
pub enum Verb {
    MoveTo,
    LineTo,
    QuadTo,
    CurveTo,
    Close,
}

/// Element of a path represented by a verb and its associated points.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Element {
    MoveTo(Point),
    LineTo(Point),
    QuadTo(Point, Point),
    CurveTo(Point, Point, Point),
    Close,
}

impl Element {
    /// Returns the verb that describes the action of the path element.
    pub fn verb(&self) -> Verb {
        match self {
            Self::MoveTo(..) => Verb::MoveTo,
            Self::LineTo(..) => Verb::LineTo,
            Self::QuadTo(..) => Verb::QuadTo,
            Self::CurveTo(..) => Verb::CurveTo,
            Self::Close => Verb::Close,
        }
    }
}

impl Rect {
    pub fn elements(&self) -> impl Iterator<Item = Element> + Clone {
        let elements = [
            Element::MoveTo((self.min.x, self.min.y).into()),
            Element::LineTo((self.max.x, self.min.y).into()),
            Element::LineTo((self.max.x, self.max.y).into()),
            Element::LineTo((self.min.x, self.max.y).into()),
            Element::Close,
        ];
        (0..5).map(move |i| elements[i])
    }
}
