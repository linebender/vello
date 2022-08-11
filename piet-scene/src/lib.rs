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

mod brush;
mod geometry;
mod path;
mod resource;
mod scene;

pub mod glyph;

pub use brush::*;
pub use geometry::*;
pub use path::*;
pub use resource::*;
pub use scene::*;

/// Implement conversions to and from Kurbo types when the `kurbo` feature is
/// enabled.
#[cfg(feature = "kurbo")]
mod kurbo_conv {
    use super::geometry::{Affine, Point, Rect};
    use super::path::PathElement;

    impl Point {
        /// Creates a new point from the equivalent kurbo type.
        pub fn from_kurbo(point: kurbo::Point) -> Self {
            Self::new(point.x as f32, point.y as f32)
        }
    }

    impl From<Point> for kurbo::Point {
        fn from(p: Point) -> kurbo::Point {
            Self::new(p.x as f64, p.y as f64)
        }
    }

    impl Affine {
        /// Creates a new affine transformation from the equivalent kurbo type.
        pub fn from_kurbo(affine: kurbo::Affine) -> Self {
            let c = affine.as_coeffs();
            Self {
                xx: c[0] as f32,
                yx: c[1] as f32,
                xy: c[2] as f32,
                yy: c[3] as f32,
                dx: c[4] as f32,
                dy: c[5] as f32,
            }
        }
    }

    impl From<Affine> for kurbo::Affine {
        fn from(a: Affine) -> Self {
            Self::new([
                a.xx as f64,
                a.yx as f64,
                a.yx as f64,
                a.yy as f64,
                a.dx as f64,
                a.dy as f64,
            ])
        }
    }

    impl Rect {
        /// Creates a new rectangle from the equivalent kurbo type.
        pub fn from_kurbo(rect: kurbo::Rect) -> Self {
            Self {
                min: Point::new(rect.x0 as f32, rect.y0 as f32),
                max: Point::new(rect.x1 as f32, rect.y1 as f32),
            }
        }
    }

    impl From<Rect> for kurbo::Rect {
        fn from(r: Rect) -> Self {
            Self {
                x0: r.min.x as f64,
                y0: r.min.y as f64,
                x1: r.max.x as f64,
                y1: r.max.y as f64,
            }
        }
    }

    impl PathElement {
        /// Creates a new path element from the equivalent kurbo type.
        pub fn from_kurbo(el: kurbo::PathEl) -> Self {
            use kurbo::PathEl::*;
            match el {
                MoveTo(p0) => Self::MoveTo(Point::from_kurbo(p0)),
                LineTo(p0) => Self::LineTo(Point::from_kurbo(p0)),
                QuadTo(p0, p1) => Self::QuadTo(Point::from_kurbo(p0), Point::from_kurbo(p1)),
                CurveTo(p0, p1, p2) => Self::CurveTo(
                    Point::from_kurbo(p0),
                    Point::from_kurbo(p1),
                    Point::from_kurbo(p2),
                ),
                ClosePath => Self::Close,
            }
        }
    }

    impl From<PathElement> for kurbo::PathEl {
        fn from(e: PathElement) -> Self {
            use PathElement::*;
            match e {
                MoveTo(p0) => Self::MoveTo(p0.into()),
                LineTo(p0) => Self::LineTo(p0.into()),
                QuadTo(p0, p1) => Self::QuadTo(p0.into(), p1.into()),
                CurveTo(p0, p1, p2) => Self::CurveTo(p0.into(), p1.into(), p2.into()),
                Close => Self::ClosePath,
            }
        }
    }
}
