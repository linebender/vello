// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR MIT

/*!
Representations of animated values.
*/

use super::*;

use kurbo::PathEl;

/// Animated affine transformation.
#[derive(Clone, Debug)]
pub struct Transform {
    /// Anchor point.
    pub anchor: Value<Point>,
    /// Translation.
    pub position: Value<Point>,
    /// Rotation angle.
    pub rotation: Value<f32>,
    /// Scale factor.
    pub scale: Value<Vec2>,
    /// Skew factor.
    pub skew: Value<f32>,
    /// Skew angle.
    pub skew_angle: Value<f32>,
}

impl Transform {
    /// Returns true if the transform is fixed.
    pub fn is_fixed(&self) -> bool {
        self.anchor.is_fixed()
            && self.position.is_fixed()
            && self.rotation.is_fixed()
            && self.scale.is_fixed()
            && self.skew.is_fixed()
            && self.skew_angle.is_fixed()
    }

    /// Evaluates the transform at the specified frame.
    pub fn evaluate(&self, frame: f32) -> Affine {
        let anchor = self.anchor.evaluate(frame);
        let position = self.position.evaluate(frame);
        let rotation = self.rotation.evaluate(frame) as f64;
        let scale = self.scale.evaluate(frame);
        let skew = self.skew.evaluate(frame) as f64;
        let skew_angle = self.skew_angle.evaluate(frame) as f64;
        let skew_matrix = if skew != 0.0 {
            const SKEW_LIMIT: f64 = 85.0;
            let skew = -skew.min(SKEW_LIMIT).max(-SKEW_LIMIT);
            let angle = skew_angle.to_radians();
            fn make_skew(x: f64) -> Affine {
                Affine::new([1.0, x.tan(), 0.0, 1.0, 0.0, 0.0])
            }
            Affine::rotate(angle) * make_skew(skew) * Affine::rotate(-angle)
        } else {
            Affine::IDENTITY
        };
        Affine::translate((position.x, position.y))
            * Affine::rotate(rotation.to_radians())
            * skew_matrix
            * Affine::scale_non_uniform(scale.x / 100.0, scale.y / 100.0)
            * Affine::translate((-anchor.x, -anchor.y))
    }

    /// Converts the animated value to its model representation.
    pub fn to_model(self) -> super::Transform {
        if self.is_fixed() {
            super::Transform::Fixed(self.evaluate(0.0))
        } else {
            super::Transform::Animated(self)
        }
    }
}

/// Animated ellipse.
#[derive(Clone, Debug)]
pub struct Ellipse {
    /// True if the ellipse should be drawn in CCW order.
    pub is_ccw: bool,
    /// Position of the ellipse.
    pub position: Value<Point>,
    /// Size of the ellipse.
    pub size: Value<Size>,
}

impl Ellipse {
    pub fn is_fixed(&self) -> bool {
        self.position.is_fixed() && self.size.is_fixed()
    }

    pub fn evaluate(&self, frame: f32) -> kurbo::Ellipse {
        let position = self.position.evaluate(frame);
        let size = self.size.evaluate(frame);
        let radii = (size.width * 0.5, size.height * 0.5);
        kurbo::Ellipse::new(position, radii, 0.0)
    }
}

/// Animated rounded rectangle.
#[derive(Clone, Debug)]
pub struct Rect {
    /// True if the rect should be drawn in CCW order.
    pub is_ccw: bool,
    /// Position of the rectangle.
    pub position: Value<Point>,
    /// Size of the rectangle.
    pub size: Value<Size>,
    /// Radius of the rectangle corners.
    pub corner_radius: Value<f32>,
}

impl Rect {
    /// Returns true if the rectangle is fixed.
    pub fn is_fixed(&self) -> bool {
        self.position.is_fixed() && self.size.is_fixed() && self.corner_radius.is_fixed()
    }

    /// Evaluates the rectangle at the specified frame.
    pub fn evaluate(&self, frame: f32) -> kurbo::RoundedRect {
        let position = self.position.evaluate(frame);
        let size = self.size.evaluate(frame);
        let position = Point::new(
            position.x - size.width * 0.5,
            position.y - size.height * 0.5,
        );
        let radius = self.corner_radius.evaluate(frame);
        kurbo::RoundedRect::new(
            position.x,
            position.y,
            position.x + size.width,
            position.y + size.height,
            radius as f64,
        )
    }
}

/// Animated star or polygon.
#[derive(Clone, Debug)]
pub struct Star {
    pub is_polygon: bool,
    pub direction: f64,
    pub position: Value<Point>,
    pub inner_radius: Value<f32>,
    pub inner_roundness: Value<f32>,
    pub outer_radius: Value<f32>,
    pub outer_roundness: Value<f32>,
    pub rotation: Value<f32>,
    pub points: Value<f32>,
}

impl Star {
    pub fn is_fixed(&self) -> bool {
        self.position.is_fixed()
            && self.inner_radius.is_fixed()
            && self.inner_roundness.is_fixed()
            && self.outer_radius.is_fixed()
            && self.outer_roundness.is_fixed()
            && self.rotation.is_fixed()
            && self.points.is_fixed()
    }
}

/// Animated cubic spline.
#[derive(Clone, Debug)]
pub struct Spline {
    /// True if the spline is closed.
    pub is_closed: bool,
    /// Collection of times.
    pub times: Vec<Time>,
    /// Collection of splines.
    pub values: Vec<Vec<Point>>,
}

impl Spline {
    /// Evalutes the spline at the given frame and emits the elements
    /// to the specified path.
    pub fn evaluate(&self, frame: f32, path: &mut Vec<PathEl>) -> bool {
        use super::SplineToPath as _;
        let Some(([ix0, ix1], t)) = Time::frames_and_weight(&self.times, frame) else {
            return false;
        };
        let (Some(from), Some(to)) = (self.values.get(ix0), self.values.get(ix1)) else {
            return false;
        };
        (from.as_slice(), to.as_slice(), t as f64).to_path(self.is_closed, path);
        true
    }
}

/// Animated repeater effect.
#[derive(Clone, Debug)]
pub struct Repeater {
    /// Number of times elements should be repeated.
    pub copies: Value<f32>,
    /// Offset applied to each element.
    pub offset: Value<f32>,
    /// Anchor point.
    pub anchor_point: Value<Point>,
    /// Translation.
    pub position: Value<Point>,
    /// Rotation in degrees.
    pub rotation: Value<f32>,
    /// Scale.
    pub scale: Value<Vec2>,
    /// Opacity of the first element.
    pub start_opacity: Value<f32>,
    /// Opacity of the last element.
    pub end_opacity: Value<f32>,
}

impl Repeater {
    /// Returns true if the repeater contains no animated properties.
    pub fn is_fixed(&self) -> bool {
        self.copies.is_fixed()
            && self.offset.is_fixed()
            && self.anchor_point.is_fixed()
            && self.position.is_fixed()
            && self.rotation.is_fixed()
            && self.scale.is_fixed()
            && self.start_opacity.is_fixed()
            && self.end_opacity.is_fixed()
    }

    /// Evaluates the repeater at the specified frame.
    pub fn evaluate(&self, frame: f32) -> fixed::Repeater {
        let copies = self.copies.evaluate(frame).round() as u32;
        let offset = self.offset.evaluate(frame);
        let anchor_point = self.anchor_point.evaluate(frame);
        let position = self.position.evaluate(frame);
        let rotation = self.rotation.evaluate(frame);
        let scale = self.scale.evaluate(frame);
        let start_opacity = self.start_opacity.evaluate(frame);
        let end_opacity = self.end_opacity.evaluate(frame);
        fixed::Repeater {
            copies,
            offset,
            anchor_point,
            position,
            rotation,
            scale,
            start_opacity,
            end_opacity,
        }
    }

    /// Converts the animated value to its model representation.
    pub fn to_model(self) -> super::Repeater {
        if self.is_fixed() {
            super::Repeater::Fixed(self.evaluate(0.0))
        } else {
            super::Repeater::Animated(self)
        }
    }
}

/// Animated stroke properties.
#[derive(Clone, Debug)]
pub struct Stroke {
    /// Width of the stroke.
    pub width: Value<f32>,
    /// Join style.
    pub join: peniko::Join,
    /// Limit for miter joins.
    pub miter_limit: Option<f32>,
    /// Cap style.
    pub cap: peniko::Cap,
}

impl Stroke {
    /// Returns true if the stroke is fixed.
    pub fn is_fixed(&self) -> bool {
        self.width.is_fixed()
    }

    /// Evaluates the stroke at the specified frame.
    pub fn evaluate(&self, frame: f32) -> peniko::Stroke {
        let width = self.width.evaluate(frame);
        let mut stroke = peniko::Stroke::new(width)
            .with_caps(self.cap)
            .with_join(self.join);
        if let Some(miter_limit) = self.miter_limit {
            stroke.miter_limit = miter_limit;
        }
        stroke
    }

    /// Converts the animated value to its model representation.
    pub fn to_model(self) -> super::Stroke {
        if self.is_fixed() {
            super::Stroke::Fixed(self.evaluate(0.0))
        } else {
            super::Stroke::Animated(self)
        }
    }
}

/// Animated linear or radial gradient.
#[derive(Clone, Debug)]
pub struct Gradient {
    /// True if the gradient is radial.
    pub is_radial: bool,
    /// Starting point.
    pub start_point: Value<Point>,
    /// Ending point.
    pub end_point: Value<Point>,
    /// Stop offsets and color values.
    pub stops: super::ColorStops,
}

impl Gradient {
    /// Returns true if the value contains no animated properties.
    pub fn is_fixed(&self) -> bool {
        self.start_point.is_fixed() && self.end_point.is_fixed() && self.stops.is_fixed()
    }

    /// Evaluates the animated value at the given frame.
    pub fn evaluate(&self, frame: f32) -> peniko::Brush {
        let start = self.start_point.evaluate(frame);
        let end = self.end_point.evaluate(frame);
        let stops = self.stops.evaluate(frame).to_owned();
        if self.is_radial {
            let radius = (end.to_vec2() - start.to_vec2()).hypot() as f32;
            let mut grad = peniko::Gradient::new_radial(start, radius);
            grad.stops = stops;
            grad.into()
        } else {
            let mut grad = peniko::Gradient::new_linear(start, end);
            grad.stops = stops;
            grad.into()
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColorStops {
    pub frames: Vec<Time>,
    pub values: Vec<Vec<f32>>,
    pub count: usize,
}

impl ColorStops {
    pub fn evaluate(&self, frame: f32) -> fixed::ColorStops {
        self.evaluate_inner(frame).unwrap_or_default()
    }

    fn evaluate_inner(&self, frame: f32) -> Option<fixed::ColorStops> {
        let ([ix0, ix1], t) = Time::frames_and_weight(&self.frames, frame)?;
        let v0 = self.values.get(ix0)?;
        let v1 = self.values.get(ix1)?;
        let opacity_start = if v0.len() > self.count * 4 {
            Some(self.count * 4)
        } else {
            None
        };
        let mut stops: fixed::ColorStops = Default::default();
        for i in 0..self.count {
            let j = i * 4;
            let offset = v0.get(j)?.lerp(v1.get(j)?, t);
            let r = v0.get(j + 1)?.lerp(v1.get(j + 1)?, t) as f64;
            let g = v0.get(j + 2)?.lerp(v1.get(j + 2)?, t) as f64;
            let b = v0.get(j + 3)?.lerp(v1.get(j + 3)?, t) as f64;
            let a = if let Some(_opacity_start) = opacity_start {
                // TODO: find and lerp opacities
                1.0
            } else {
                1.0
            };
            stops.push((offset, fixed::Color::rgba(r, g, b, a)).into())
        }
        Some(stops)
    }
}

/// Animated brush.
#[derive(Clone, Debug)]
pub enum Brush {
    /// Solid color.
    Solid(Value<Color>),
    /// Gradient color.
    Gradient(Gradient),
}

impl Brush {
    /// Returns true if the value contains no animated properties.
    pub fn is_fixed(&self) -> bool {
        match self {
            Self::Solid(value) => value.is_fixed(),
            Self::Gradient(value) => value.is_fixed(),
        }
    }

    /// Evaluates the animation at the specified time.
    pub fn evaluate(&self, alpha: f32, frame: f32) -> fixed::Brush {
        match self {
            Self::Solid(value) => value.evaluate(frame).with_alpha_factor(alpha).into(),
            Self::Gradient(value) => value.evaluate(frame),
        }
    }

    /// Converts the animated value to its model representation.
    pub fn to_model(self) -> super::Brush {
        if self.is_fixed() {
            super::Brush::Fixed(self.evaluate(1.0, 0.0))
        } else {
            super::Brush::Animated(self)
        }
    }
}
