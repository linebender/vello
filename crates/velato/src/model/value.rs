// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello::kurbo;
use vello::peniko;

/// Fixed or animated value.
#[derive(Clone, Debug)]
pub enum Value<T: Lerp> {
    /// Fixed value.
    Fixed(T),
    /// Animated value.
    Animated(Animated<T>),
}

impl<T: Lerp> Value<T> {
    /// Returns true if the value is fixed.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }

    /// Returns the value at a specified frame.
    pub fn evaluate(&self, frame: f32) -> T {
        match self {
            Self::Fixed(fixed) => fixed.clone(),
            Self::Animated(animated) => animated.evaluate(frame),
        }
    }
}

impl<T: Lerp + Default> Default for Value<T> {
    fn default() -> Self {
        Self::Fixed(T::default())
    }
}

/// Borrowed or owned value.
#[derive(Clone, Debug)]
pub enum ValueRef<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<'a, T> AsRef<T> for ValueRef<'a, T> {
    fn as_ref(&self) -> &T {
        match self {
            Self::Borrowed(value) => value,
            Self::Owned(value) => value,
        }
    }
}

impl<'a, T: Clone> ValueRef<'a, T> {
    pub fn to_owned(self) -> T {
        match self {
            Self::Borrowed(value) => value.clone(),
            Self::Owned(value) => value,
        }
    }
}

/// Time for a particular keyframe, represented as a frame number.
#[derive(Copy, Clone, Default, Debug)]
pub struct Time {
    /// Frame number.
    pub frame: f32,
}

impl Time {
    /// Returns the frame indices and interpolation weight for the given frame.
    pub(crate) fn frames_and_weight(times: &[Time], frame: f32) -> Option<([usize; 2], f32)> {
        if times.is_empty() {
            return None;
        }
        use core::cmp::Ordering::*;
        let ix = match times.binary_search_by(|x| {
            if x.frame < frame {
                Less
            } else if x.frame > frame {
                Greater
            } else {
                Equal
            }
        }) {
            Ok(ix) => ix,
            Err(ix) => ix.saturating_sub(1),
        };
        let ix0 = ix.min(times.len() - 1);
        let ix1 = (ix0 + 1).min(times.len() - 1);
        let t0 = times[ix0].frame;
        let t1 = times[ix1].frame;
        let t = (frame - t0) / (t1 - t0);
        Some(([ix0, ix1], t.clamp(0.0, 1.0)))
    }
}

#[derive(Clone, Debug)]
pub struct Animated<T: Lerp> {
    pub times: Vec<Time>,
    pub values: Vec<T>,
}

impl<T: Lerp> Animated<T> {
    /// Returns the value at the specified frame.
    pub fn evaluate(&self, frame: f32) -> T {
        self.evaluate_inner(frame).unwrap_or_default()
    }

    fn evaluate_inner(&self, frame: f32) -> Option<T> {
        let ([ix0, ix1], t) = Time::frames_and_weight(&self.times, frame)?;
        Some(self.values.get(ix0)?.lerp(self.values.get(ix1)?, t))
    }
}

pub trait Lerp: Clone + Default {
    fn lerp(&self, other: &Self, t: f32) -> Self;
}

impl Lerp for f32 {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        (*self) + (*other - *self) * t
    }
}

impl Lerp for f64 {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        (*self) + (*other - *self) * t as f64
    }
}

impl Lerp for kurbo::Point {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Self::new(self.x.lerp(&other.x, t), self.y.lerp(&other.y, t))
    }
}

impl Lerp for kurbo::Vec2 {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Self::new(self.x.lerp(&other.x, t), self.y.lerp(&other.y, t))
    }
}

impl Lerp for kurbo::Size {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        Self::new(
            self.width.lerp(&other.width, t),
            self.height.lerp(&other.height, t),
        )
    }
}

impl Lerp for peniko::Color {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        let r = (self.r as f64 / 255.0).lerp(&(other.r as f64 / 255.0), t);
        let g = (self.r as f64 / 255.0).lerp(&(other.g as f64 / 255.0), t);
        let b = (self.r as f64 / 255.0).lerp(&(other.b as f64 / 255.0), t);
        let a = (self.r as f64 / 255.0).lerp(&(other.a as f64 / 255.0), t);
        peniko::Color::rgba(r, g, b, a)
    }
}
