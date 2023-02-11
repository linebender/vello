// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR MIT

use vello::kurbo::{self, Affine, PathEl, Point, Shape as _, Size, Vec2};
use vello::peniko::{self, BlendMode, Color};

use core::ops::Range;

pub mod animated;
pub mod fixed;

mod spline;
mod value;

pub use value::{Animated, Lerp, Time, Value, ValueRef};

pub(crate) use spline::SplineToPath;

macro_rules! simple_value {
    ($name:ident) => {
        #[derive(Clone, Debug)]
        pub enum $name {
            Fixed(fixed::$name),
            Animated(animated::$name),
        }

        impl $name {
            pub fn is_fixed(&self) -> bool {
                matches!(self, Self::Fixed(_))
            }
            pub fn evaluate(&self, frame: f32) -> ValueRef<fixed::$name> {
                match self {
                    Self::Fixed(value) => ValueRef::Borrowed(value),
                    Self::Animated(value) => ValueRef::Owned(value.evaluate(frame)),
                }
            }
        }
    };
}

simple_value!(Transform);
simple_value!(Stroke);
simple_value!(Repeater);
simple_value!(ColorStops);

#[derive(Clone, Debug)]
pub enum Brush {
    Fixed(fixed::Brush),
    Animated(animated::Brush),
}

impl Brush {
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }

    pub fn evaluate(&self, alpha: f32, frame: f32) -> ValueRef<fixed::Brush> {
        match self {
            Self::Fixed(value) => {
                if alpha == 1.0 {
                    ValueRef::Borrowed(value)
                } else {
                    ValueRef::Owned(fixed::brush_with_alpha(value, alpha))
                }
            }
            Self::Animated(value) => ValueRef::Owned(value.evaluate(alpha, frame)),
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::Fixed(Affine::IDENTITY)
    }
}

#[derive(Clone, Debug)]
pub enum Geometry {
    Fixed(Vec<PathEl>),
    Rect(animated::Rect),
    Ellipse(animated::Ellipse),
    Spline(animated::Spline),
}

impl Geometry {
    pub fn evaluate(&self, frame: f32, path: &mut Vec<PathEl>) {
        match self {
            Self::Fixed(value) => {
                path.extend_from_slice(value);
            }
            Self::Rect(value) => {
                path.extend(value.evaluate(frame).path_elements(0.1));
            }
            Self::Ellipse(value) => {
                path.extend(value.evaluate(frame).path_elements(0.1));
            }
            Self::Spline(value) => {
                value.evaluate(frame, path);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Draw {
    /// Parameters for a stroked draw operation.
    pub stroke: Option<Stroke>,
    /// Brush for the draw operation.
    pub brush: Brush,
    /// Opacity of the draw operation.
    pub opacity: Value<f32>,
}

/// Elements of a shape layer.
#[derive(Clone, Debug)]
pub enum Shape {
    /// Group of shapes with an optional transform.
    Group(Vec<Shape>, Option<GroupTransform>),
    /// Geometry element.
    Geometry(Geometry),
    /// Fill or stroke element.
    Draw(Draw),
    /// Repeater element.
    Repeater(Repeater),
}

/// Transform and opacity for a shape group.
#[derive(Clone, Debug)]
pub struct GroupTransform {
    pub transform: Transform,
    pub opacity: Value<f32>,
}

/// Layer in an animation.
#[derive(Clone, Default, Debug)]
pub struct Layer {
    /// Name of the layer.
    pub name: String,
    /// Index of the transform parent layer.
    pub parent: Option<usize>,
    /// Transform for the entire layer.
    pub transform: Transform,
    /// Opacity for the entire layer.
    pub opacity: Value<f32>,
    /// Width of the layer.
    pub width: u32,
    /// Height of the layer.
    pub height: u32,
    /// Blend mode for the layer.
    pub blend_mode: Option<peniko::BlendMode>,
    /// Range of frames in which the layer is active.
    pub frames: Range<f32>,
    /// Frame time stretch factor.
    pub stretch: f32,
    /// Starting frame for the layer (only applied to instances).
    pub start_frame: f32,
    /// List of masks applied to the content.
    pub masks: Vec<Mask>,
    /// True if the layer is used as a mask.
    pub is_mask: bool,
    /// Mask blend mode and layer.
    pub mask_layer: Option<(BlendMode, usize)>,
    /// Content of the layer.
    pub content: Content,
}

/// Matte layer mode.
#[derive(Copy, Clone, PartialEq, Eq, Default, Debug)]
pub enum Matte {
    #[default]
    Normal,
    Alpha,
    InvertAlpha,
    Luma,
    InvertLuma,
}

/// Mask for a layer.
#[derive(Clone, Debug)]
pub struct Mask {
    /// Blend mode for the mask.
    pub mode: peniko::BlendMode,
    /// Geometry that defines the shape of the mask.
    pub geometry: Geometry,
    /// Opacity of the mask.
    pub opacity: Value<f32>,
}

/// Content of a layer.
#[derive(Clone, Default, Debug)]
pub enum Content {
    /// Emtpy layer.
    #[default]
    None,
    /// Asset instance with the specified name and time remapping.
    Instance {
        name: String,
        time_remap: Value<f32>,
    },
    /// Collection of shapes.
    Shape(Vec<Shape>),
}
