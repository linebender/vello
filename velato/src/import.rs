// Copyright 2023 Google LLC
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{model, model::*, Composition};

use vello::kurbo::{Point, Size, Vec2};
use vello::peniko::{self, BlendMode, Color, Compose, Mix};

use std::collections::HashMap;

pub fn import_composition(
    source: impl AsRef<[u8]>,
) -> Result<Composition, Box<dyn std::error::Error>> {
    let source = bodymovin::Bodymovin::from_bytes(source)?;
    let mut target = Composition::default();
    target.frame_rate = source.frame_rate as f32;
    target.frames = source.in_point as f32..source.out_point as f32;
    target.width = source.width as u32;
    target.height = source.height as u32;
    let mut idmap: HashMap<usize, usize> = HashMap::default();
    for asset in &source.assets {
        match asset {
            bodymovin::sources::Asset::PreComp(precomp) => {
                idmap.clear();
                let mut layers = vec![];
                let mut mask_layer = None;
                for layer in &precomp.layers {
                    let index = layers.len();
                    if let Some((mut layer, id, mask_blend)) = conv_layer(layer) {
                        if let (Some(mask_blend), Some(mask_layer)) =
                            (mask_blend, mask_layer.take())
                        {
                            layer.mask_layer = Some((mask_blend, mask_layer));
                        }
                        if layer.is_mask {
                            mask_layer = Some(index);
                        }
                        idmap.insert(id, index);
                        layers.push(layer);
                    }
                }
                for layer in &mut layers {
                    if let Some(parent) = layer.parent {
                        layer.parent = idmap.get(&parent).copied();
                    }
                }
                target.assets.insert(precomp.id.clone(), layers);
            }
            _ => {}
        }
    }
    idmap.clear();
    let mut layers = vec![];
    let mut mask_layer = None;
    for layer in &source.layers {
        let index = layers.len();
        if let Some((mut layer, id, mask_blend)) = conv_layer(layer) {
            if let (Some(mask_blend), Some(mask_layer)) = (mask_blend, mask_layer.take()) {
                layer.mask_layer = Some((mask_blend, mask_layer));
            }
            if layer.is_mask {
                mask_layer = Some(index);
            }
            idmap.insert(id, index);
            layers.push(layer);
        }
    }
    for layer in &mut layers {
        if let Some(parent) = layer.parent {
            layer.parent = idmap.get(&parent).copied();
        }
    }
    target.layers = layers;
    Ok(target)
}

fn conv_layer(source: &bodymovin::layers::AnyLayer) -> Option<(Layer, usize, Option<BlendMode>)> {
    use bodymovin::layers::AnyLayer;
    let mut layer = Layer::default();
    let params;
    match source {
        AnyLayer::Null(value) => {
            params = setup_layer(value, &mut layer);
        }
        AnyLayer::PreComp(value) => {
            params = setup_layer(value, &mut layer);
            let name = value.mixin.ref_id.clone();
            let time_remap = conv_scalar(&value.mixin.time_remapping);
            layer.content = Content::Instance { name, time_remap };
        }
        AnyLayer::Shape(value) => {
            params = setup_layer(value, &mut layer);
            let mut shapes = vec![];
            for shape in &value.mixin.shapes {
                if let Some(shape) = conv_shape(shape) {
                    shapes.push(shape);
                }
            }
            layer.content = Content::Shape(shapes);
        }
        _ => return None,
    }
    let (id, matte_mode) = params;
    Some((layer, id, matte_mode))
}

fn setup_layer<T>(
    source: &bodymovin::layers::Layer<T>,
    target: &mut Layer,
) -> (usize, Option<BlendMode>) {
    use bodymovin::helpers::MatteMode;
    target.name = source.name.clone().unwrap_or_default();
    target.parent = source.parent.map(|i| i as usize);
    let (transform, opacity) = conv_transform(&source.transform);
    target.transform = transform;
    target.opacity = opacity;
    target.width = source.width.unwrap_or(0) as _;
    target.height = source.height.unwrap_or(0) as _;
    target.is_mask = source.is_track_matte;
    let matte_mode = source.matte_mode.as_ref().map(|mode| match mode {
        MatteMode::Normal => Mix::Normal.into(),
        MatteMode::Alpha | MatteMode::Luma => Compose::SrcIn.into(),
        MatteMode::InvertAlpha | MatteMode::InvertLuma => Compose::SrcOut.into(),
    });
    target.blend_mode = conv_blend_mode(&source.blend_mode);
    if target.blend_mode == Some(peniko::Mix::Normal.into()) {
        target.blend_mode = None;
    }
    target.frames = source.in_point as f32..source.out_point as f32;
    target.stretch = source.stretch as f32;
    target.start_frame = source.start_time as f32;
    for mask_source in &source.masks {
        if let Some(geometry) = conv_shape_geometry(&mask_source.points) {
            let mode = peniko::BlendMode::default();
            let opacity = conv_scalar(&mask_source.opacity);
            target.masks.push(Mask {
                mode,
                geometry,
                opacity,
            })
        }
    }
    (source.index as usize, matte_mode)
}

fn conv_transform(value: &bodymovin::helpers::Transform) -> (Transform, Value<f32>) {
    let transform = animated::Transform {
        anchor: conv_point(&value.anchor_point),
        position: conv_point(&value.position),
        scale: conv_vec2(&value.scale),
        rotation: conv_scalar(&value.rotation),
        skew: conv_scalar(&value.skew),
        skew_angle: conv_scalar(&value.skew_axis),
    };
    let opacity = conv_scalar(&value.opacity);
    (transform.to_model(), opacity)
}

fn conv_shape_transform(value: &bodymovin::shapes::Transform) -> GroupTransform {
    let transform = animated::Transform {
        anchor: conv_point(&value.anchor_point),
        position: conv_point(&value.position),
        scale: conv_vec2(&value.scale),
        rotation: conv_scalar(&value.rotation),
        skew: conv_scalar(&value.skew),
        skew_angle: conv_scalar(&value.skew_axis),
    };
    let opacity = conv_scalar(&value.opacity);
    GroupTransform {
        transform: transform.to_model(),
        opacity,
    }
}

fn conv_scalar(value: &bodymovin::properties::Scalar) -> Value<f32> {
    use bodymovin::properties::Value::*;
    match &value.value {
        Fixed(value) => Value::Fixed(*value as f32),
        Animated(animated) => {
            let mut frames = vec![];
            let mut values = vec![];
            let mut last_value = None;
            for value in animated {
                if let Some(data) = value.start_value.as_ref().or(last_value.flatten()) {
                    frames.push(Time {
                        frame: value.start_time as f32,
                    });
                    values.push(data.0 as f32);
                }
                last_value = Some(value.end_value.as_ref());
            }
            Value::Animated(model::Animated {
                times: frames,
                values,
            })
        }
    }
}

fn conv_multi<T: Lerp>(
    value: &bodymovin::properties::MultiDimensional,
    f: impl Fn(&Vec<f64>) -> T,
) -> Value<T> {
    use bodymovin::properties::Value::*;
    match &value.value {
        Fixed(value) => Value::Fixed(f(value)),
        Animated(animated) => {
            let mut frames = vec![];
            let mut values = vec![];
            let mut last_value = None;
            for value in animated {
                if let Some(data) = value.start_value.as_ref().or(last_value) {
                    frames.push(Time {
                        frame: value.start_time as f32,
                    });
                    values.push(f(data));
                }
                last_value = value.end_value.as_ref();
            }
            Value::Animated(model::Animated {
                times: frames,
                values,
            })
        }
    }
}

fn conv_point(value: &bodymovin::properties::MultiDimensional) -> Value<Point> {
    conv_multi(value, |x| {
        Point::new(
            x.get(0).copied().unwrap_or(0.0),
            x.get(1).copied().unwrap_or(0.0),
        )
    })
}

fn conv_color(value: &bodymovin::properties::MultiDimensional) -> Value<Color> {
    conv_multi(value, |x| {
        Color::rgb(
            x.get(0).copied().unwrap_or(0.0),
            x.get(1).copied().unwrap_or(0.0),
            x.get(2).copied().unwrap_or(0.0),
        )
    })
}

fn conv_vec2(value: &bodymovin::properties::MultiDimensional) -> Value<Vec2> {
    conv_multi(value, |x| {
        Vec2::new(
            x.get(0).copied().unwrap_or(0.0),
            x.get(1).copied().unwrap_or(0.0),
        )
    })
}

fn conv_size(value: &bodymovin::properties::MultiDimensional) -> Value<Size> {
    conv_multi(value, |x| {
        Size::new(
            x.get(0).copied().unwrap_or(0.0),
            x.get(1).copied().unwrap_or(0.0),
        )
    })
}

fn conv_gradient_colors(value: &bodymovin::helpers::GradientColors) -> ColorStops {
    use bodymovin::properties::Value::*;
    let count = value.count as usize;
    match &value.colors.value {
        Fixed(value) => {
            let mut stops = fixed::ColorStops::new();
            for chunk in value.chunks_exact(4) {
                stops.push(
                    (
                        chunk[0] as f32,
                        fixed::Color::rgba(chunk[1], chunk[2], chunk[3], 1.0),
                    )
                        .into(),
                )
            }
            ColorStops::Fixed(stops)
        }
        Animated(animated) => {
            let mut frames = vec![];
            let mut values = vec![];
            let mut last_value = None;
            for value in animated {
                if let Some(data) = value.start_value.as_ref().or(last_value.flatten()) {
                    frames.push(Time {
                        frame: value.start_time as f32,
                    });
                    values.push(data.iter().map(|x| *x as f32).collect::<Vec<_>>());
                }
                last_value = Some(value.end_value.as_ref());
            }
            ColorStops::Animated(animated::ColorStops {
                frames,
                values,
                count,
            })
        }
    }
}

fn conv_draw(value: &bodymovin::shapes::AnyShape) -> Option<Draw> {
    use bodymovin::helpers::{LineCap, LineJoin};
    use bodymovin::shapes::{AnyShape, GradientType};
    use peniko::{Cap, Join};
    match value {
        AnyShape::Fill(value) => {
            let color = conv_color(&value.color);
            let brush = animated::Brush::Solid(color).to_model();
            let opacity = conv_scalar(&value.opacity);
            Some(Draw {
                stroke: None,
                brush,
                opacity,
            })
        }
        AnyShape::Stroke(value) => {
            let stroke = animated::Stroke {
                width: conv_scalar(&value.width),
                join: match value.line_join {
                    LineJoin::Bevel => Join::Bevel,
                    LineJoin::Round => Join::Round,
                    LineJoin::Miter => Join::Miter,
                },
                miter_limit: value.miter_limit.map(|x| x as f32),
                cap: match value.line_cap {
                    LineCap::Butt => Cap::Butt,
                    LineCap::Round => Cap::Round,
                    LineCap::Square => Cap::Square,
                },
            };
            let color = conv_color(&value.color);
            let brush = animated::Brush::Solid(color).to_model();
            let opacity = conv_scalar(&value.opacity);
            Some(Draw {
                stroke: Some(stroke.to_model()),
                brush,
                opacity,
            })
        }
        AnyShape::GradientFill(value) => {
            let is_radial = matches!(value.ty, GradientType::Radial);
            let start_point = conv_point(&value.start_point);
            let end_point = conv_point(&value.end_point);
            let gradient = animated::Gradient {
                is_radial,
                start_point,
                end_point,
                stops: conv_gradient_colors(&value.gradient_colors),
            };
            let brush = animated::Brush::Gradient(gradient).to_model();
            Some(Draw {
                stroke: None,
                brush,
                opacity: Value::Fixed(100.0),
            })
        }
        AnyShape::GradientStroke(value) => {
            let stroke = animated::Stroke {
                width: conv_scalar(&value.stroke_width),
                join: match value.line_join {
                    LineJoin::Bevel => Join::Bevel,
                    LineJoin::Round => Join::Round,
                    LineJoin::Miter => Join::Miter,
                },
                miter_limit: value.miter_limit.map(|x| x as f32),
                cap: match value.line_cap {
                    LineCap::Butt => Cap::Butt,
                    LineCap::Round => Cap::Round,
                    LineCap::Square => Cap::Square,
                },
            };
            let is_radial = matches!(value.ty, GradientType::Radial);
            let start_point = conv_point(&value.start_point);
            let end_point = conv_point(&value.end_point);
            let gradient = animated::Gradient {
                is_radial,
                start_point,
                end_point,
                stops: conv_gradient_colors(&value.gradient_colors),
            };
            let brush = animated::Brush::Gradient(gradient).to_model();
            Some(Draw {
                stroke: Some(stroke.to_model()),
                brush,
                opacity: Value::Fixed(100.0),
            })
        }
        _ => None,
    }
}

fn conv_shape(value: &bodymovin::shapes::AnyShape) -> Option<Shape> {
    use bodymovin::shapes::AnyShape;
    if let Some(draw) = conv_draw(value) {
        return Some(Shape::Draw(draw));
    } else if let Some(geometry) = conv_geometry(value) {
        return Some(Shape::Geometry(geometry));
    }
    match value {
        AnyShape::Group(value) => {
            let mut shapes = vec![];
            let mut group_transform = None;
            for item in &value.items {
                match item {
                    AnyShape::Transform(transform) => {
                        group_transform = Some(conv_shape_transform(transform));
                    }
                    _ => {
                        if let Some(shape) = conv_shape(item) {
                            shapes.push(shape);
                        }
                    }
                }
            }
            if !shapes.is_empty() {
                Some(Shape::Group(shapes, group_transform))
            } else {
                None
            }
        }
        AnyShape::Repeater(value) => {
            let repeater = animated::Repeater {
                copies: conv_scalar(&value.copies),
                offset: conv_scalar(&value.offset),
                anchor_point: conv_point(&value.transform.anchor_point),
                position: conv_point(&value.transform.position),
                rotation: conv_scalar(&value.transform.rotation),
                scale: conv_vec2(&value.transform.scale),
                start_opacity: conv_scalar(&value.transform.start_opacity),
                end_opacity: conv_scalar(&value.transform.end_opacity),
            };
            Some(Shape::Repeater(repeater.to_model()))
        }
        _ => None,
    }
}

fn conv_geometry(value: &bodymovin::shapes::AnyShape) -> Option<Geometry> {
    use bodymovin::shapes::AnyShape;
    match value {
        AnyShape::Ellipse(value) => {
            let ellipse = animated::Ellipse {
                is_ccw: value.direction as i32 == 3,
                position: conv_point(&value.position),
                size: conv_size(&value.size),
            };
            Some(Geometry::Ellipse(ellipse))
        }
        AnyShape::Rect(value) => {
            let rect = animated::Rect {
                is_ccw: value.direction as i32 == 3,
                position: conv_point(&value.position),
                size: conv_size(&value.size),
                corner_radius: conv_scalar(&value.rounded_corners),
            };
            Some(Geometry::Rect(rect))
        }
        AnyShape::Shape(value) => conv_shape_geometry(&value.vertices),
        _ => None,
    }
}

fn conv_shape_geometry(value: &bodymovin::properties::Shape) -> Option<Geometry> {
    use bodymovin::properties::Value::*;
    let mut is_closed = false;
    match &value.value {
        Fixed(value) => {
            let (points, is_closed) = conv_spline(value);
            let mut path = vec![];
            points.as_slice().to_path(is_closed, &mut path);
            Some(Geometry::Fixed(path))
        }
        Animated(animated) => {
            let mut frames = vec![];
            let mut values = vec![];
            let mut last_value = None;
            for value in animated {
                if let Some(data) = value.start_value.as_ref().or(last_value) {
                    frames.push(Time {
                        frame: value.start_time as f32,
                    });
                    let (points, is_frame_closed) = conv_spline(data.get(0)?);
                    values.push(points);
                    is_closed |= is_frame_closed;
                }
                last_value = value.end_value.as_ref();
            }
            Some(Geometry::Spline(animated::Spline {
                is_closed,
                times: frames,
                values,
            }))
        }
    }
}

fn conv_spline(value: &bodymovin::properties::ShapeValue) -> (Vec<Point>, bool) {
    use core::iter::repeat;
    let mut points = Vec::with_capacity(value.vertices.len() * 3);
    let is_closed = value.closed.unwrap_or(false);
    for ((v, i), o) in value
        .vertices
        .iter()
        .zip(value.in_point.iter().chain(repeat(&(0.0, 0.0))))
        .zip(value.out_point.iter().chain(repeat(&(0.0, 0.0))))
    {
        points.push((v.0, v.1).into());
        points.push((i.0, i.1).into());
        points.push((o.0, o.1).into());
    }
    (points, is_closed)
}

fn conv_blend_mode(value: &bodymovin::helpers::BlendMode) -> Option<BlendMode> {
    use bodymovin::helpers::BlendMode::*;
    Some(match value {
        Normal => return None,
        Multiply => BlendMode::from(Mix::Multiply),
        Screen => BlendMode::from(Mix::Screen),
        Overlay => BlendMode::from(Mix::Overlay),
        Darken => BlendMode::from(Mix::Darken),
        Lighten => BlendMode::from(Mix::Lighten),
        ColorDodge => BlendMode::from(Mix::ColorDodge),
        ColorBurn => BlendMode::from(Mix::ColorBurn),
        HardLight => BlendMode::from(Mix::HardLight),
        SoftLight => BlendMode::from(Mix::SoftLight),
        Difference => BlendMode::from(Mix::Difference),
        Exclusion => BlendMode::from(Mix::Exclusion),
        Hue => BlendMode::from(Mix::Hue),
        Saturation => BlendMode::from(Mix::Saturation),
        Color => BlendMode::from(Mix::Color),
        Luminosity => BlendMode::from(Mix::Luminosity),
    })
}
