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

//! Support for glyph rendering.

pub use moscato::pinot;

use crate::scene::{SceneBuilder, SceneFragment};
use peniko::kurbo::{Affine, Rect};
use peniko::{Brush, Color, Fill};

use moscato::{Context, Scaler};
use pinot::{types::Tag, FontRef};

use smallvec::SmallVec;

/// General context for creating scene fragments for glyph outlines.
pub struct GlyphContext {
    ctx: Context,
}

impl GlyphContext {
    /// Creates a new context.
    pub fn new() -> Self {
        Self {
            ctx: Context::new(),
        }
    }

    /// Creates a new provider for generating scene fragments for glyphs from
    /// the specified font and settings.
    pub fn new_provider<'a, V>(
        &'a mut self,
        font: &FontRef<'a>,
        font_id: Option<u64>,
        ppem: f32,
        hint: bool,
        variations: V,
    ) -> GlyphProvider<'a>
    where
        V: IntoIterator,
        V::Item: Into<(Tag, f32)>,
    {
        let scaler = if let Some(font_id) = font_id {
            self.ctx
                .new_scaler_with_id(font, font_id)
                .size(ppem)
                .hint(hint)
                .variations(variations)
                .build()
        } else {
            self.ctx
                .new_scaler(font)
                .size(ppem)
                .hint(hint)
                .variations(variations)
                .build()
        };
        GlyphProvider { scaler }
    }
}

/// Generator for scene fragments containing glyph outlines for a specific
/// font.
pub struct GlyphProvider<'a> {
    scaler: Scaler<'a>,
}

impl<'a> GlyphProvider<'a> {
    /// Returns a scene fragment containing the commands to render the
    /// specified glyph.
    pub fn get(&mut self, gid: u16, brush: Option<&Brush>) -> Option<SceneFragment> {
        let glyph = self.scaler.glyph(gid)?;
        let path = glyph.path(0)?;
        let mut fragment = SceneFragment::default();
        let mut builder = SceneBuilder::for_fragment(&mut fragment);
        builder.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            brush.unwrap_or(&Brush::Solid(Color::rgb8(255, 255, 255))),
            None,
            &convert_path(path.elements()),
        );
        builder.finish();
        Some(fragment)
    }

    /// Returns a scene fragment containing the commands and resources to
    /// render the specified color glyph.
    pub fn get_color(&mut self, palette_index: u16, gid: u16) -> Option<SceneFragment> {
        use moscato::Command;
        let glyph = self.scaler.color_glyph(palette_index, gid)?;
        let mut fragment = SceneFragment::default();
        let mut builder = SceneBuilder::for_fragment(&mut fragment);
        let mut xform_stack: SmallVec<[Affine; 8]> = SmallVec::new();
        for command in glyph.commands() {
            match command {
                Command::PushTransform(xform) => {
                    let xform = if let Some(parent) = xform_stack.last() {
                        convert_transform(xform) * *parent
                    } else {
                        convert_transform(xform)
                    };
                    xform_stack.push(xform);
                }
                Command::PopTransform => {
                    xform_stack.pop();
                }
                Command::PushClip(path_index) => {
                    let path = glyph.path(*path_index)?;
                    if let Some(xform) = xform_stack.last() {
                        builder.push_layer(
                            Default::default(),
                            Affine::IDENTITY,
                            &convert_transformed_path(path.elements(), xform),
                        );
                    } else {
                        builder.push_layer(
                            Default::default(),
                            Affine::IDENTITY,
                            &convert_path(path.elements()),
                        );
                    }
                }
                Command::PopClip => builder.pop_layer(),
                Command::PushLayer(bounds) => {
                    let mut min = convert_point(bounds.min);
                    let mut max = convert_point(bounds.max);
                    if let Some(xform) = xform_stack.last() {
                        min = *xform * min;
                        max = *xform * max;
                    }
                    let rect = Rect::from_points(min, max);
                    builder.push_layer(Default::default(), Affine::IDENTITY, &rect);
                }
                Command::PopLayer => builder.pop_layer(),
                Command::BeginBlend(bounds, mode) => {
                    let mut min = convert_point(bounds.min);
                    let mut max = convert_point(bounds.max);
                    if let Some(xform) = xform_stack.last() {
                        min = *xform * min;
                        max = *xform * max;
                    }
                    let rect = Rect::from_points(min, max);
                    builder.push_layer(convert_blend(*mode), Affine::IDENTITY, &rect);
                }
                Command::EndBlend => builder.pop_layer(),
                Command::SimpleFill(path_index, brush, brush_xform) => {
                    let path = glyph.path(*path_index)?;
                    let brush = convert_brush(brush);
                    let brush_xform = brush_xform.map(|xform| convert_transform(&xform));
                    if let Some(xform) = xform_stack.last() {
                        builder.fill(
                            Fill::NonZero,
                            Affine::IDENTITY,
                            &brush,
                            brush_xform.map(|x| x * *xform),
                            &convert_transformed_path(path.elements(), xform),
                        );
                    } else {
                        builder.fill(
                            Fill::NonZero,
                            Affine::IDENTITY,
                            &brush,
                            brush_xform,
                            &convert_path(path.elements()),
                        );
                    }
                }
                Command::Fill(_brush, _brush_xform) => {
                    // TODO: this needs to compute a bounding box for
                    // the parent clips
                }
            }
        }
        builder.finish();
        Some(fragment)
    }
}

fn convert_path(path: impl Iterator<Item = moscato::Element> + Clone) -> peniko::kurbo::BezPath {
    let mut result = peniko::kurbo::BezPath::new();
    for el in path {
        result.push(convert_path_el(&el));
    }
    result
}

fn convert_transformed_path(
    path: impl Iterator<Item = moscato::Element> + Clone,
    xform: &Affine,
) -> peniko::kurbo::BezPath {
    let mut result = peniko::kurbo::BezPath::new();
    for el in path {
        result.push(*xform * convert_path_el(&el));
    }
    result
}

fn convert_blend(mode: moscato::CompositeMode) -> peniko::BlendMode {
    use moscato::CompositeMode;
    use peniko::{BlendMode, Compose, Mix};
    let mut mix = Mix::Normal;
    let mut compose = Compose::SrcOver;
    match mode {
        CompositeMode::Clear => compose = Compose::Clear,
        CompositeMode::Src => compose = Compose::Copy,
        CompositeMode::Dest => compose = Compose::Dest,
        CompositeMode::SrcOver => {}
        CompositeMode::DestOver => compose = Compose::DestOver,
        CompositeMode::SrcIn => compose = Compose::SrcIn,
        CompositeMode::DestIn => compose = Compose::DestIn,
        CompositeMode::SrcOut => compose = Compose::SrcOut,
        CompositeMode::DestOut => compose = Compose::DestOut,
        CompositeMode::SrcAtop => compose = Compose::SrcAtop,
        CompositeMode::DestAtop => compose = Compose::DestAtop,
        CompositeMode::Xor => compose = Compose::Xor,
        CompositeMode::Plus => compose = Compose::Plus,
        CompositeMode::Screen => mix = Mix::Screen,
        CompositeMode::Overlay => mix = Mix::Overlay,
        CompositeMode::Darken => mix = Mix::Darken,
        CompositeMode::Lighten => mix = Mix::Lighten,
        CompositeMode::ColorDodge => mix = Mix::ColorDodge,
        CompositeMode::ColorBurn => mix = Mix::ColorBurn,
        CompositeMode::HardLight => mix = Mix::HardLight,
        CompositeMode::SoftLight => mix = Mix::SoftLight,
        CompositeMode::Difference => mix = Mix::Difference,
        CompositeMode::Exclusion => mix = Mix::Exclusion,
        CompositeMode::Multiply => mix = Mix::Multiply,
        CompositeMode::HslHue => mix = Mix::Hue,
        CompositeMode::HslSaturation => mix = Mix::Saturation,
        CompositeMode::HslColor => mix = Mix::Color,
        CompositeMode::HslLuminosity => mix = Mix::Luminosity,
    }
    BlendMode { mix, compose }
}

fn convert_transform(xform: &moscato::Transform) -> peniko::kurbo::Affine {
    peniko::kurbo::Affine::new([
        xform.xx as f64,
        xform.yx as f64,
        xform.xy as f64,
        xform.yy as f64,
        xform.dx as f64,
        xform.dy as f64,
    ])
}

fn convert_point(point: moscato::Point) -> peniko::kurbo::Point {
    peniko::kurbo::Point::new(point.x as f64, point.y as f64)
}

fn convert_brush(brush: &moscato::Brush) -> peniko::Brush {
    use peniko::{LinearGradient, RadialGradient};
    match brush {
        moscato::Brush::Solid(color) => Brush::Solid(Color {
            r: color.r,
            g: color.g,
            b: color.b,
            a: color.a,
        }),
        moscato::Brush::LinearGradient(grad) => Brush::LinearGradient(LinearGradient {
            start: convert_point(grad.start),
            end: convert_point(grad.end),
            stops: convert_stops(&grad.stops),
            extend: convert_extend(grad.extend),
        }),
        moscato::Brush::RadialGradient(grad) => Brush::RadialGradient(RadialGradient {
            start_center: convert_point(grad.center0),
            end_center: convert_point(grad.center1),
            start_radius: grad.radius0,
            end_radius: grad.radius1,
            stops: convert_stops(&grad.stops),
            extend: convert_extend(grad.extend),
        }),
    }
}

fn convert_stops(stops: &[moscato::ColorStop]) -> peniko::ColorStops {
    stops
        .iter()
        .map(|stop| {
            (
                stop.offset,
                Color {
                    r: stop.color.r,
                    g: stop.color.g,
                    b: stop.color.b,
                    a: stop.color.a,
                },
            )
                .into()
        })
        .collect()
}

fn convert_extend(extend: moscato::ExtendMode) -> peniko::Extend {
    use peniko::Extend::*;
    match extend {
        moscato::ExtendMode::Pad => Pad,
        moscato::ExtendMode::Repeat => Repeat,
        moscato::ExtendMode::Reflect => Reflect,
    }
}

fn convert_path_el(el: &moscato::Element) -> peniko::kurbo::PathEl {
    use peniko::kurbo::PathEl::*;
    match el {
        moscato::Element::MoveTo(p0) => MoveTo(convert_point(*p0)),
        moscato::Element::LineTo(p0) => LineTo(convert_point(*p0)),
        moscato::Element::QuadTo(p0, p1) => QuadTo(convert_point(*p0), convert_point(*p1)),
        moscato::Element::CurveTo(p0, p1, p2) => {
            CurveTo(convert_point(*p0), convert_point(*p1), convert_point(*p2))
        }
        moscato::Element::Close => ClosePath,
    }
}
