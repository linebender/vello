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

use crate::brush::{Brush, Color};
use crate::geometry::Affine;
use crate::path::PathElement;
use crate::scene::{Fill, SceneBuilder, SceneFragment};

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
            brush.unwrap_or(&Brush::Solid(Color::rgb8(255, 255, 255))),
            None,
            convert_path(path.elements()),
        );
        builder.finish();
        Some(fragment)
    }

    /// Returns a scene fragment containing the commands and resources to
    /// render the specified color glyph.
    pub fn get_color(&mut self, palette_index: u16, gid: u16) -> Option<SceneFragment> {
        use crate::geometry::*;
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
                            convert_transformed_path(path.elements(), xform),
                        );
                    } else {
                        builder.push_layer(Default::default(), convert_path(path.elements()));
                    }
                }
                Command::PopClip => builder.pop_layer(),
                Command::PushLayer(bounds) => {
                    let mut rect = Rect {
                        min: Point::new(bounds.min.x, bounds.min.y),
                        max: Point::new(bounds.max.x, bounds.max.y),
                    };
                    if let Some(xform) = xform_stack.last() {
                        rect = rect.transform(xform);
                    }
                    builder.push_layer(Default::default(), rect.elements());
                }
                Command::PopLayer => builder.pop_layer(),
                Command::BeginBlend(bounds, mode) => {
                    let mut rect = Rect {
                        min: Point::new(bounds.min.x, bounds.min.y),
                        max: Point::new(bounds.max.x, bounds.max.y),
                    };
                    if let Some(xform) = xform_stack.last() {
                        rect = rect.transform(xform);
                    }
                    builder.push_layer(convert_blend(*mode), rect.elements())
                }
                Command::EndBlend => builder.pop_layer(),
                Command::SimpleFill(path_index, brush, brush_xform) => {
                    let path = glyph.path(*path_index)?;
                    let brush = convert_brush(brush);
                    let brush_xform = brush_xform.map(|xform| convert_transform(&xform));
                    if let Some(xform) = xform_stack.last() {
                        builder.fill(
                            Fill::NonZero,
                            &brush,
                            brush_xform.map(|x| x * *xform),
                            convert_transformed_path(path.elements(), xform),
                        );
                    } else {
                        builder.fill(
                            Fill::NonZero,
                            &brush,
                            brush_xform,
                            convert_path(path.elements()),
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

fn convert_path(
    path: impl Iterator<Item = moscato::Element> + Clone,
) -> impl Iterator<Item = PathElement> + Clone {
    use crate::geometry::Point;
    path.map(|el| match el {
        moscato::Element::MoveTo(p0) => PathElement::MoveTo(Point::new(p0.x, p0.y)),
        moscato::Element::LineTo(p0) => PathElement::LineTo(Point::new(p0.x, p0.y)),
        moscato::Element::QuadTo(p0, p1) => {
            PathElement::QuadTo(Point::new(p0.x, p0.y), Point::new(p1.x, p1.y))
        }
        moscato::Element::CurveTo(p0, p1, p2) => PathElement::CurveTo(
            Point::new(p0.x, p0.y),
            Point::new(p1.x, p1.y),
            Point::new(p2.x, p2.y),
        ),
        moscato::Element::Close => PathElement::Close,
    })
}

fn convert_transformed_path(
    path: impl Iterator<Item = moscato::Element> + Clone,
    xform: &Affine,
) -> impl Iterator<Item = PathElement> + Clone {
    use crate::geometry::Point;
    let xform = *xform;
    path.map(move |el| match el {
        moscato::Element::MoveTo(p0) => {
            PathElement::MoveTo(Point::new(p0.x, p0.y).transform(&xform))
        }
        moscato::Element::LineTo(p0) => {
            PathElement::LineTo(Point::new(p0.x, p0.y).transform(&xform))
        }
        moscato::Element::QuadTo(p0, p1) => PathElement::QuadTo(
            Point::new(p0.x, p0.y).transform(&xform),
            Point::new(p1.x, p1.y).transform(&xform),
        ),
        moscato::Element::CurveTo(p0, p1, p2) => PathElement::CurveTo(
            Point::new(p0.x, p0.y).transform(&xform),
            Point::new(p1.x, p1.y).transform(&xform),
            Point::new(p2.x, p2.y).transform(&xform),
        ),
        moscato::Element::Close => PathElement::Close,
    })
}

fn convert_blend(mode: moscato::CompositeMode) -> crate::scene::BlendMode {
    use crate::scene::{BlendMode, Compose, Mix};
    use moscato::CompositeMode;
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

fn convert_transform(xform: &moscato::Transform) -> crate::geometry::Affine {
    crate::geometry::Affine {
        xx: xform.xx,
        yx: xform.yx,
        xy: xform.xy,
        yy: xform.yy,
        dx: xform.dx,
        dy: xform.dy,
    }
}

fn convert_brush(brush: &moscato::Brush) -> crate::brush::Brush {
    use crate::brush::*;
    use crate::geometry::*;
    match brush {
        moscato::Brush::Solid(color) => Brush::Solid(Color {
            r: color.r,
            g: color.g,
            b: color.b,
            a: color.a,
        }),
        moscato::Brush::LinearGradient(grad) => Brush::LinearGradient(LinearGradient {
            start: Point::new(grad.start.x, grad.start.y),
            end: Point::new(grad.end.x, grad.end.y),
            stops: convert_stops(&grad.stops),
            extend: convert_extend(grad.extend),
        }),
        moscato::Brush::RadialGradient(grad) => Brush::RadialGradient(RadialGradient {
            center0: Point::new(grad.center0.x, grad.center0.y),
            center1: Point::new(grad.center1.x, grad.center1.y),
            radius0: grad.radius0,
            radius1: grad.radius1,
            stops: convert_stops(&grad.stops),
            extend: convert_extend(grad.extend),
        }),
    }
}

fn convert_stops(stops: &[moscato::ColorStop]) -> crate::brush::GradientStops {
    use crate::brush::GradientStop;
    stops
        .iter()
        .map(|stop| GradientStop {
            offset: stop.offset,
            color: Color {
                r: stop.color.r,
                g: stop.color.g,
                b: stop.color.b,
                a: stop.color.a,
            },
        })
        .collect()
}

fn convert_extend(extend: moscato::ExtendMode) -> crate::brush::ExtendMode {
    use crate::brush::ExtendMode::*;
    match extend {
        moscato::ExtendMode::Pad => Pad,
        moscato::ExtendMode::Repeat => Repeat,
        moscato::ExtendMode::Reflect => Reflect,
    }
}
