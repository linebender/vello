// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Drawing COLR glyphs.

use crate::color::Srgb;
use crate::color::{AlphaColor, DynamicColor};
use crate::glyph::{ColorGlyph, OutlinePath};
use crate::kurbo::{Affine, BezPath, Point, Rect, Shape};
use crate::math::FloatExt;
use crate::peniko::{self, BlendMode, ColorStops, Compose, Extend, Gradient, Mix};
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use peniko::{LinearGradientPosition, RadialGradientPosition, SweepGradientPosition};
use skrifa::color::{Brush, ColorPainter, ColorStop, CompositeMode, Transform};
use skrifa::outline::DrawSettings;
use skrifa::raw::TableProvider;
use skrifa::raw::types::BoundingBox;
use skrifa::{GlyphId, MetadataProvider};
use smallvec::SmallVec;

/// A trait for clients capable of rendering COLR glyphs.
pub trait ColrRenderer {
    /// Push a new clip layer.
    fn push_clip_layer(&mut self, clip: &BezPath);
    /// Push a new blend layer.
    fn push_blend_layer(&mut self, blend_mode: BlendMode);
    /// Fill the current area with the given solid color.
    fn fill_solid(&mut self, color: AlphaColor<Srgb>);
    /// Fill the current area with the given gradient color.
    fn fill_gradient(&mut self, gradient: Gradient);
    /// Set the transform for paints.
    fn set_paint_transform(&mut self, affine: Affine);
    /// Pop the last clip/blend layer.
    fn pop_layer(&mut self);
}

/// An abstraction for painting COLR glyphs.
pub struct ColrPainter<'a> {
    transforms: Vec<Affine>,
    color_glyph: Box<ColorGlyph<'a>>,
    context_color: AlphaColor<Srgb>,
    painter: &'a mut dyn ColrRenderer,
    layer_count: u32,
}

impl Debug for ColrPainter<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ColrPainter()").finish()
    }
}

impl<'a> ColrPainter<'a> {
    /// Create a new COLR painter.
    pub fn new(
        color_glyph: Box<ColorGlyph<'a>>,
        context_color: AlphaColor<Srgb>,
        painter: &'a mut impl ColrRenderer,
    ) -> Self {
        Self {
            transforms: vec![color_glyph.draw_transform],
            color_glyph,
            context_color,
            painter,
            layer_count: 0,
        }
    }

    /// Paint the underlying glyph.
    pub fn paint(&mut self) {
        let color_glyph = self.color_glyph.skrifa_glyph.clone();
        let location_ref = self.color_glyph.location;
        // Ignore errors for now.
        let _ = color_glyph.paint(location_ref, self);

        // In certain malformed fonts (i.e. if there is a cycle), skrifa will not
        // ensure that the push/pop count is the same, so we pop the remaining ones here.
        for _ in 0..self.layer_count {
            self.painter.pop_layer();
        }
    }

    fn cur_transform(&self) -> Affine {
        self.transforms.last().copied().unwrap_or_default()
    }

    fn palette_index_to_color(&self, palette_index: u16, alpha: f32) -> Option<AlphaColor<Srgb>> {
        if palette_index != u16::MAX {
            let color = self
                .color_glyph
                .font_ref
                .cpal()
                .ok()?
                .color_records_array()?
                .ok()?[palette_index as usize];

            Some(
                AlphaColor::from_rgba8(color.red, color.green, color.blue, color.alpha)
                    .multiply_alpha(alpha),
            )
        } else {
            Some(self.context_color.multiply_alpha(alpha))
        }
    }

    fn convert_stops(&self, stops: &[ColorStop]) -> ColorStops {
        let mut stops = stops
            .iter()
            .map(|s| {
                let color = self
                    .palette_index_to_color(s.palette_index, s.alpha)
                    .unwrap_or(AlphaColor::BLACK);

                peniko::ColorStop {
                    offset: s.offset,
                    color: DynamicColor::from_alpha_color(color),
                }
            })
            .collect::<SmallVec<[peniko::ColorStop; 4]>>();

        // Pad stops if necessary, since vello requires offsets
        // to start at 0.0 and end at 1.0.
        let first_stop = stops[0];
        let last_stop = *stops.last().unwrap();

        if first_stop.offset != 0.0 {
            let mut new_stop = first_stop;
            new_stop.offset = 0.0;
            stops.insert(0, new_stop);
        }

        if last_stop.offset != 1.0 {
            let mut new_stop = last_stop;
            new_stop.offset = 1.0;
            stops.push(new_stop);
        }

        // The COLR spec has the very specific requirement that if there are multiple stops with the
        // offset 1.0, only the last one should be used. We abstract this away by removing all such
        // superfluous stops.
        while let Some(stop) = stops.get(stops.len() - 2).map(|s| s.offset) {
            if (stop - 1.0).is_nearly_zero() {
                stops.remove(stops.len() - 2);
            } else {
                break;
            }
        }

        ColorStops(stops)
    }
}

impl ColorPainter for ColrPainter<'_> {
    fn push_transform(&mut self, t: Transform) {
        let affine = Affine::new([
            f64::from(t.xx),
            f64::from(t.yx),
            f64::from(t.xy),
            f64::from(t.yy),
            f64::from(t.dx),
            f64::from(t.dy),
        ]);
        self.transforms.push(self.cur_transform() * affine);
    }

    fn pop_transform(&mut self) {
        self.transforms.pop();
    }

    fn push_clip_glyph(&mut self, glyph_id: GlyphId) {
        let mut outline_builder = OutlinePath::new();

        let outline_glyphs = self.color_glyph.font_ref.outline_glyphs();
        let Some(outline_glyph) = outline_glyphs.get(glyph_id) else {
            return;
        };

        let _ = outline_glyph.draw(
            DrawSettings::unhinted(
                skrifa::instance::Size::unscaled(),
                self.color_glyph.location,
            ),
            &mut outline_builder,
        );

        let finished = outline_builder.0;
        let transformed = self.cur_transform() * finished;

        self.painter.push_clip_layer(&transformed);
        self.layer_count += 1;
    }

    fn push_clip_box(&mut self, clip_box: BoundingBox<f32>) {
        let rect = Rect::new(
            f64::from(clip_box.x_min),
            f64::from(clip_box.y_min),
            f64::from(clip_box.x_max),
            f64::from(clip_box.y_max),
        );
        let transformed = self.cur_transform() * rect.to_path(0.1);

        self.painter.push_clip_layer(&transformed);
        self.layer_count += 1;
    }

    fn pop_clip(&mut self) {
        self.painter.pop_layer();
        self.layer_count -= 1;
    }

    fn fill(&mut self, brush: Brush<'_>) {
        match brush {
            Brush::Solid {
                palette_index,
                alpha,
            } => {
                let color = self
                    .palette_index_to_color(palette_index, alpha)
                    .unwrap_or(AlphaColor::BLACK);

                self.painter.fill_solid(color);
            }
            Brush::LinearGradient {
                p0,
                p1,
                color_stops,
                extend,
            } => {
                let p0 = convert_point(p0);
                let p1 = convert_point(p1);
                let extend = convert_extend(extend);
                let stops = self.convert_stops(color_stops);

                if stops.len() == 1 {
                    self.painter.fill_solid(stops[0].color.to_alpha_color());
                } else {
                    let grad = Gradient {
                        kind: LinearGradientPosition { start: p0, end: p1 }.into(),
                        stops,
                        extend,
                        ..Default::default()
                    };
                    self.painter.set_paint_transform(self.cur_transform());
                    self.painter.fill_gradient(grad);
                }
            }
            Brush::RadialGradient {
                c0,
                r0,
                c1,
                r1,
                color_stops,
                extend,
            } => {
                // TODO: Radial gradients with negative r0.

                let p0 = convert_point(c0);
                let p1 = convert_point(c1);
                let extend = convert_extend(extend);
                let stops = self.convert_stops(color_stops);

                if r1 <= 0.0 || stops.len() == 1 {
                    self.painter.fill_solid(stops[0].color.to_alpha_color());

                    return;
                }

                let grad = Gradient {
                    kind: RadialGradientPosition {
                        start_center: p0,
                        start_radius: r0,
                        end_center: p1,
                        end_radius: r1,
                    }
                    .into(),
                    stops,
                    extend,
                    ..Default::default()
                };

                self.painter.set_paint_transform(self.cur_transform());
                self.painter.fill_gradient(grad);
            }
            Brush::SweepGradient {
                c0,
                start_angle,
                mut end_angle,
                color_stops,
                extend,
            } => {
                let p0 = convert_point(c0);
                let extend = convert_extend(extend);
                let stops = self.convert_stops(color_stops);

                if stops.len() == 1 {
                    self.painter.fill_solid(stops[0].color.to_alpha_color());

                    return;
                }

                if start_angle == end_angle {
                    match extend {
                        Extend::Pad => {
                            // Vello doesn't accept sweep gradient with same start and end
                            // angle, so add an artificial, small offset.
                            end_angle += 0.01;
                        }
                        _ => {
                            // Cannot be reached,
                            // see https://github.com/googlefonts/fontations/issues/1017.
                            unreachable!()
                        }
                    }
                }

                // We need to invert the direction of the gradient to bridge the gap between
                // peniko and COLR.
                let grad = Gradient {
                    kind: SweepGradientPosition {
                        center: Point::new(p0.x, -p0.y),
                        start_angle: start_angle.to_radians(),
                        end_angle: end_angle.to_radians(),
                    }
                    .into(),
                    stops,
                    extend,
                    ..Default::default()
                };

                let paint_transform = self.cur_transform() * Affine::scale_non_uniform(1.0, -1.0);

                self.painter.set_paint_transform(paint_transform);
                self.painter.fill_gradient(grad);
            }
        };
    }

    fn push_layer(&mut self, composite_mode: CompositeMode) {
        let blend_mode = match composite_mode {
            CompositeMode::Clear => BlendMode::new(Mix::Normal, Compose::Clear),
            CompositeMode::Src => BlendMode::new(Mix::Normal, Compose::Copy),
            CompositeMode::Dest => BlendMode::new(Mix::Normal, Compose::Dest),
            CompositeMode::SrcOver => BlendMode::new(Mix::Normal, Compose::SrcOver),
            CompositeMode::DestOver => BlendMode::new(Mix::Normal, Compose::DestOver),
            CompositeMode::SrcIn => BlendMode::new(Mix::Normal, Compose::SrcIn),
            CompositeMode::DestIn => BlendMode::new(Mix::Normal, Compose::DestIn),
            CompositeMode::SrcOut => BlendMode::new(Mix::Normal, Compose::SrcOut),
            CompositeMode::DestOut => BlendMode::new(Mix::Normal, Compose::DestOut),
            CompositeMode::SrcAtop => BlendMode::new(Mix::Normal, Compose::SrcAtop),
            CompositeMode::DestAtop => BlendMode::new(Mix::Normal, Compose::DestAtop),
            CompositeMode::Xor => BlendMode::new(Mix::Normal, Compose::Xor),
            CompositeMode::Plus => BlendMode::new(Mix::Normal, Compose::Plus),
            CompositeMode::Screen => BlendMode::new(Mix::Screen, Compose::SrcOver),
            CompositeMode::Overlay => BlendMode::new(Mix::Overlay, Compose::SrcOver),
            CompositeMode::Darken => BlendMode::new(Mix::Darken, Compose::SrcOver),
            CompositeMode::Lighten => BlendMode::new(Mix::Lighten, Compose::SrcOver),
            CompositeMode::ColorDodge => BlendMode::new(Mix::ColorDodge, Compose::SrcOver),
            CompositeMode::ColorBurn => BlendMode::new(Mix::ColorBurn, Compose::SrcOver),
            CompositeMode::HardLight => BlendMode::new(Mix::HardLight, Compose::SrcOver),
            CompositeMode::SoftLight => BlendMode::new(Mix::SoftLight, Compose::SrcOver),
            CompositeMode::Difference => BlendMode::new(Mix::Difference, Compose::SrcOver),
            CompositeMode::Exclusion => BlendMode::new(Mix::Exclusion, Compose::SrcOver),
            CompositeMode::Multiply => BlendMode::new(Mix::Multiply, Compose::SrcOver),
            CompositeMode::HslHue => BlendMode::new(Mix::Hue, Compose::SrcOver),
            CompositeMode::HslSaturation => BlendMode::new(Mix::Saturation, Compose::SrcOver),
            CompositeMode::HslColor => BlendMode::new(Mix::Color, Compose::SrcOver),
            CompositeMode::HslLuminosity => BlendMode::new(Mix::Luminosity, Compose::SrcOver),
            CompositeMode::Unknown => BlendMode::new(Mix::Normal, Compose::SrcOver),
        };

        self.painter.push_blend_layer(blend_mode);
        self.layer_count += 1;
    }

    fn pop_layer(&mut self) {
        self.painter.pop_layer();
        self.layer_count -= 1;
    }
}

fn convert_extend(extend: skrifa::color::Extend) -> Extend {
    match extend {
        skrifa::color::Extend::Pad => Extend::Pad,
        skrifa::color::Extend::Repeat => Extend::Repeat,
        skrifa::color::Extend::Reflect => Extend::Reflect,
        skrifa::color::Extend::Unknown => Extend::Pad,
    }
}

fn convert_point(point: skrifa::raw::types::Point<f32>) -> Point {
    Point::new(f64::from(point.x), f64::from(point.y))
}

pub(crate) fn convert_bounding_box(rect: BoundingBox<f32>) -> Rect {
    Rect::new(
        f64::from(rect.x_min),
        f64::from(rect.y_min),
        f64::from(rect.x_max),
        f64::from(rect.y_max),
    )
}
