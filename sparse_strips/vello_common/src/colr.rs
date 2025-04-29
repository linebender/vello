use crate::color::{ColorSpaceTag, HueDirection, Srgb};
use crate::glyph::OutlinePath;
use crate::kurbo::{Affine, BezPath, Point, Rect, Shape};
use crate::peniko;
use crate::peniko::{BlendMode, ColorStops, Compose, Extend, Mix};
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use skrifa::color::{Brush, ColorPainter, ColorStop, CompositeMode, Transform};
use skrifa::instance::LocationRef;
use skrifa::outline::DrawSettings;
use skrifa::raw::TableProvider;
use skrifa::raw::types::BoundingBox;
use skrifa::{FontRef, GlyphId, MetadataProvider};
use smallvec::SmallVec;
use vello_api::color::{AlphaColor, DynamicColor};
use vello_api::paint::Gradient;
use vello_api::peniko::GradientKind;

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
    /// Pop the last clip layer.
    fn pop_clip_layer(&mut self);
    /// Pop the last blend layer.
    fn pop_blend_layer(&mut self);
}

/// An abstraction for painting COLR glyphs.
pub struct ColrPainter<'a> {
    transforms: Vec<Affine>,
    font_ref: &'a FontRef<'a>,
    context_color: AlphaColor<Srgb>,
    painter: Box<&'a mut dyn ColrRenderer>,
    layer_count: u32,
}

impl Debug for ColrPainter<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ColrPainter()").finish()
    }
}

impl<'a> ColrPainter<'a> {
    /// Create a new COLR painter.
    ///
    /// `initial_transform` represents an initial transformation that should be applied
    /// to the whole glyph. By default, glyphs will be drawn in glyph space (i.e. with
    /// coordinates based on the units per em of the font).
    /// `context_color` is the color that should be assumed for fills with a palette index
    /// of `u16::MAX`.
    pub fn new(
        initial_transform: Affine,
        font_ref: &'a FontRef<'a>,
        context_color: AlphaColor<Srgb>,
        painter: &'a mut impl ColrRenderer,
    ) -> Self {
        Self {
            transforms: vec![initial_transform],
            font_ref,
            context_color,
            painter: Box::new(painter),
            layer_count: 0,
        }
    }

    fn cur_transform(&self) -> Affine {
        self.transforms.last().copied().unwrap_or_default()
    }

    fn palette_index_to_color(&self, palette_index: u16, alpha: f32) -> Option<AlphaColor<Srgb>> {
        if palette_index != u16::MAX {
            let color =
                self.font_ref.cpal().ok()?.color_records_array()?.ok()?[palette_index as usize];

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

        ColorStops(stops)
    }

    /// Get the number of remaining layers in the painter.
    pub fn remaining_layers(&self) -> u32 {
        // In certain malformed fonts (i.e. if there is a cycle), skrifa will not
        // ensure that the push/pop count is the same, so the client needs to manually
        // pop any remaining layers.
        self.layer_count
    }

    /// A shorthand for `std::mem::drop`.
    pub fn finish(self) {}
}

impl ColorPainter for ColrPainter<'_> {
    fn push_transform(&mut self, t: Transform) {
        let affine = Affine::new([
            t.xx as f64,
            t.yx as f64,
            t.xy as f64,
            t.yy as f64,
            t.dx as f64,
            t.dy as f64,
        ]);
        self.transforms.push(self.cur_transform() * affine);
    }

    fn pop_transform(&mut self) {
        self.transforms.pop();
    }

    fn push_clip_glyph(&mut self, glyph_id: GlyphId) {
        let mut outline_builder = OutlinePath::new(false);

        let outline_glyphs = self.font_ref.outline_glyphs();
        let Some(outline_glyph) = outline_glyphs.get(glyph_id) else {
            return;
        };
        let _ = outline_glyph.draw(
            DrawSettings::unhinted(skrifa::instance::Size::unscaled(), LocationRef::default()),
            &mut outline_builder,
        );

        let finished = outline_builder.0;
        let transformed = self.cur_transform() * finished;

        self.painter.push_clip_layer(&transformed);
        self.layer_count += 1;
    }

    fn push_clip_box(&mut self, clip_box: BoundingBox<f32>) {
        let rect = Rect::new(
            clip_box.x_min as f64,
            clip_box.y_min as f64,
            clip_box.x_max as f64,
            clip_box.y_max as f64,
        );
        let transformed = self.cur_transform() * rect.to_path(0.1);

        self.painter.push_clip_layer(&transformed);
        self.layer_count += 1;
    }

    fn pop_clip(&mut self) {
        self.painter.pop_clip_layer();
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
                        kind: GradientKind::Linear { start: p0, end: p1 },
                        stops,
                        transform: self.cur_transform(),
                        extend,
                        ..Default::default()
                    };

                    self.painter.fill_gradient(grad);
                }
            }
            Brush::RadialGradient {
                c0,
                mut r0,
                c1,
                mut r1,
                color_stops,
                extend,
            } => {
                // TODO: Radial gradients with negative r0.

                let p0 = convert_point(c0);
                let p1 = convert_point(c1);
                let extend = convert_extend(extend);
                let mut stops = self.convert_stops(color_stops);

                if r1 <= 0.0 || stops.len() == 1 {
                    self.painter.fill_solid(stops[0].color.to_alpha_color());

                    return;
                }

                let grad = Gradient {
                    kind: GradientKind::Radial {
                        start_center: p0,
                        start_radius: r0,
                        end_center: p1,
                        end_radius: r1,
                    },
                    stops,
                    transform: self.cur_transform(),
                    extend,
                    ..Default::default()
                };

                self.painter.fill_gradient(grad);
            }
            Brush::SweepGradient {
                c0,
                mut start_angle,
                mut end_angle,
                color_stops,
                extend,
            } => {
                let p0 = convert_point(c0);
                let extend = convert_extend(extend);
                let mut stops = self.convert_stops(color_stops);

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

                let grad = Gradient {
                    kind: GradientKind::Sweep {
                        center: p0,
                        start_angle,
                        end_angle,
                    },
                    stops,
                    transform: self.cur_transform(),
                    extend,
                    ..Default::default()
                };

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
        self.painter.pop_blend_layer();
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
    Point::new(point.x as f64, point.y as f64)
}

pub(crate) fn convert_bounding_box(rect: BoundingBox<f32>) -> Rect {
    Rect::new(
        rect.x_min as f64,
        rect.y_min as f64,
        rect.x_max as f64,
        rect.y_max as f64,
    )
}
