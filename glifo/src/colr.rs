// Copyright 2025 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Drawing COLR glyphs.

use crate::atlas::commands::AtlasPaint;
use crate::color::Srgb;
use crate::color::{AlphaColor, DynamicColor};
use crate::glyph::{GlyphColr, OutlinePath};
use crate::interface::DrawSink;
use crate::kurbo::{Affine, Point, Rect, Shape};
use crate::peniko::{self, BlendMode, ColorStops, Compose, Extend, Gradient, Mix};
use crate::util::FloatExt;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Debug;
use peniko::{LinearGradientPosition, RadialGradientPosition, SweepGradientPosition};
use skrifa::color::{Brush, ColorPainter, ColorStop, CompositeMode, Transform};
use skrifa::instance::LocationRef;
use skrifa::outline::{DrawSettings, pen::ControlBoundsPen};
use skrifa::raw::TableProvider;
use skrifa::raw::types::BoundingBox;
use skrifa::{FontRef, GlyphId, MetadataProvider};
use smallvec::SmallVec;

trait ColrDrawSinkExt: DrawSink {
    fn fill_with_paint(&mut self, rect: &Rect, paint: AtlasPaint) {
        self.set_paint(paint);
        self.fill_rect(rect);
    }

    fn fill_solid(&mut self, rect: &Rect, color: AlphaColor<Srgb>) {
        self.fill_with_paint(rect, AtlasPaint::Solid(color));
    }

    fn fill_gradient(&mut self, rect: &Rect, gradient: Gradient) {
        self.fill_with_paint(rect, AtlasPaint::Gradient(gradient));
    }
}

impl<T: DrawSink + ?Sized> ColrDrawSinkExt for T {}

/// An abstraction for painting COLR glyphs.
pub(crate) struct ColrPainter<'a> {
    transforms: Vec<Affine>,
    colr_glyph: &'a GlyphColr<'a>,
    context_color: AlphaColor<Srgb>,
    painter: &'a mut dyn DrawSink,
    stack: Vec<ColrStackEntry>,
    skip_blend_layers: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColrStackEntry {
    ClipPath,
    BlendLayer,
}

impl Debug for ColrPainter<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ColrPainter()").finish()
    }
}

impl<'a> ColrPainter<'a> {
    /// Create a new COLR painter.
    pub(crate) fn new(
        colr_glyph: &'a GlyphColr<'a>,
        context_color: AlphaColor<Srgb>,
        painter: &'a mut dyn DrawSink,
    ) -> Self {
        Self {
            transforms: vec![colr_glyph.draw_transform],
            colr_glyph,
            context_color,
            painter,
            stack: Vec::new(),
            // In case the emoji doesn't use non-default blending, we can ignore layers
            // completely and use src-over compositing throughout.
            skip_blend_layers: !colr_glyph.has_non_default_blend,
        }
    }

    /// Paint the underlying glyph.
    pub(crate) fn paint(&mut self) {
        let skrifa_glyph = self.colr_glyph.skrifa_glyph.clone();
        let location_ref = self.colr_glyph.location;
        // Ignore errors for now.
        let _ = skrifa_glyph.paint(location_ref, self);

        // In certain malformed fonts (i.e. if there is a cycle), skrifa will not
        // ensure that the push/pop count is the same, so we pop the remaining ones here.
        while let Some(entry) = self.stack.pop() {
            match entry {
                ColrStackEntry::ClipPath => self.painter.pop_clip_path(),
                ColrStackEntry::BlendLayer => self.painter.pop_layer(),
            }
        }
    }

    fn cur_transform(&self) -> Affine {
        self.transforms.last().copied().unwrap_or_default()
    }

    fn palette_index_to_color(&self, palette_index: u16, alpha: f32) -> Option<AlphaColor<Srgb>> {
        if palette_index != u16::MAX {
            let color = self
                .colr_glyph
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

    fn push_clip(&mut self, clip: &crate::kurbo::BezPath) {
        self.painter.push_clip_path(clip);
        self.stack.push(ColrStackEntry::ClipPath);
    }

    fn pop_stack_entry(&mut self, expected: ColrStackEntry) -> bool {
        if self.stack.last().copied() == Some(expected) {
            self.stack.pop();
            true
        } else {
            false
        }
    }
}

pub(crate) struct ColrGlyphInfo {
    /// A conservative bounding box of the glyph.
    pub(crate) bbox: Option<Rect>,
    /// Whether the glyph uses any non-default blending.
    pub(crate) has_non_default_blend: bool,
}

pub(crate) fn get_colr_info<'a>(
    font_ref: &'a FontRef<'a>,
    color_glyph: &skrifa::color::ColorGlyph<'a>,
    location: LocationRef<'a>,
) -> ColrGlyphInfo {
    let mut extractor = GlyphInfoExtractor::new(font_ref, location);
    let _ = color_glyph.paint(location, &mut extractor);
    extractor.finish()
}

struct GlyphInfoExtractor<'a> {
    transforms: Vec<Affine>,
    clip_stack: Vec<Rect>,
    coarse_bbox: Option<Rect>,
    has_non_default_blend: bool,
    font_ref: &'a FontRef<'a>,
    location: LocationRef<'a>,
}

impl<'a> GlyphInfoExtractor<'a> {
    fn new(font_ref: &'a FontRef<'a>, location: LocationRef<'a>) -> Self {
        Self {
            transforms: vec![Affine::IDENTITY],
            clip_stack: Vec::new(),
            coarse_bbox: None,
            has_non_default_blend: false,
            font_ref,
            location,
        }
    }

    fn cur_transform(&self) -> Affine {
        self.transforms.last().copied().unwrap_or_default()
    }

    fn push_clip_bbox(&mut self, clip_bbox: Rect) {
        let active = self
            .clip_stack
            .last()
            .copied()
            .map_or(clip_bbox, |parent| parent.intersect(clip_bbox));
        self.coarse_bbox = Some(
            self.coarse_bbox
                .map_or(active, |coarse_bbox| coarse_bbox.union(active)),
        );
        self.clip_stack.push(active);
    }

    fn transform_rect(&self, rect: Rect) -> Rect {
        self.cur_transform().transform_rect_bbox(rect)
    }

    fn finish(self) -> ColrGlyphInfo {
        ColrGlyphInfo {
            bbox: self.coarse_bbox,
            has_non_default_blend: self.has_non_default_blend,
        }
    }
}

impl ColorPainter for ColrPainter<'_> {
    fn push_transform(&mut self, t: Transform) {
        self.transforms
            .push(self.cur_transform() * convert_affine(t));
    }

    fn pop_transform(&mut self) {
        self.transforms.pop();
    }

    fn push_clip_glyph(&mut self, glyph_id: GlyphId) {
        let mut outline_builder = OutlinePath::new();

        let outline_glyphs = self.colr_glyph.font_ref.outline_glyphs();
        let Some(outline_glyph) = outline_glyphs.get(glyph_id) else {
            return;
        };

        let _ = outline_glyph.draw(
            DrawSettings::unhinted(skrifa::instance::Size::unscaled(), self.colr_glyph.location),
            &mut outline_builder,
        );

        let finished = outline_builder.path;
        let transformed = self.cur_transform() * finished;

        self.push_clip(&transformed);
    }

    fn push_clip_box(&mut self, clip_box: BoundingBox<f32>) {
        let rect = Rect::new(
            f64::from(clip_box.x_min),
            f64::from(clip_box.y_min),
            f64::from(clip_box.x_max),
            f64::from(clip_box.y_max),
        );
        let transformed = self.cur_transform() * rect.to_path(0.1);

        self.push_clip(&transformed);
    }

    fn pop_clip(&mut self) {
        if self.pop_stack_entry(ColrStackEntry::ClipPath) {
            self.painter.pop_clip_path();
        }
    }

    fn fill(&mut self, brush: Brush<'_>) {
        // Ceil so that we don't apply unnecessary anti-aliasing in case the
        // glyph area is at a sub-pixel position.
        let fill_rect = &self.colr_glyph.area.ceil();

        match brush {
            Brush::Solid {
                palette_index,
                alpha,
            } => {
                let color = self
                    .palette_index_to_color(palette_index, alpha)
                    .unwrap_or(AlphaColor::BLACK);

                self.painter.fill_solid(fill_rect, color);
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
                    self.painter
                        .fill_solid(fill_rect, stops[0].color.to_alpha_color());
                } else {
                    let grad = Gradient {
                        kind: LinearGradientPosition { start: p0, end: p1 }.into(),
                        stops,
                        extend,
                        ..Default::default()
                    };
                    self.painter.set_paint_transform(self.cur_transform());
                    self.painter.fill_gradient(fill_rect, grad);
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
                    self.painter
                        .fill_solid(fill_rect, stops[0].color.to_alpha_color());

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
                self.painter.fill_gradient(fill_rect, grad);
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
                    self.painter
                        .fill_solid(fill_rect, stops[0].color.to_alpha_color());

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
                self.painter.fill_gradient(fill_rect, grad);
            }
        };
    }

    fn push_layer(&mut self, composite_mode: CompositeMode) {
        let blend_mode = convert_composite_mode(composite_mode);

        if !self.skip_blend_layers {
            self.painter.push_blend_layer(blend_mode);
            self.stack.push(ColrStackEntry::BlendLayer);
        }
    }

    fn pop_layer(&mut self) {
        if !self.skip_blend_layers && self.pop_stack_entry(ColrStackEntry::BlendLayer) {
            self.painter.pop_layer();
        }
    }
}

impl ColorPainter for GlyphInfoExtractor<'_> {
    fn push_transform(&mut self, t: Transform) {
        self.transforms
            .push(self.cur_transform() * convert_affine(t));
    }

    fn pop_transform(&mut self) {
        self.transforms.pop();
    }

    fn push_clip_glyph(&mut self, glyph_id: GlyphId) {
        let mut outline_bbox = ControlBoundsPen::default();

        let outline_glyphs = self.font_ref.outline_glyphs();
        let Some(outline_glyph) = outline_glyphs.get(glyph_id) else {
            return;
        };

        let _ = outline_glyph.draw(
            DrawSettings::unhinted(skrifa::instance::Size::unscaled(), self.location),
            &mut outline_bbox,
        );

        if let Some(outline_bbox) = outline_bbox.bounding_box().map(convert_bounding_box) {
            self.push_clip_bbox(self.transform_rect(outline_bbox));
        }
    }

    fn push_clip_box(&mut self, clip_box: BoundingBox<f32>) {
        self.push_clip_bbox(self.transform_rect(convert_bounding_box(clip_box)));
    }

    fn pop_clip(&mut self) {
        self.clip_stack.pop();
    }

    fn fill(&mut self, _brush: Brush<'_>) {}

    fn push_layer(&mut self, composite_mode: CompositeMode) {
        self.has_non_default_blend |=
            convert_composite_mode(composite_mode) != BlendMode::default();
    }

    fn pop_layer(&mut self) {}
}

fn convert_composite_mode(composite_mode: CompositeMode) -> BlendMode {
    match composite_mode {
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
        CompositeMode::Unknown => BlendMode::default(),
    }
}

fn convert_affine(transform: Transform) -> Affine {
    Affine::new([
        f64::from(transform.xx),
        f64::from(transform.yx),
        f64::from(transform.xy),
        f64::from(transform.yy),
        f64::from(transform.dx),
        f64::from(transform.dy),
    ])
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
