// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use crate::fine::Fine;
use alloc::vec;
use alloc::vec::Vec;
use vello_common::coarse::Wide;
use vello_common::encode::{EncodeExt, EncodedPaint};
use vello_common::flatten::Line;
use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder, PreparedGlyph};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use vello_common::mask::Mask;
use vello_common::paint::{Paint, PaintType};
use vello_common::peniko::Font;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Fill, Mix};
use vello_common::pixmap::Pixmap;
use vello_common::strip::Strip;
use vello_common::tile::Tiles;
use vello_common::{flatten, strip};

pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;
/// A render context.
#[derive(Debug)]
pub struct RenderContext {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) wide: Wide,
    pub(crate) alphas: Vec<u8>,
    pub(crate) line_buf: Vec<Line>,
    pub(crate) tiles: Tiles,
    pub(crate) strip_buf: Vec<Strip>,
    pub(crate) paint: PaintType,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) encoded_paints: Vec<EncodedPaint>,
}

impl RenderContext {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        let wide = Wide::new(width, height);

        let alphas = vec![];
        let line_buf = vec![];
        let tiles = Tiles::new();
        let strip_buf = vec![];

        let transform = Affine::IDENTITY;
        let fill_rule = Fill::NonZero;
        let paint = BLACK.into();
        let stroke = Stroke {
            width: 1.0,
            join: Join::Bevel,
            start_cap: Cap::Butt,
            end_cap: Cap::Butt,
            ..Default::default()
        };
        let encoded_paints = vec![];

        Self {
            width,
            height,
            wide,
            alphas,
            line_buf,
            tiles,
            strip_buf,
            transform,
            paint,
            fill_rule,
            stroke,
            encoded_paints,
        }
    }

    fn encode_current_paint(&mut self) -> Paint {
        match self.paint.clone() {
            PaintType::Solid(s) => s.into(),
            PaintType::Gradient(mut g) => {
                // TODO: Add caching?
                g.transform = self.transform * g.transform;
                g.encode_into(&mut self.encoded_paints)
            }
            PaintType::Image(mut i) => {
                i.transform = self.transform * i.transform;
                i.encode_into(&mut self.encoded_paints)
            }
        }
    }

    /// Fill a path.
    pub fn fill_path(&mut self, path: &BezPath) {
        flatten::fill(path, self.transform, &mut self.line_buf);
        let paint = self.encode_current_paint();
        self.render_path(self.fill_rule, paint);
    }

    /// Stroke a path.
    pub fn stroke_path(&mut self, path: &BezPath) {
        flatten::stroke(path, &self.stroke, self.transform, &mut self.line_buf);
        let paint = self.encode_current_paint();
        self.render_path(Fill::NonZero, paint);
    }

    /// Fill a rectangle.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.fill_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Stroke a rectangle.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    pub fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self> {
        GlyphRunBuilder::new(font.clone(), self.transform, self)
    }

    /// Push a new layer with the given properties.
    ///
    /// Note that the mask, if provided, needs to have the same size as the render context. Otherwise,
    /// it will be ignored. In addition to that, the mask will not be affected by the current
    /// transformation matrix in place.
    pub fn push_layer(
        &mut self,
        clip_path: Option<&BezPath>,
        blend_mode: Option<BlendMode>,
        opacity: Option<u8>,
        mask: Option<Mask>,
    ) {
        let clip = if let Some(c) = clip_path {
            flatten::fill(c, self.transform, &mut self.line_buf);
            self.make_strips(self.fill_rule);
            Some((self.strip_buf.as_slice(), self.fill_rule))
        } else {
            None
        };

        let mask = mask.and_then(|m| {
            if m.width() != self.width || m.height() != self.height {
                None
            } else {
                Some(m)
            }
        });

        self.wide.push_layer(
            clip,
            blend_mode.unwrap_or(BlendMode::new(Mix::Normal, Compose::SrcOver)),
            mask,
            opacity.unwrap_or(255),
        );
    }

    /// Push a new clip layer.
    pub fn push_clip_layer(&mut self, path: &BezPath) {
        self.push_layer(Some(path), None, None, None);
    }

    /// Push a new blend layer.
    pub fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.push_layer(None, Some(blend_mode), None, None);
    }

    /// Push a new opacity layer.
    pub fn push_opacity_layer(&mut self, opacity: u8) {
        self.push_layer(None, None, Some(opacity), None);
    }

    /// Push a new mask layer.
    ///
    /// Note that the mask, if provided, needs to have the same size as the render context. Otherwise,
    /// it will be ignored. In addition to that, the mask will not be affected by the current
    /// transformation matrix in place.
    pub fn push_mask_layer(&mut self, mask: Mask) {
        self.push_layer(None, None, None, Some(mask));
    }

    /// Pop the last-pushed layer.
    pub fn pop_layer(&mut self) {
        self.wide.pop_layer();
    }

    /// Set the current stroke.
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.stroke = stroke;
    }

    /// Set the current paint.
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.paint = paint.into();
    }

    /// Set the current fill rule.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.fill_rule = fill_rule;
    }

    /// Set the current transform.
    pub fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Reset the current transform.
    pub fn reset_transform(&mut self) {
        self.transform = Affine::IDENTITY;
    }

    /// Reset the render context.
    pub fn reset(&mut self) {
        self.line_buf.clear();
        self.tiles.reset();
        self.alphas.clear();
        self.strip_buf.clear();
        self.wide.reset();
    }

    /// Render the current context into a pixmap.
    pub fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
        assert!(
            !self.wide.has_layers(),
            "some layers haven't been popped yet"
        );

        let mut fine = Fine::new(pixmap.width, pixmap.height);

        let width_tiles = self.wide.width_tiles();
        let height_tiles = self.wide.height_tiles();
        for y in 0..height_tiles {
            for x in 0..width_tiles {
                let wtile = self.wide.get(x, y);
                fine.set_coords(x, y);

                fine.clear(wtile.bg.as_premul_rgba8().to_u8_array());
                for cmd in &wtile.cmds {
                    fine.run_cmd(cmd, &self.alphas, &self.encoded_paints);
                }
                fine.pack(&mut pixmap.buf);
            }
        }
    }

    /// Return the width of the pixmap.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Return the height of the pixmap.
    pub fn height(&self) -> u16 {
        self.height
    }

    // Assumes that `line_buf` contains the flattened path.
    fn render_path(&mut self, fill_rule: Fill, paint: Paint) {
        self.make_strips(fill_rule);
        self.wide.generate(&self.strip_buf, fill_rule, paint);
    }

    fn make_strips(&mut self, fill_rule: Fill) {
        self.tiles
            .make_tiles(&self.line_buf, self.width, self.height);
        self.tiles.sort_tiles();

        strip::render(
            &self.tiles,
            &mut self.strip_buf,
            &mut self.alphas,
            fill_rule,
            &self.line_buf,
        );
    }
}

impl GlyphRenderer for RenderContext {
    fn fill_glyph(&mut self, glyph: PreparedGlyph<'_>) {
        match glyph {
            PreparedGlyph::Outline(glyph) => {
                flatten::fill(glyph.path, glyph.transform, &mut self.line_buf);
                let paint = self.encode_current_paint();
                self.render_path(Fill::NonZero, paint);
            }
        }
    }

    fn stroke_glyph(&mut self, glyph: PreparedGlyph<'_>) {
        match glyph {
            PreparedGlyph::Outline(glyph) => {
                flatten::stroke(
                    glyph.path,
                    &self.stroke,
                    glyph.transform,
                    &mut self.line_buf,
                );
                let paint = self.encode_current_paint();
                self.render_path(Fill::NonZero, paint);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::RenderContext;
    use vello_common::kurbo::{Rect, Shape};

    #[test]
    fn reset_render_context() {
        let mut ctx = RenderContext::new(100, 100);
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);

        ctx.fill_rect(&rect);

        assert!(!ctx.line_buf.is_empty());
        assert!(!ctx.strip_buf.is_empty());
        assert!(!ctx.alphas.is_empty());

        ctx.reset();

        assert!(ctx.line_buf.is_empty());
        assert!(ctx.strip_buf.is_empty());
        assert!(ctx.alphas.is_empty());
    }

    #[test]
    fn clip_overflow() {
        let mut ctx = RenderContext::new(100, 100);

        ctx.alphas
            .extend(core::iter::repeat_n(255, u16::MAX as usize + 1));

        ctx.push_clip_layer(&Rect::new(20.0, 20.0, 180.0, 180.0).to_path(0.1));
        ctx.pop_layer();
    }
}
