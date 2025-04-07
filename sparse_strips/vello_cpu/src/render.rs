// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use crate::fine::Fine;
use vello_common::coarse::{SceneState, Wide};
use vello_common::flatten::Line;
use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder, PreparedGlyph};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use vello_common::paint::Paint;
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
    pub(crate) paint: Paint,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
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
        let blend_mode = BlendMode::new(Mix::Normal, Compose::SrcOver);

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
            blend_mode,
        }
    }

    /// Save the current scene state.
    pub fn save(&mut self) {
        self.wide.state_stack.push(SceneState { n_clip: 0 });
    }

    /// Restore the previous scene state.
    pub fn restore(&mut self) {
        self.wide.pop_clips();
        self.wide.state_stack.pop();
    }

    /// Fill a path.
    pub fn fill_path(&mut self, path: &BezPath) {
        flatten::fill(path, self.transform, &mut self.line_buf);
        self.render_path(self.fill_rule, self.paint.clone());
    }

    /// Stroke a path.
    pub fn stroke_path(&mut self, path: &BezPath) {
        flatten::stroke(path, &self.stroke, self.transform, &mut self.line_buf);
        self.render_path(Fill::NonZero, self.paint.clone());
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

    /// Clip a path.
    pub fn clip(&mut self, path: &BezPath) {
        flatten::fill(path, self.transform, &mut self.line_buf);
        self.make_strips(self.fill_rule);
        let strips = core::mem::take(&mut self.strip_buf);
        self.wide.push_clip(strips, self.fill_rule);
    }

    /// Set the current blend mode.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.blend_mode = blend_mode;
    }

    /// Set the current stroke.
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.stroke = stroke;
    }

    /// Set the current paint.
    pub fn set_paint(&mut self, paint: Paint) {
        self.paint = paint;
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
        if let Some(state) = self.wide.state_stack.last() {
            if state.n_clip > 0 {
                panic!("All clips must be popped before rendering");
            }
        }
        let mut fine = Fine::new(pixmap.width, pixmap.height);

        let width_tiles = self.wide.width_tiles();
        let height_tiles = self.wide.height_tiles();
        for y in 0..height_tiles {
            for x in 0..width_tiles {
                let wtile = self.wide.get(x, y);

                fine.clear(wtile.bg.to_u8_array());
                for cmd in &wtile.cmds {
                    fine.run_cmd(cmd, &self.alphas);
                }
                fine.pack(x, y, &mut pixmap.buf);
            }
        }
    }

    /// Finish the coarse rasterization prior to fine rendering.
    ///
    /// This method is called when the render context is finished with rendering.
    /// It pops all the clips from the wide tiles.
    pub fn finish(&mut self) {
        self.wide.pop_clips();
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
                self.render_path(Fill::NonZero, self.paint.clone());
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
                self.render_path(Fill::NonZero, self.paint.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::RenderContext;
    use vello_common::kurbo::Rect;

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
}
