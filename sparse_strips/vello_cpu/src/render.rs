// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use peniko::{BlendMode, Compose, Fill, Mix};
use peniko::color::palette::css::BLACK;
use vello_common::coarse::Wide;
use vello_common::{flatten, strip};
use vello_common::flatten::Line;
use vello_common::paint::Paint;
use vello_common::strip::Strip;
use vello_common::tile::Tiles;
use crate::fine::Fine;
use crate::pixmap::Pixmap;
use crate::util::ColorExt;

pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// A render context.
#[derive(Debug)]
pub struct RenderContext {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) wide: Wide,
    pub(crate) alphas: Vec<u32>,
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
    /// Create a new render context.
    pub fn new(width: usize, height: usize) -> Self {
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

    /// Fill a path.
    pub fn fill_path(&mut self, path: &BezPath) {
        flatten::fill(&path, self.transform, &mut self.line_buf);
        self.render_path(self.fill_rule, self.paint.clone());
    }

    /// Stroke a path.
    pub fn stroke_path(&mut self, path: &BezPath) {
        flatten::stroke(&path, &self.stroke, self.transform, &mut self.line_buf);
        self.render_path(Fill::NonZero, self.paint.clone());
    }

    /// Fill a rectangle.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.fill_path(&rect.to_path(DEFAULT_TOLERANCE).into());
    }

    /// Stroke a rectangle.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE).into());
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
        self.wide.reset();
    }

    /// Render the current context into a pixmap.
    pub fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
        let mut fine = Fine::new(pixmap.width, pixmap.height, &mut pixmap.buf);

        let width_tiles = self.wide.width_tiles();
        let height_tiles = self.wide.height_tiles();
        for y in 0..height_tiles {
            for x in 0..width_tiles {
                let tile = self.wide.get(x, y);

                fine.clear(tile.bg.premultiply().to_rgba8_fast());
                for cmd in &tile.cmds {
                    fine.run_cmd(cmd, &self.alphas);
                }
                fine.pack(x, y);
            }
        }
    }

    /// Return the width of the pixmap.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Return the height of the pixmap.
    pub fn height(&self) -> usize {
        self.height
    }

    // Assumes that `line_buf` contains the flattened path.
    fn render_path(&mut self, fill_rule: Fill, paint: Paint) {
        self.tiles.make_tiles(&self.line_buf);
        self.tiles.sort_tiles();

        strip::render(
            &self.tiles,
            &mut self.strip_buf,
            &mut self.alphas,
            fill_rule,
        );

        self.wide.generate(&self.strip_buf, fill_rule, paint);
    }
}