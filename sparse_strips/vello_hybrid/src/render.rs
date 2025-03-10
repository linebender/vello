// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.

use crate::gpu::{GpuStrip, RenderData};
use kurbo::{Affine, BezPath, Cap, Join, Rect, Shape, Stroke};
use peniko::color::palette::css::BLACK;
use peniko::{BlendMode, Compose, Fill, Mix};
use vello_common::coarse::{WIDE_TILE_WIDTH, Wide};
use vello_common::flatten::Line;
use vello_common::paint::Paint;
use vello_common::strip::{STRIP_HEIGHT, Strip};
use vello_common::tile::Tiles;
use vello_common::{flatten, strip};
use vello_cpu::fine::Fine;
use vello_cpu::pixmap::Pixmap;
use vello_cpu::util::ColorExt;

/// Default tolerance for curve flattening
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// A render context for hybrid CPU/GPU rendering.
///
/// This context maintains the state for path rendering and manages the rendering
/// pipeline from paths to strips that can be rendered by the GPU.
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
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        // TODO: Use u16 for width/height everywhere else, too.
        let wide = Wide::new(width.into(), height.into());

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
            width: width.into(),
            height: height.into(),
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

    /// Fill a path with the current paint and fill rule.
    pub fn fill_path(&mut self, path: &BezPath) {
        flatten::fill(path, self.transform, &mut self.line_buf);
        self.render_path(self.fill_rule, self.paint.clone());
    }

    /// Stroke a path with the current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        flatten::stroke(path, &self.stroke, self.transform, &mut self.line_buf);
        self.render_path(Fill::NonZero, self.paint.clone());
    }

    /// Fill a rectangle with the current paint and fill rule.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.fill_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Set the blend mode for subsequent rendering operations.
    pub fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        self.blend_mode = blend_mode;
    }

    /// Set the stroke settings for subsequent stroke operations.
    pub fn set_stroke(&mut self, stroke: Stroke) {
        self.stroke = stroke;
    }

    /// Set the paint for subsequent rendering operations.
    pub fn set_paint(&mut self, paint: Paint) {
        self.paint = paint;
    }

    /// Set the fill rule for subsequent fill operations.
    pub fn set_fill_rule(&mut self, fill_rule: Fill) {
        self.fill_rule = fill_rule;
    }

    /// Set the transform for subsequent rendering operations.
    pub fn set_transform(&mut self, transform: Affine) {
        self.transform = transform;
    }

    /// Reset the transform to identity.
    pub fn reset_transform(&mut self) {
        self.transform = Affine::IDENTITY;
    }

    /// Reset all rendering state to default values.
    pub fn reset(&mut self) {
        self.wide.reset();
    }

    /// Render the current content to a pixmap.
    pub fn render_to_pixmap(&self, pixmap: &mut Pixmap) {
        // TODO: Use u16 here, too, instead of casting.
        let mut fine = Fine::new(
            pixmap.width as usize,
            pixmap.height as usize,
            &mut pixmap.buf,
        );

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

    /// Get the width of the render context.
    pub fn width(&self) -> u16 {
        self.width as u16
    }

    /// Get the height of the render context.
    pub fn height(&self) -> u16 {
        self.height as u16
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

impl RenderContext {
    /// Prepares render data from the current context for GPU rendering
    ///
    /// This method converts the rendering context's state into a format
    /// suitable for GPU rendering, including strips and alpha values.
    pub fn prepare_render_data(&self) -> RenderData {
        let mut strips: Vec<GpuStrip> = Vec::new();
        let width_tiles = (self.width).div_ceil(WIDE_TILE_WIDTH);
        let height_tiles = (self.height).div_ceil(STRIP_HEIGHT);
        for y in 0..height_tiles {
            for x in 0..width_tiles {
                let tile = &self.wide.tiles[y * width_tiles + x];
                let tile_x = x * WIDE_TILE_WIDTH;
                let tile_y = y * STRIP_HEIGHT;
                let bg = tile.bg.to_rgba8().to_u32();
                if bg != 0 {
                    let strip = GpuStrip {
                        x: tile_x as u16,
                        y: tile_y as u16,
                        width: WIDE_TILE_WIDTH as u16,
                        dense_width: 0,
                        col: 0,
                        rgba: bg,
                    };
                    strips.push(strip);
                }
                for cmd in &tile.cmds {
                    match cmd {
                        vello_common::coarse::Cmd::Fill(fill) => {
                            let color: peniko::color::AlphaColor<peniko::color::Srgb> =
                                match fill.paint {
                                    Paint::Solid(color) => color,
                                    _ => peniko::color::AlphaColor::TRANSPARENT,
                                };
                            let strip = GpuStrip {
                                x: (tile_x as u32 + fill.x) as u16,
                                y: tile_y as u16,
                                width: fill.width as u16,
                                dense_width: 0,
                                col: 0,
                                rgba: color.to_rgba8().to_u32(),
                            };
                            strips.push(strip);
                        }
                        vello_common::coarse::Cmd::AlphaFill(cmd_strip) => {
                            let color: peniko::color::AlphaColor<peniko::color::Srgb> =
                                match cmd_strip.paint {
                                    Paint::Solid(color) => color,
                                    _ => peniko::color::AlphaColor::TRANSPARENT,
                                };
                            let strip = GpuStrip {
                                x: (tile_x as u32 + cmd_strip.x) as u16,
                                y: tile_y as u16,
                                width: cmd_strip.width as u16,
                                dense_width: cmd_strip.width as u16,
                                col: cmd_strip.alpha_ix as u32,
                                rgba: color.to_rgba8().to_u32(),
                            };
                            strips.push(strip);
                        }
                    }
                }
            }
        }
        RenderData {
            strips,
            alphas: self.alphas.clone(),
        }
    }
}
