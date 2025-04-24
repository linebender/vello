// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Basic render operations.
extern crate alloc;
extern crate std;
use alloc::collections::VecDeque;
use std::println;

use crate::render::{GpuStrip, RenderData};
use alloc::vec;
use alloc::vec::Vec;
use vello_common::coarse::{Wide, WideTile};
use vello_common::encode::{EncodeExt, EncodedPaint};
use vello_common::flatten::Line;
use vello_common::glyph::{GlyphRenderer, GlyphRunBuilder, PreparedGlyph};
use vello_common::kurbo::{Affine, BezPath, Cap, Join, Point, Rect, Shape, Stroke};
use vello_common::paint::{IndexedPaint, Paint, PaintType};
use vello_common::peniko::Font;
use vello_common::peniko::color::palette::css::BLACK;
use vello_common::peniko::{BlendMode, Compose, Fill, Mix};
use vello_common::strip::Strip;
use vello_common::tile::{Tile, Tiles};
use vello_common::{flatten, strip};

/// Default tolerance for curve flattening
pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

/// A render state which contains the style properties for path rendering and
/// the current transform.
#[derive(Debug)]
struct RenderState {
    pub(crate) paint: PaintType,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
}

/// A render context for hybrid CPU/GPU rendering.
///
/// This context maintains the state for path rendering and manages the rendering
/// pipeline from paths to strips that can be rendered by the GPU.
#[derive(Debug)]
pub struct Scene {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) wide: Wide,
    pub(crate) alphas: Vec<u8>,
    pub(crate) line_buf: Vec<Line>,
    pub(crate) tiles: Tiles,
    pub(crate) strip_buf: Vec<Strip>,
    pub(crate) encoded_paints: Vec<EncodedPaint>,
    pub(crate) upload_paints: VecDeque<IndexedPaint>,
    pub(crate) paint: PaintType,
    pub(crate) stroke: Stroke,
    pub(crate) transform: Affine,
    pub(crate) fill_rule: Fill,
    pub(crate) blend_mode: BlendMode,
}

impl Scene {
    /// Create a new render context with the given width and height in pixels.
    pub fn new(width: u16, height: u16) -> Self {
        let render_state = Self::default_render_state();
        Self {
            width,
            height,
            wide: Wide::new(width, height),
            alphas: vec![],
            line_buf: vec![],
            tiles: Tiles::new(),
            strip_buf: vec![],
            encoded_paints: vec![],
            upload_paints: VecDeque::new(),
            paint: render_state.paint,
            stroke: render_state.stroke,
            transform: render_state.transform,
            fill_rule: render_state.fill_rule,
            blend_mode: render_state.blend_mode,
        }
    }

    /// Create default rendering state.
    fn default_render_state() -> RenderState {
        let transform = Affine::IDENTITY;
        let fill_rule = Fill::NonZero;
        let paint = PaintType::Solid(BLACK);
        let stroke = Stroke {
            width: 1.0,
            join: Join::Bevel,
            start_cap: Cap::Butt,
            end_cap: Cap::Butt,
            ..Default::default()
        };
        let blend_mode = BlendMode::new(Mix::Normal, Compose::SrcOver);
        RenderState {
            transform,
            fill_rule,
            paint,
            stroke,
            blend_mode,
        }
    }

    /// Fill a path with the current paint and fill rule.
    pub fn fill_path(&mut self, path: &BezPath) {
        flatten::fill(path, self.transform, &mut self.line_buf);
        let paint = self.encode_current_paint();
        self.render_path(self.fill_rule, paint);
    }

    /// Stroke a path with the current paint and stroke settings.
    pub fn stroke_path(&mut self, path: &BezPath) {
        flatten::stroke(path, &self.stroke, self.transform, &mut self.line_buf);
        let paint = self.encode_current_paint();
        self.render_path(Fill::NonZero, paint);
    }

    /// Fill a rectangle with the current paint and fill rule.
    pub fn fill_rect(&mut self, rect: &Rect) {
        self.fill_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Stroke a rectangle with the current paint and stroke settings.
    pub fn stroke_rect(&mut self, rect: &Rect) {
        self.stroke_path(&rect.to_path(DEFAULT_TOLERANCE));
    }

    /// Creates a builder for drawing a run of glyphs that have the same attributes.
    pub fn glyph_run(&mut self, font: &Font) -> GlyphRunBuilder<'_, Self> {
        GlyphRunBuilder::new(font.clone(), self.transform, self)
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
    pub fn set_paint(&mut self, paint: impl Into<PaintType>) {
        self.paint = paint.into();
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

    /// Reset scene to default values.
    pub fn reset(&mut self) {
        self.wide.reset();
        self.alphas.clear();
        self.line_buf.clear();
        self.tiles.reset();
        self.strip_buf.clear();

        let render_state = Self::default_render_state();
        self.transform = render_state.transform;
        self.fill_rule = render_state.fill_rule;
        self.paint = render_state.paint;
        self.stroke = render_state.stroke;
        self.blend_mode = render_state.blend_mode;
    }

    /// Get the width of the render context.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// Get the height of the render context.
    pub fn height(&self) -> u16 {
        self.height
    }

    // Assumes that `line_buf` contains the flattened path.
    fn render_path(&mut self, fill_rule: Fill, paint: Paint) {
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

        self.wide.generate(&self.strip_buf, fill_rule, paint);
    }

    fn encode_current_paint(&mut self) -> Paint {
        match self.paint.clone() {
            PaintType::Solid(s) => s.into(),
            PaintType::Image(mut i) => {
                i.transform = self.transform * i.transform;
                let paint = i.encode_into(&mut self.encoded_paints);
                if let Paint::Indexed(indexed_paint) = &paint {
                    self.upload_paints.push_back(indexed_paint.clone());
                }
                paint
            }
            _ => unimplemented!("unsupported paint type: {:?}", self.paint),
        }
    }
}

impl Scene {
    /// Prepares render data from the current context for GPU rendering
    ///
    /// This method converts the rendering context's state into a format
    /// suitable for GPU rendering, including strips and alpha values.
    pub fn prepare_render_data(&self) -> RenderData {
        let mut strips: Vec<GpuStrip> = Vec::new();
        let wide_tiles_per_row = (self.width).div_ceil(WideTile::WIDTH);
        let wide_tiles_per_col = (self.height).div_ceil(Tile::HEIGHT);
        for wide_tile_row in 0..wide_tiles_per_col {
            for wide_tile_col in 0..wide_tiles_per_row {
                let wide_tile_idx = usize::from(wide_tile_row) * usize::from(wide_tiles_per_row)
                    + usize::from(wide_tile_col);
                let wide_tile = &self.wide.tiles[wide_tile_idx];
                let wide_tile_x = wide_tile_col * WideTile::WIDTH;
                let wide_tile_y = wide_tile_row * Tile::HEIGHT;
                let bg = wide_tile.bg.to_u32();
                if bg != 0 {
                    strips.push(GpuStrip {
                        x: wide_tile_x,
                        y: wide_tile_y,
                        width: WideTile::WIDTH,
                        dense_width: 0,
                        col_idx: 0,
                        paint_type: 0,
                        paint_data: bg,
                        uv: 0,
                        x_advance: [0.0, 0.0],
                        y_advance: [0.0, 0.0],
                    });
                }
                for cmd in &wide_tile.cmds {
                    match cmd {
                        vello_common::coarse::Cmd::Fill(cmd) => {
                            let strip_x = wide_tile_x + cmd.x;
                            let strip_y = wide_tile_y;
                            let alpha_col = 0;

                            let (col_idx, paint_type, paint_data, uv, x_advance, y_advance) =
                                match &cmd.paint {
                                    Paint::Solid(color) => {
                                        let rgba = color.to_u32();
                                        (alpha_col, 0, rgba, 0, [0.0, 0.0], [0.0, 0.0])
                                    }
                                    Paint::Indexed(indexed_paint) => {
                                        let encoded_paint =
                                            self.encoded_paints.get(indexed_paint.index());
                                        match encoded_paint {
                                            Some(EncodedPaint::Image(encoded_image)) => {
                                                let start_p = encoded_image.transform
                                                    * Point::new(strip_x as f64, strip_y as f64);
                                                let u0 = start_p.x as u16;
                                                let v0 = start_p.y as u16;
                                                let extend_x = encoded_image.extends.0 as u16;
                                                let extend_y = encoded_image.extends.1 as u16;
                                                let x_advance = encoded_image.x_advance;
                                                let y_advance = encoded_image.y_advance;
                                                (
                                                    alpha_col,
                                                    2,
                                                    pack_u16s_to_u32(extend_x, extend_y),
                                                    pack_u16s_to_u32(u0, v0),
                                                    [x_advance.x as f32, x_advance.y as f32],
                                                    [y_advance.x as f32, y_advance.y as f32],
                                                )
                                            }
                                            _ => (0, 0, 0, 0, [0.0, 0.0], [0.0, 0.0]),
                                        }
                                    }
                                    _ => unimplemented!("unsupported paint type: {:?}", cmd.paint),
                                };

                            strips.push(GpuStrip {
                                x: strip_x,
                                y: strip_y,
                                width: cmd.width,
                                dense_width: 0,
                                col_idx,
                                paint_type,
                                paint_data,
                                uv,
                                x_advance,
                                y_advance,
                            });
                        }
                        vello_common::coarse::Cmd::AlphaFill(cmd) => {
                            let strip_x = wide_tile_x + cmd.x;
                            let strip_y = wide_tile_y;

                            // msg is a variable here to work around rustfmt failure
                            let msg = "GpuStrip fields use u16 and values are expected to fit within that range";
                            let alpha_col = (cmd.alpha_idx / usize::from(Tile::HEIGHT))
                                .try_into()
                                .expect(msg);

                            let (col_idx, paint_type, paint_data, uv, x_advance, y_advance) =
                                match &cmd.paint {
                                    Paint::Solid(color) => {
                                        let rgba = color.to_u32();
                                        (alpha_col, 1, rgba, 0, [0.0, 0.0], [0.0, 0.0])
                                    }
                                    Paint::Indexed(indexed_paint) => {
                                        let encoded_paint =
                                            self.encoded_paints.get(indexed_paint.index());
                                        match encoded_paint {
                                            Some(EncodedPaint::Image(encoded_image)) => {
                                                let start_p = encoded_image.transform
                                                    * Point::new(strip_x as f64, strip_y as f64);
                                                let u0 = start_p.x as u16;
                                                let v0 = start_p.y as u16;
                                                let extend_x = encoded_image.extends.0 as u16;
                                                let extend_y = encoded_image.extends.1 as u16;
                                                let x_advance = encoded_image.x_advance;
                                                let y_advance = encoded_image.y_advance;
                                                (
                                                    alpha_col,
                                                    2,
                                                    pack_u16s_to_u32(extend_x, extend_y),
                                                    pack_u16s_to_u32(u0, v0),
                                                    [x_advance.x as f32, x_advance.y as f32],
                                                    [y_advance.x as f32, y_advance.y as f32],
                                                )
                                            }
                                            _ => (alpha_col, 0, 0, 0, [0.0, 0.0], [0.0, 0.0]),
                                        }
                                    }
                                    _ => unimplemented!("unsupported paint type: {:?}", cmd.paint),
                                };

                            strips.push(GpuStrip {
                                x: strip_x,
                                y: strip_y,
                                width: cmd.width,
                                dense_width: cmd.width,
                                col_idx,
                                paint_type,
                                paint_data,
                                uv,
                                x_advance,
                                y_advance,
                            });
                        }
                        _ => {
                            unimplemented!("unsupported command: {:?}", cmd);
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

impl GlyphRenderer for Scene {
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

/// Pack two u16 values into a single u32 using byte representation.
/// The first value goes into the high 16 bits, the second into the low 16 bits.
fn pack_u16s_to_u32(a: u16, b: u16) -> u32 {
    let a_bytes = a.to_ne_bytes();
    let b_bytes = b.to_ne_bytes();
    u32::from_ne_bytes([a_bytes[0], a_bytes[1], b_bytes[0], b_bytes[1]])
}
